"""Session memory management using Lakebase (Databricks PostgreSQL)."""

import contextlib
import json
import os
import urllib.parse
from typing import Any
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Handles session message persistence using Lakebase (PostgreSQL)."""

    def __init__(
        self,
        host: str,
        instance_name: str,
    ):
        self.host = host
        self.instance_name = instance_name
        self._pool: ConnectionPool | None = None
        self.client_id = os.getenv("DATABRICKS_CLIENT_ID", None)
        self._table_ensured = False

    def _get_connection_string(self) -> str:
        """Build connection string for Lakebase.

        Supports two authentication modes:
        - SPN (production): Set DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, DATABRICKS_HOST
        - User (local testing): Uses default WorkspaceClient auth (e.g., ~/.databrickscfg)
        """
        w = WorkspaceClient()

        if self.client_id:
            # SPN authentication
            username = self.client_id
        else:
            # User authentication (local testing)
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        # Exchange auth for a short-lived Lakebase database token
        pg_credential = w.database.generate_database_credential(
            request_id=str(uuid4()), instance_names=[self.instance_name]
        )

        return f"postgresql://{username}:{pg_credential.token}@{self.host}:5432/databricks_postgres?sslmode=require"

    def _get_pool(self) -> ConnectionPool:
        """Get or create connection pool."""
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(conninfo=conn_string, min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        """Reset pool to force new credentials on next use."""
        if self._pool is not None:
            with contextlib.suppress(Exception):
                self._pool.close()
            self._pool = None
        self._table_ensured = False

    def _ensure_messages_table(self, conn: psycopg.Connection) -> None:
        """Create messages table if it doesn't exist (cached after first success)."""
        if self._table_ensured:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
            ON session_messages(session_id)
        """)
        self._table_ensured = True

    def load_messages(self, session_id: str, _retried: bool = False) -> list[dict[str, Any]]:
        """Load previous messages for a session.

        On connection errors (AdminShutdown, stale connections), resets the pool
        and retries once with a fresh connection.
        """
        try:
            with self._get_pool().connection() as conn:
                self._ensure_messages_table(conn)
                result = conn.execute(
                    """
                    SELECT message_data FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError as exc:
            self._reset_pool()
            if not _retried:
                logger.warning(f"Lakebase load_messages connection error ({exc}) — retrying with fresh pool")
                return self.load_messages(session_id, _retried=True)
            raise
        except Exception as e:
            logger.warning(f"Failed to load session messages: {e}")
            return []

    def save_messages(self, session_id: str, messages: list[dict[str, Any]], _retried: bool = False) -> None:
        """Append messages to a session.

        On connection errors (AdminShutdown, stale connections), resets the pool
        and retries once with a fresh connection.
        """
        try:
            with self._get_pool().connection() as conn:
                self._ensure_messages_table(conn)
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError as exc:
            self._reset_pool()
            if not _retried:
                logger.warning(f"Lakebase save_messages connection error ({exc}) — retrying with fresh pool")
                return self.save_messages(session_id, messages, _retried=True)
            raise
        except Exception as e:
            logger.warning(f"Failed to save session messages: {e}")

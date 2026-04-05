"""Vector search management for the Databricks Architecture Designer KB."""

import time

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from arch_designer_agent.config import ProjectConfig


class VectorSearchManager:
    """Manages vector search endpoint and index for the KB chunks table."""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: ProjectConfig object
            endpoint_name: VS endpoint name (falls back to config)
            embedding_model: Embedding model endpoint (falls back to config)
            usage_policy_id: Optional usage policy ID
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.kb_chunks_index"
        self._source_table = f"{self.catalog}.{self.schema}.kb_chunks"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = endpoints_response.get("endpoints", []) if isinstance(endpoints_response, dict) else []
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None)) == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self) -> "VectorSearchClient":
        """Create or get the kb_chunks vector search index.

        The source table (kb_chunks) must have Change Data Feed enabled,
        which is done automatically by 2.2_kb_chunking.py.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()
        source_table = self._source_table

        # Try to get existing index — only swallow "not found", re-raise endpoint errors
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception as e:
            if "NOT_FOUND" in str(e) and "endpoint" in str(e).lower():
                # Index exists in UC but its endpoint is dead — caller handles this
                raise
            logger.info(f"Index {self.index_name} not found, will create it")

        # Try to create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def _delete_stale_index_and_recreate(self, wait_timeout: int) -> None:
        """Delete an index that references a deleted endpoint, then rebuild."""
        logger.warning(
            f"Index {self.index_name} references a deleted endpoint. "
            "Deleting stale index and recreating from scratch..."
        )
        self.client.delete_index(self.endpoint_name, self.index_name)
        index = self.create_or_get_index()
        self._wait_for_index_online(index, timeout_seconds=wait_timeout)
        index.sync()
        logger.info("✓ Index sync triggered (after stale-index recovery)")

    def sync_index(self, wait_timeout: int = 600) -> None:
        """Create/get the index, wait until ONLINE, then trigger a sync.

        Args:
            wait_timeout: Seconds to wait for the index to become ONLINE
                          before raising TimeoutError.
        """
        try:
            index = self.create_or_get_index()
        except Exception as e:
            if "NOT_FOUND" in str(e) and "endpoint" in str(e).lower():
                self._delete_stale_index_and_recreate(wait_timeout)
                return
            raise

        try:
            self._wait_for_index_online(index, timeout_seconds=wait_timeout)
        except Exception as e:
            if "NOT_FOUND" in str(e) and "endpoint" in str(e).lower():
                self._delete_stale_index_and_recreate(wait_timeout)
                return
            raise

        logger.info(f"Syncing vector search index: {self.index_name}")
        try:
            index.sync()
        except Exception as e:
            if "NOT_FOUND" in str(e) and "endpoint" in str(e).lower():
                self._delete_stale_index_and_recreate(wait_timeout)
                return
            raise
        logger.info("✓ Index sync triggered")

    def _wait_for_index_online(
        self,
        index: object,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 20,
    ) -> None:
        """Poll index.describe() until the index reaches an ONLINE state.

        A newly-created index starts in PROVISIONING and is not ready to
        accept .sync() calls until it transitions to ONLINE. Calling sync()
        before the index is ready returns a 400 Bad Request.
        """
        logger.info(
            f"Waiting for index to become ONLINE (timeout={timeout_seconds}s, poll every {poll_interval_seconds}s)…"
        )
        start = time.time()
        while True:
            desc = index.describe()
            # SDK may return a dict or a typed object depending on version
            if isinstance(desc, dict):
                status = desc.get("status", {})
                state = status.get("detailed_state", "") or status.get("ready_state", "")
            else:
                status_obj = getattr(desc, "status", None)
                state = getattr(status_obj, "detailed_state", "") or getattr(status_obj, "ready_state", "")

            state_str = str(state).upper()
            logger.info(f"  Index state: {state_str}")

            if "ONLINE" in state_str:
                logger.info("✓ Index is ONLINE — ready to sync")
                return

            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Vector search index {self.index_name} did not become ONLINE "
                    f"within {timeout_seconds}s (last state: {state_str})"
                )
            time.sleep(poll_interval_seconds)

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
        query_type: str = "hybrid",
    ) -> dict:
        """Search the KB vector index.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional metadata filters
                     e.g. {"source_type": "accelerator"}
            query_type: "ANN" (semantic only) or "hybrid"

        Returns:
            Search results dictionary
        """
        index = self.client.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=[
                "chunk_id",
                "text",
                "title",
                "source_type",
                "source_repo",
                "section_header",
                "url",
            ],
            num_results=num_results,
            filters=filters,
            query_type=query_type,
        )
        return results

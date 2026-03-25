"""
KB Chunking pipeline for the Databricks Architecture Designer Agent.

knowledge_base table
   ↓ (KBProcessor.chunk_and_save)
kb_chunks table (clean text + metadata, CDF enabled)
   ↓ (VectorSearchManager - separate class, 2.4 notebook)
Vector Search Index (embeddings)
"""

import hashlib

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from arch_designer_agent.config import ProjectConfig

# ---------------------------------------------------------------------------
# Schema for kb_chunks table
# ---------------------------------------------------------------------------
KB_CHUNKS_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("source_doc_id", StringType(), True),
        StructField("source_type", StringType(), True),
        StructField("source_repo", StringType(), True),
        StructField("title", StringType(), True),
        StructField("section_header", StringType(), True),
        StructField("text", StringType(), True),
        StructField("word_count", IntegerType(), True),
        StructField("url", StringType(), True),
        StructField("topics", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ]
)

# Splitter config — tuned for databricks-gte-large-en (512-token limit)
_CHUNK_SIZE = 1500  # characters ≈ 375 tokens
_CHUNK_OVERLAP = 150  # 10 % overlap for context continuity

_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class KBProcessor:
    """
    KBProcessor handles the chunking pipeline:
    - Reads new documents from databricks_knowledge_base
    - Chunks them using a two-stage Markdown-aware strategy
    - Writes chunks to kb_chunks with Change Data Feed enabled
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize KBProcessor.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with catalog/schema settings
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema

        self.kb_table = f"{self.catalog}.{self.schema}.databricks_knowledge_base"
        self.chunks_table = f"{self.catalog}.{self.schema}.kb_chunks"

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_and_save(self) -> int:
        """
        Chunk new documents from databricks_knowledge_base and save to
        kb_chunks. Skips documents already chunked (incremental).

        Returns:
            Number of new chunks written.
        """
        existing_doc_ids = self._load_existing_chunk_doc_ids()

        kb_df = self.spark.table(self.kb_table).select(
            "doc_id",
            "content_text",
            "source_type",
            "source_repo",
            "title",
            "url",
            "topics",
            "ingestion_timestamp",
        )

        if existing_doc_ids:
            kb_df = kb_df.filter(~F.col("doc_id").isin(existing_doc_ids))

        total_docs = kb_df.count()
        logger.info(f"  {total_docs:,} new documents to chunk")

        if total_docs == 0:
            logger.info("Nothing new to chunk — kb_chunks is up to date.")
            return 0

        rows = kb_df.collect()
        chunk_rows = []
        for row in rows:
            for chunk_id, text, section_header, source_doc_id in self._chunk_doc(
                doc_id=row["doc_id"],
                content=row["content_text"] or "",
            ):
                wc = len(text.split())
                if wc < 20:  # skip trivially short chunks
                    continue
                chunk_rows.append(
                    (
                        chunk_id,
                        source_doc_id,
                        row["source_type"],
                        row["source_repo"],
                        row["title"],
                        section_header,
                        text,
                        wc,
                        row["url"],
                        row["topics"],
                        row["ingestion_timestamp"],
                    )
                )

        logger.info(
            f"Produced {len(chunk_rows):,} chunks from {total_docs:,} documents"
            f" (avg {len(chunk_rows) // max(total_docs, 1)} chunks/doc)"
        )

        chunks_df = self.spark.createDataFrame(chunk_rows, schema=KB_CHUNKS_SCHEMA)
        (
            chunks_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(self.chunks_table)
        )
        self.spark.sql(
            f"ALTER TABLE {self.chunks_table} "
            f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        logger.info(f"Appended {len(chunk_rows):,} chunks to {self.chunks_table}")
        return len(chunk_rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_chunk_doc_ids(self) -> set[str]:
        """Return doc_ids already chunked. Empty set on first run."""
        if not self.spark.catalog.tableExists(self.chunks_table):
            return set()
        ids = {
            r["source_doc_id"]
            for r in self.spark.table(self.chunks_table)
            .select("source_doc_id")
            .distinct()
            .collect()
        }
        logger.info(f"  {len(ids):,} doc_ids already chunked (incremental skip)")
        return ids

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        return hashlib.md5(f"{doc_id}:chunk:{index}".encode()).hexdigest()

    def _chunk_doc(self, doc_id: str, content: str) -> list[tuple[str, str, str, str]]:
        """Two-stage chunk a single document.

        Returns list of (chunk_id, text, section_header, doc_id).
        """
        results: list[tuple[str, str, str, str]] = []
        has_headers = any(line.startswith("#") for line in content.splitlines()[:50])

        if has_headers:
            md_docs = self._md_splitter.split_text(content)
            sub_docs = self._char_splitter.split_documents(md_docs)
            for i, d in enumerate(sub_docs):
                header = " > ".join(
                    v for k in ("h1", "h2", "h3", "h4") if (v := d.metadata.get(k))
                )
                results.append(
                    (self._make_chunk_id(doc_id, i), d.page_content, header, doc_id)
                )
        else:
            for i, text in enumerate(self._char_splitter.split_text(content)):
                results.append((self._make_chunk_id(doc_id, i), text, "", doc_id))

        return results

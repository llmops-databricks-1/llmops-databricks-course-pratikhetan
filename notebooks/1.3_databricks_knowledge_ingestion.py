# Databricks notebook source
"""
Databricks Architecture Designer — Knowledge Base Ingestion

Builds a unified knowledge base across three source types, all written to
`{catalog}.{schema}.databricks_knowledge_base`:

  Section 1 — Solution Accelerators
      193 public repos from `databricks-industry-solutions` GitHub org.
      Filters: active, stars >= 2, updated within 3 years, word_count >= 150.
      source_type = "accelerator"

  Section 2 — Open-Source Docs
      Architecture-relevant docs from four GitHub repos:
        • mlflow/mlflow           → tracking, model registry, serving, tracing, LLMs
        • delta-io/delta          → best practices, streaming, optimisations
        • databricks/databricks-sdk-py → jobs, clusters, model serving, vector search
        • apache/spark            → structured streaming, SQL tuning, cluster design
      Path-whitelist filtered; stubs (< 200 words) and giant API refs
      (> 15k words) excluded.
      source_type = "oss_docs"

  Section 3 — Databricks Blog architecture posts
      Fetched via RSS feeds, filtered by title keywords + recency (< 3 years),
      full text extracted with trafilatura.
      source_type = "blog"
"""

import hashlib
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import requests
import trafilatura
from delta.tables import DeltaTable
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from arch_designer_agent.config import get_env, load_config

# COMMAND ----------
# -- Config --------------------------------------------------------------------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
KB_TABLE = "databricks_knowledge_base"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{KB_TABLE}"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Target table: {FULL_TABLE}")

# COMMAND ----------
# -- GitHub auth (optional) ---------------------------------------------------
# Without token: 60 req/hr.  With token: 5,000 req/hr.
# Store in Databricks Secrets: scope=llmops_course  key=github_token

try:
    GITHUB_TOKEN = dbutils.secrets.get(scope="llmops_course", key="github_token")  # noqa: F821  # type: ignore[name-defined]
    logger.info("GitHub token loaded from secrets")
except Exception:
    GITHUB_TOKEN = None
    logger.warning("No GitHub token — unauthenticated (60 req/hr)")

_GH_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "llmops-course"}
if GITHUB_TOKEN:
    _GH_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

GH_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

# COMMAND ----------
# -- Shared helpers ------------------------------------------------------------


def _make_doc_id(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()


def _gh_get(
    url: str, params: dict | None = None, max_retries: int = 4
) -> dict | list | None:
    """GitHub API GET with exponential backoff on rate-limit responses."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=_GH_HEADERS, params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 403, 503):
            reset = resp.headers.get("X-RateLimit-Reset")
            wait = (
                max(int(reset) - int(time.time()) + 2, 5) if reset else 10 * (2**attempt)
            )
            logger.warning(
                f"HTTP {resp.status_code} — sleeping {wait}s (attempt {attempt + 1})"
            )
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    return None


def _raw_get(url: str) -> str | None:
    """Fetch raw file content (README, markdown doc) from GitHub raw URLs."""
    try:
        resp = requests.get(url, headers={"User-Agent": "llmops-course"}, timeout=15)
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


def _clean_markdown(text: str, max_chars: int = 50_000) -> str:
    """Strip image lines, collapse whitespace, cap length."""
    lines = [line for line in text.splitlines() if not line.strip().startswith("![")]
    return "\n".join(lines)[:max_chars]


def _slugify_title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.replace("_", "-").split("-"))


# Shared Delta schema
KB_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("source_type", StringType(), True),
        StructField("source_repo", StringType(), True),
        StructField("repo_name", StringType(), True),
        StructField("title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("topics", StringType(), True),
        StructField("language", StringType(), True),
        StructField("stars", IntegerType(), True),
        StructField("content_text", StringType(), True),
        StructField("word_count", IntegerType(), True),
        StructField("url", StringType(), True),
        StructField("last_updated", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ]
)

_NOW = datetime.now(UTC).isoformat()


# COMMAND ----------
# =============================================================================
# SECTION 1 — Solution Accelerators (databricks-industry-solutions)
# =============================================================================

GH_ORG = "databricks-industry-solutions"

# Quality filters
_ACC_MIN_STARS = 0
_ACC_MIN_WORDS = 100
_ACC_MAX_AGE_YRS = 4

_CUTOFF_DATE = (datetime.now(UTC) - timedelta(days=_ACC_MAX_AGE_YRS * 365)).isoformat()


def _list_org_repos(org: str) -> list[dict]:
    repos, page = [], 1
    while True:
        batch = _gh_get(
            f"{GH_API}/orgs/{org}/repos",
            params={"type": "public", "per_page": 100, "page": page},
        )
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)
    logger.info(f"  Found {len(repos)} repos in {org}")
    return repos


def _fetch_readme(repo_name: str, branch: str = "main") -> str | None:
    for br in [branch, "master", "main"]:
        for fname in ["README.md", "readme.md", "Readme.md"]:
            text = _raw_get(f"{RAW_BASE}/{GH_ORG}/{repo_name}/{br}/{fname}")
            if text:
                return text
    return None


def _process_accelerator(repo: dict) -> dict | None:
    name = repo.get("name", "")
    stars = repo.get("stargazers_count") or 0
    updated = repo.get("pushed_at") or repo.get("updated_at") or ""
    topics = repo.get("topics") or []

    # Quality filters — org listing already guarantees Databricks repos
    if repo.get("archived"):
        return None
    if repo.get("fork"):
        return None
    if updated < _CUTOFF_DATE:
        return None

    branch = repo.get("default_branch") or "main"
    readme = _fetch_readme(name, branch) or repo.get("description") or ""
    cleaned = _clean_markdown(readme)
    wc = len(cleaned.split())

    if wc < _ACC_MIN_WORDS:
        return None

    return {
        "doc_id": _make_doc_id(f"acc:{name}"),
        "source_type": "accelerator",
        "source_repo": f"{GH_ORG}/{name}",
        "repo_name": name,
        "title": _slugify_title(name),
        "description": repo.get("description") or "",
        "topics": ", ".join(topics),
        "language": repo.get("language") or "",
        "stars": stars,
        "content_text": cleaned,
        "word_count": wc,
        "url": repo.get("html_url") or f"https://github.com/{GH_ORG}/{name}",
        "last_updated": updated,
        "ingestion_timestamp": _NOW,
    }


logger.info("=== Section 1: Solution Accelerators ===")
raw_repos = _list_org_repos(GH_ORG)
acc_docs: list[dict] = []

with ThreadPoolExecutor(max_workers=10) as exe:
    futures = {exe.submit(_process_accelerator, r): r["name"] for r in raw_repos}
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                acc_docs.append(result)
        except Exception as exc:
            logger.warning(f"  Accelerator error {futures[future]}: {exc}")

logger.info(
    f"Section 1 complete: {len(acc_docs)} accelerators (filtered from {len(raw_repos)})"
)

# COMMAND ----------
# =============================================================================
# SECTION 2 — Open-Source Docs (MLflow, Delta, SDK, Spark)
# =============================================================================

# ---------------------------------------------------------------------------
# Two ingestion strategies for OSS docs:
#   A) Direct URL list  — for sites with stable rendered HTML (Delta, Spark)
#   B) GitHub traversal — for repos with clean markdown sources (MLflow, SDK)
# ---------------------------------------------------------------------------

# --- Strategy A: Direct URL fetch (trafilatura) ----------------------------
# Delta Lake OSS docs  (docs.delta.io) — Sphinx-generated static HTML
_DELTA_URLS = [
    ("Delta Lake Quick Start", "https://docs.delta.io/latest/quick-start.html"),
    ("Delta Lake Best Practices", "https://docs.delta.io/latest/best-practices.html"),
    (
        "Delta Lake Batch Reads and Writes",
        "https://docs.delta.io/latest/delta-batch.html",
    ),
    ("Delta Lake Streaming", "https://docs.delta.io/latest/delta-streaming.html"),
    (
        "Delta Lake Table Utility Commands",
        "https://docs.delta.io/latest/delta-utility.html",
    ),
    ("Delta Lake Optimizations", "https://docs.delta.io/latest/optimizations-oss.html"),
    (
        "Delta Lake Change Data Feed",
        "https://docs.delta.io/latest/delta-change-data-feed.html",
    ),
    (
        "Delta Lake Concurrency Control",
        "https://docs.delta.io/latest/concurrency-control.html",
    ),
    (
        "Delta Lake Column Mapping",
        "https://docs.delta.io/latest/delta-column-mapping.html",
    ),
    ("Delta Lake Constraints", "https://docs.delta.io/latest/delta-constraints.html"),
    ("Delta Lake Uniform", "https://docs.delta.io/latest/delta-uniform.html"),
    (
        "Delta Lake Storage Configuration",
        "https://docs.delta.io/latest/delta-storage.html",
    ),
]

# Databricks public docs — architecture & optimization
_DATABRICKS_DOCS_URLS = [
    (
        "Databricks Performance Tuning",
        "https://docs.databricks.com/en/optimizations/performance-tuning.html",
    ),
    (
        "Databricks Cluster Sizing",
        "https://docs.databricks.com/en/clusters/cluster-sizing.html",
    ),
    (
        "Databricks SQL Optimization",
        "https://docs.databricks.com/en/sql/performance.html",
    ),
    (
        "Databricks Delta Lake Optimization",
        "https://docs.databricks.com/en/delta/optimizations.html",
    ),
    ("Databricks Best Practices", "https://docs.databricks.com/en/best-practices.html"),
    ("Databricks Monitoring", "https://docs.databricks.com/en/monitoring/index.html"),
    (
        "Databricks Architecture Overview",
        "https://docs.databricks.com/en/architecture/index.html",
    ),
    (
        "Databricks Security Architecture",
        "https://docs.databricks.com/en/security/index.html",
    ),
    (
        "Databricks Lakehouse Design",
        "https://docs.databricks.com/en/lakehouse/index.html",
    ),
    (
        "Databricks MLflow Architecture",
        "https://docs.databricks.com/en/mlflow/index.html",
    ),
]

# Apache Spark docs  (spark.apache.org) — server-rendered HTML, trafilatura works well
# structured-streaming-programming-guide.html and sql-data-sources.html return None
# from trafilatura (likely too large / extraction fails); replaced with sub-pages.
_SPARK_URLS = [
    (
        "Spark SQL Performance Tuning",
        "https://spark.apache.org/docs/latest/sql-performance-tuning.html",
    ),
    ("Spark Configuration", "https://spark.apache.org/docs/latest/configuration.html"),
    (
        "Spark SQL Programming Guide",
        "https://spark.apache.org/docs/latest/sql-programming-guide.html",
    ),
    (
        "Spark SQL Data Sources Avro",
        "https://spark.apache.org/docs/latest/sql-data-sources-avro.html",
    ),
    (
        "Spark Cluster Mode Overview",
        "https://spark.apache.org/docs/latest/cluster-overview.html",
    ),
    (
        "Submitting Spark Applications",
        "https://spark.apache.org/docs/latest/submitting-applications.html",
    ),
    (
        "Spark Monitoring and Instrumentation",
        "https://spark.apache.org/docs/latest/monitoring.html",
    ),
    ("Spark Tuning Guide", "https://spark.apache.org/docs/latest/tuning.html"),
    (
        "Spark SQL Data Sources Parquet",
        "https://spark.apache.org/docs/latest/sql-data-sources-parquet.html",
    ),
    (
        "Spark SQL Data Sources JSON",
        "https://spark.apache.org/docs/latest/sql-data-sources-json.html",
    ),
    (
        "Spark SQL Data Sources JDBC",
        "https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html",
    ),
    ("Spark MLlib Guide", "https://spark.apache.org/docs/latest/ml-guide.html"),
    (
        "Spark on Kubernetes",
        "https://spark.apache.org/docs/latest/running-on-kubernetes.html",
    ),
    ("Spark on YARN", "https://spark.apache.org/docs/latest/running-on-yarn.html"),
    (
        "RDD Programming Guide",
        "https://spark.apache.org/docs/latest/rdd-programming-guide.html",
    ),
    ("Spark Quick Start", "https://spark.apache.org/docs/latest/quick-start.html"),
]

# MLflow docs  (mlflow.org) — only python_api/* are static Sphinx HTML and reliably
# extracted by trafilatura.  Concept pages (tracking.html, model-registry.html, etc.)
# are React-rendered and return None.  Narrative docs come via GitHub traversal below.
_MLFLOW_URLS = [
    ("MLflow Module API", "https://mlflow.org/docs/latest/python_api/mlflow.html"),
    (
        "MLflow Tracking API",
        "https://mlflow.org/docs/latest/python_api/mlflow.tracking.html",
    ),
    ("MLflow Models API", "https://mlflow.org/docs/latest/python_api/mlflow.models.html"),
    ("MLflow Client API", "https://mlflow.org/docs/latest/python_api/mlflow.client.html"),
    ("MLflow Pyfunc", "https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html"),
    (
        "MLflow Spark Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.spark.html",
    ),
    (
        "MLflow PyTorch Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html",
    ),
    (
        "MLflow Sklearn Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html",
    ),
    (
        "MLflow TensorFlow Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html",
    ),
    (
        "MLflow XGBoost Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.xgboost.html",
    ),
    (
        "MLflow LightGBM Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html",
    ),
    (
        "MLflow LangChain Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.langchain.html",
    ),
    (
        "MLflow OpenAI Integration",
        "https://mlflow.org/docs/latest/python_api/mlflow.openai.html",
    ),
    ("MLflow Data API", "https://mlflow.org/docs/latest/python_api/mlflow.data.html"),
    ("MLflow Metrics", "https://mlflow.org/docs/latest/python_api/mlflow.metrics.html"),
    (
        "MLflow Artifacts API",
        "https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html",
    ),
]

_DIRECT_URL_SOURCES = [
    ("delta-io/delta", _DELTA_URLS),
    ("apache/spark", _SPARK_URLS),
    ("mlflow/mlflow", _MLFLOW_URLS),
    ("databricks/docs", _DATABRICKS_DOCS_URLS),
]

_OSS_MIN_WORDS = 100
_OSS_MAX_WORDS = 50_000  # large pages like Spark config / structured streaming are 20k+


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_url_doc(title: str, url: str, source_repo: str) -> dict | None:
    """Fetch a single doc page and return a knowledge base record.

    Tries trafilatura first (fast path), then falls back to requests with
    browser headers so sites that block the trafilatura UA still work.
    Uses favor_recall=True so content-light pages still get extracted.
    """
    # --- fast path: trafilatura built-in downloader ---
    downloaded = trafilatura.fetch_url(url)
    # --- fallback: requests with a browser User-Agent ---
    if not downloaded:
        try:
            r = requests.get(url, headers=_BROWSER_HEADERS, timeout=20)
            if r.status_code == 200:
                downloaded = r.text
        except Exception:
            pass
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )
    if not text:
        return None
    wc = len(text.split())
    if wc < _OSS_MIN_WORDS or wc > _OSS_MAX_WORDS:
        return None
    return {
        "doc_id": _make_doc_id(f"oss:{url}"),
        "source_type": "oss_docs",
        "source_repo": source_repo,
        "repo_name": source_repo.split("/")[-1],
        "title": title,
        "description": f"{source_repo} documentation: {title}",
        "topics": "",
        "language": "",
        "stars": 0,
        "content_text": text[:50_000],
        "word_count": wc,
        "url": url,
        "last_updated": "",
        "ingestion_timestamp": _NOW,
    }


# --- Strategy B: GitHub traversal for MLflow narrative docs + SDK ----------
_GH_SOURCES: dict[str, dict] = {
    "mlflow/mlflow": {
        "branch": "master",
        # try docs/ first; Sphinx RST may be at either level
        "scan_dirs": ["docs", "docs/source"],
        "keywords": [
            "tracking",
            "model",
            "registry",
            "deploy",
            "serving",
            "evaluation",
            "tracing",
            "llm",
            "langchain",
            "gateway",
            "autolog",
            "recipes",
            "projects",
            "quickstart",
            "concept",
            "guide",
            "tutorial",
            "intro",
            "configuration",
            "artifact",
            "experiment",
            "run",
            "metric",
            "param",
            "signature",
            "flavor",
            "plugin",
            "search",
            "backend",
            "sklearn",
            "pytorch",
            "tensorflow",
            "spark",
            "xgboost",
            "lightgbm",
            "transformers",
            "openai",
            "prompt",
            "chat",
            "mlflow",
        ],
        "exts": {".rst", ".md"},
    },
    "databricks/databricks-sdk-py": {
        "branch": "main",
        "scan_dirs": ["docs"],
        "keywords": [
            "jobs",
            "clusters",
            "serving",
            "vector",
            "secrets",
            "workflows",
            "pipelines",
            "experiments",
            "models",
            "permissions",
            "sql",
            "compute",
            "catalog",
            "files",
            "apps",
            "marketplace",
            "dashboards",
            "unity",
            "repos",
            "tokens",
            "warehouses",
            "alerts",
            "queries",
            "instance",
            "libraries",
            "policies",
            "pools",
            "endpoints",
            "tables",
        ],
        "exts": {".rst", ".md"},
    },
}

_SKIP_NAMES = {
    "changelog",
    "contributing",
    "license",
    "migration",
    "release-notes",
    "release_notes",
    "history",
    "authors",
    "roadmap",
}


def _list_repo_files(repo: str, branch: str, directory: str) -> tuple[list[dict], str]:
    """List doc files in a GitHub dir.  Tries the configured branch, then master/main
    as fallbacks so the call succeeds even when the default branch name is uncertain.
    Returns (files, resolved_branch).
    """
    # deduplicated, order preserved
    candidates = list(dict.fromkeys([branch, "master", "main"]))
    for br in candidates:
        data = _gh_get(f"{GH_API}/repos/{repo}/contents/{directory}", params={"ref": br})
        if not isinstance(data, list):
            continue  # 404 or error — try next branch
        files = []
        for item in data:
            if item.get("type") == "file":
                files.append(item)
            elif item.get("type") == "dir":
                sub = _gh_get(
                    f"{GH_API}/repos/{repo}/contents/{item['path']}", params={"ref": br}
                )
                if isinstance(sub, list):
                    files.extend(f for f in sub if f.get("type") == "file")
        return files, br
    return [], branch


def _fetch_gh_docs(repo: str, cfg: dict) -> list[dict]:
    branch, keywords, exts, scan_dirs = (
        cfg["branch"],
        cfg["keywords"],
        cfg["exts"],
        cfg["scan_dirs"],
    )
    docs = []
    # dedup across overlapping scan_dirs
    seen_paths: set[str] = set()
    for directory in scan_dirs:
        raw_files, resolved_br = _list_repo_files(repo, branch, directory)
        ext_ok = [
            f
            for f in raw_files
            if ("." + f.get("path", "").rsplit(".", 1)[-1].lower()) in exts
        ]
        kw_ok = [
            f
            for f in ext_ok
            if not any(
                skip in f.get("path", "").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
                for skip in _SKIP_NAMES
            )
            and any(kw in f.get("path", "").lower() for kw in keywords)
        ]
        logger.info(
            f"      {directory} (branch={resolved_br}): "
            f"{len(raw_files)} files → {len(ext_ok)} ext-match → {len(kw_ok)} kw-match"
        )
        for f in kw_ok:
            path = f.get("path", "")
            if path in seen_paths:
                continue
            seen_paths.add(path)
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            content = _raw_get(
                f.get("download_url") or f"{RAW_BASE}/{repo}/{resolved_br}/{path}"
            )
            if not content:
                continue
            cleaned = _clean_markdown(content)
            wc = len(cleaned.split())
            if wc < _OSS_MIN_WORDS or wc > _OSS_MAX_WORDS:
                continue
            docs.append(
                {
                    "doc_id": _make_doc_id(f"oss:{repo}:{path}"),
                    "source_type": "oss_docs",
                    "source_repo": repo,
                    "repo_name": repo.split("/")[-1],
                    "title": _slugify_title(stem),
                    "description": f"{repo} documentation: {_slugify_title(stem)}",
                    "topics": "",
                    "language": "",
                    "stars": 0,
                    "content_text": cleaned,
                    "word_count": wc,
                    "url": f.get("html_url")
                    or f"https://github.com/{repo}/blob/{resolved_br}/{path}",
                    "last_updated": "",
                    "ingestion_timestamp": _NOW,
                }
            )
        time.sleep(0.2)
    return docs


# --- Run Section 2 ---------------------------------------------------------
logger.info("=== Section 2: OSS Docs ===")
oss_docs: list[dict] = []

# Strategy A — direct URL fetch (Spark + MLflow via trafilatura)
for source_repo, url_list in _DIRECT_URL_SOURCES:
    logger.info(f"  Fetching direct URLs: {source_repo} ({len(url_list)} pages)")
    for title, url in url_list:
        try:
            doc = _fetch_url_doc(title, url, source_repo)
            if doc:
                oss_docs.append(doc)
                logger.info(f"    ✓ {title} ({doc['word_count']} words)")
            else:
                logger.warning(f"    ✗ {title}")
        except Exception as exc:
            logger.warning(f"    ✗ {title}: {exc}")
        time.sleep(0.5)

# Strategy B — GitHub traversal (Delta Lake + MLflow narrative + SDK)
for repo_name, repo_cfg in _GH_SOURCES.items():
    logger.info(f"  Fetching GitHub docs: {repo_name}")
    try:
        repo_docs = _fetch_gh_docs(repo_name, repo_cfg)
        oss_docs.extend(repo_docs)
        logger.info(f"    → {len(repo_docs)} docs from {repo_name}")
    except Exception as exc:
        logger.warning(f"    Failed {repo_name}: {exc}")

logger.info(f"Section 2 complete: {len(oss_docs)} OSS docs")

# COMMAND ----------
# =============================================================================
# SECTION 3 — Databricks Blog architecture posts (RSS + trafilatura)
# =============================================================================

_BLOG_RSS_FEEDS = [
    "https://www.databricks.com/feed",
    "https://www.databricks.com/blog/category/engineering-blog/feed",
    "https://www.databricks.com/blog/category/product/feed",
]

_BLOG_KEYWORDS = [
    "architecture",
    "design",
    "pipeline",
    "lakehouse",
    "llm",
    "agent",
    "rag",
    "mlops",
    "deploy",
    "streaming",
    "delta",
    "vector search",
    "best practice",
    "reference",
    "pattern",
    "workflow",
    "fine-tun",
    "inference",
    "databricks",
    "mosaic",
    "unity catalog",
    "spark",
    "mlflow",
    "feature store",
    "automl",
    "model serving",
    "real-time",
    "batch",
    "etl",
    "data engineering",
    "data science",
    "foundation model",
    "embedding",
    "retrieval",
    "chatbot",
    "copilot",
    "governance",
    "lineage",
    "catalog",
    "lakehouse",
    "genie",
    "photon",
    "serverless",
    "dlt",
    "live tables",
    "medallion",
    "bronze",
    "silver",
    "gold",
    "data mesh",
    "platform",
]

_BLOG_MAX_AGE_DAYS = 3 * 365  # only posts from last 3 years
_BLOG_MIN_WORDS = 300
_BLOG_MAX_WORKERS = 5
_CUTOFF_BLOG = datetime.now(UTC) - timedelta(days=_BLOG_MAX_AGE_DAYS)


def _parse_rss(url: str) -> list[dict]:
    """Fetch and parse an RSS feed, return list of {title, link, pub_date}."""
    try:
        resp = requests.get(url, headers={"User-Agent": "llmops-course"}, timeout=15)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if title and link:
                items.append({"title": title, "link": link, "pub_date": pub_date})
        return items
    except Exception as exc:
        logger.warning(f"  RSS parse failed {url}: {exc}")
        return []


def _parse_pub_date(date_str: str) -> datetime | None:
    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"]:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _title_matches_keywords(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _BLOG_KEYWORDS)


def _fetch_blog_post(item: dict) -> dict | None:
    url = item["link"]
    title = item["title"]
    pub_date = item["pub_date"]

    # Recency filter
    dt = _parse_pub_date(pub_date)
    if dt and dt < _CUTOFF_BLOG:
        return None

    # Title keyword filter
    if not _title_matches_keywords(title):
        return None

    # Fetch full text
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=False, no_fallback=False
    )
    if not text:
        return None

    wc = len(text.split())
    if wc < _BLOG_MIN_WORDS:
        return None

    return {
        "doc_id": _make_doc_id(f"blog:{url}"),
        "source_type": "blog",
        "source_repo": "databricks.com/blog",
        "repo_name": "databricks-blog",
        "title": title,
        "description": title,
        "topics": "",
        "language": "",
        "stars": 0,
        "content_text": text[:50_000],
        "word_count": wc,
        "url": url,
        "last_updated": pub_date,
        "ingestion_timestamp": _NOW,
    }


logger.info("=== Section 3: Databricks Blog ===")

# Collect + deduplicate feed items
all_items: list[dict] = []
seen_links: set[str] = set()
for feed_url in _BLOG_RSS_FEEDS:
    for item in _parse_rss(feed_url):
        if item["link"] not in seen_links:
            seen_links.add(item["link"])
            all_items.append(item)

logger.info(
    f"  {len(all_items)} unique RSS items collected across {len(_BLOG_RSS_FEEDS)} feeds"
)

blog_docs: list[dict] = []
with ThreadPoolExecutor(max_workers=_BLOG_MAX_WORKERS) as exe:
    futures = {exe.submit(_fetch_blog_post, item): item["link"] for item in all_items}
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                blog_docs.append(result)
        except Exception as exc:
            logger.debug(f"  Blog fetch error {futures[future]}: {exc}")

logger.info(
    f"Section 3 complete: {len(blog_docs)} blog posts (from {len(all_items)} candidates)"
)

# COMMAND ----------
# =============================================================================
# SECTION 4 — Combine + Write Delta (MERGE, idempotent)
# =============================================================================

all_docs = acc_docs + oss_docs + blog_docs
logger.info(
    f"Total documents: {len(all_docs)} "
    f"(accelerators={len(acc_docs)}, oss_docs={len(oss_docs)},"
    f" blog={len(blog_docs)})"
)

new_df = spark.createDataFrame(all_docs, schema=KB_SCHEMA)

if not spark.catalog.tableExists(FULL_TABLE):
    logger.info(f"Creating table {FULL_TABLE}")
    (
        new_df.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(FULL_TABLE)
    )
else:
    logger.info(f"Merging into {FULL_TABLE}")
    (
        DeltaTable.forName(spark, FULL_TABLE)
        .alias("existing")
        .merge(new_df.alias("incoming"), "existing.doc_id = incoming.doc_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

total = spark.table(FULL_TABLE).count()
logger.info(f"Table {FULL_TABLE}: {total} rows total")

# COMMAND ----------
# =============================================================================
# SECTION 5 — Summary Stats
# =============================================================================

from pyspark.sql.functions import avg, count  # noqa: E402
from pyspark.sql.functions import max as _max  # noqa: E402
from pyspark.sql.functions import sum as _sum  # noqa: E402

stats = (
    spark.table(FULL_TABLE)
    .groupBy("source_type")
    .agg(
        count("*").alias("doc_count"),
        _sum("word_count").alias("total_words"),
        avg("word_count").alias("avg_words"),
        _max("word_count").alias("max_words"),
    )
    .orderBy("source_type")
)

logger.info("Knowledge base breakdown:")
for row in stats.collect():
    logger.info(
        f"  [{row['source_type']:<12}]  docs={row['doc_count']:>4}  "
        f"words={row['total_words']:>8,}  avg={int(row['avg_words']):>5}"
    )

total_words = spark.table(FULL_TABLE).selectExpr("SUM(word_count) as t").first()["t"]
logger.info(f"Total knowledge base: {total_words:,} words across {total} documents")

# Databricks notebook source
import os

import mlflow
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.config import ProjectConfig, get_env, load_config
from arch_designer_agent.evaluation import (
    architectural_clarity_guideline,
    cites_databricks_service,
    databricks_scope_guideline,
    grounded_in_evidence_guideline,
    response_length_check,
)

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../../project_config.yml", env)
mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

catalog = cfg.catalog
schema = cfg.db_schema

traces_table = f"{catalog}.{schema}.arch_designer_agent_traces"
aggregated_view = f"{catalog}.{schema}.arch_agent_traces_aggregated"

# COMMAND ----------
# Get traces not yet evaluated

new_traces_df = spark.sql(f"""
    SELECT
        t.trace_id,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(t.response, '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text
    FROM {traces_table} t
    WHERE tags['model_serving_endpoint_name']
            = 'arch-designer-agent-dev'
      AND (t.assessments IS NULL OR size(t.assessments) = 0)
""")

traces_pdf = new_traces_df.toPandas()
logger.info(f"New traces to evaluate: {len(traces_pdf)}")

# COMMAND ----------
# Build eval input

eval_pdf = pd.DataFrame(
    {
        "trace_id": traces_pdf["trace_id"],
        "inputs": traces_pdf["request_preview"].apply(lambda x: {"query": x}),
        "outputs": traces_pdf["response_text"],
    }
)

# COMMAND ----------
# Run fast custom scorers on all traces and log feedback

fast_result = mlflow.genai.evaluate(
    data=eval_pdf[["inputs", "outputs"]],
    scorers=[response_length_check, cites_databricks_service],
)

for trace_id, assessments in zip(
    eval_pdf["trace_id"],
    fast_result.result_df["assessments"],
    strict=True,
):
    for a in assessments:
        name = a["assessment_name"]
        val = a["feedback"]["value"]
        mlflow.log_feedback(
            trace_id=trace_id,
            name=name,
            value=val,
        )

logger.info(f"Logged response_length_check / cites_databricks_service for {len(eval_pdf)} traces")

# COMMAND ----------
# Run LLM-judge scorers on a 10% sample and log feedback

sample_size = max(1, int(len(eval_pdf) * 0.1))
sampled_pdf = eval_pdf.sample(n=sample_size)
logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge evaluation")

llm_result = mlflow.genai.evaluate(
    data=sampled_pdf[["inputs", "outputs"]],
    scorers=[
        architectural_clarity_guideline,
        databricks_scope_guideline,
        grounded_in_evidence_guideline,
    ],
)

for trace_id, assessments in zip(
    sampled_pdf["trace_id"],
    llm_result.result_df["assessments"],
    strict=True,
):
    for a in assessments:
        name = a["assessment_name"]
        val = a["feedback"]["value"]
        mlflow.log_feedback(
            trace_id=trace_id,
            name=name,
            value=val,
        )

logger.info(f"Logged LLM-judge scores for {len(sampled_pdf)} traces")

# COMMAND ----------
# Create aggregated view
# NOTE: verify assessment field name with:
#   SELECT assessments FROM {traces_table} LIMIT 1

spark.sql(f"""
    CREATE OR REPLACE VIEW {aggregated_view} AS
    SELECT
        t.trace_id,
        t.request_time,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(t.response, '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text,
        CAST(t.execution_duration_ms / 1000.0 AS DOUBLE)
            AS latency_seconds,
        COUNT(IF(s.name = 'call_llm', 1, NULL))
            AS call_llm_exec_count,
        COUNT(IF(s.name = 'execute_tool', 1, NULL))
            AS tool_call_count,
        CAST(SUM(
            IF(
                s.name = 'call_llm',
                CAST(
                    get_json_object(
                        get_json_object(
                            s.attributes['mlflow.spanOutputs'],
                            '$.usage'
                        ),
                        '$.total_tokens'
                    ) AS INT
                ),
                0
            )
        ) AS LONG) AS total_tokens_used,
        current_timestamp() AS processed_ts,
        CASE
            WHEN element_at(
                filter(t.assessments, a -> a.name = 'response_length_check'),
                1
            ).string_value = 'true' THEN 1 ELSE 0
        END AS response_length_check,
        CASE
            WHEN element_at(
                filter(t.assessments, a -> a.name = 'cites_databricks_service'),
                1
            ).string_value = 'true' THEN 1 ELSE 0
        END AS cites_databricks_service,
        CASE
            WHEN element_at(
                filter(t.assessments, a -> a.name = 'architectural_clarity'),
                1
            ).string_value = 'Pass' THEN 1 ELSE 0
        END AS architectural_clarity,
        CASE
            WHEN element_at(
                filter(t.assessments, a -> a.name = 'stays_in_databricks_scope'),
                1
            ).string_value = 'Pass' THEN 1 ELSE 0
        END AS stays_in_databricks_scope,
        CASE
            WHEN element_at(
                filter(t.assessments, a -> a.name = 'grounded_in_evidence'),
                1
            ).string_value = 'Pass' THEN 1 ELSE 0
        END AS grounded_in_evidence
    FROM {traces_table} t
    LATERAL VIEW explode(spans) AS s
    WHERE tags['model_serving_endpoint_name']
            = 'arch-designer-agent-dev'
    GROUP BY t.trace_id, t.request_time,
             t.execution_duration_ms, t.request_preview,
             t.response, t.assessments
""")

logger.info(f"View {aggregated_view} created")

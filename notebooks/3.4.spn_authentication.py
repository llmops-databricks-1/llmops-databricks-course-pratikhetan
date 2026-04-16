# Databricks notebook source
"""
Lakebase SPN Permission Setup — run ONCE per environment before first deployment.

Pre-requisites (already done):
  - SPN created for each env
  - client_id / client_secret stored in secret scopes: dev_SPN, acc_SPN, prd_SPN
  - Lakebase instance "arch-agent-memory" created

What this notebook does:
  1. Reads the existing SPN client_id from the env secret scope
  2. Grants CAN_USE on the Lakebase instance to the SPN
  3. Prints the Postgres SQL to run manually in the Lakebase SQL Editor

Run once for dev, then again (changing env below) for acc.

--- ORIGINAL FLOW (if SPN does not exist yet) ---
If you need to create a new SPN from scratch, uncomment the block below.
It will: create SPN, generate OAuth secret, store in secret scope, grant CAN_USE.
"""

# --- ORIGINAL: Create new SPN + store secrets (skip if SPN already exists) ---
# from requests.auth import HTTPBasicAuth
#
# # Fill these in before uncommenting:
# admin_client_id = "your-admin-client-id"
# admin_client_secret = "your-admin-client-secret"
# account_id = "your-account-id"
# account_host = "https://accounts.cloud.databricks.com"
#
# # Get account-level token
# token = requests.post(
#     f"{account_host}/oidc/accounts/{account_id}/v1/token",
#     auth=HTTPBasicAuth(admin_client_id, admin_client_secret),
#     data={"grant_type": "client_credentials", "scope": "all-apis"},
# ).json()["access_token"]
#
# # Create service principal + OAuth secret
# sp = w.service_principals.create(display_name="lakebase-sp-arch-designer")
# secret_resp = requests.post(
#     f"{account_host}/api/2.0/accounts/{account_id}/servicePrincipals/{sp.id}/credentials/secrets",
#     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
# )
# secret_resp.raise_for_status()
# new_client_id = sp.application_id
# new_client_secret = secret_resp.json()["secret"]
#
# # Store credentials in secret scope
# try:
#     w.secrets.create_scope(scope=scope_name)
# except Exception:
#     pass  # scope already exists
# w.secrets.put_secret(scope=scope_name, key="client_id", string_value=new_client_id)
# w.secrets.put_secret(scope=scope_name, key="client_secret", string_value=new_client_secret)

import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# Set this to the environment you are setting up: dev | acc | prd
# Run the full notebook once per environment.
env = "dev"

scope_name = f"{env}_SPN"
instance_name = "arch-agent-memory"

# Read the existing SPN client_id from the already-populated secret scope
client_id = dbutils.secrets.get(scope_name, "client_id")  # noqa: F821
print(f"Using SPN client_id from scope '{scope_name}': {client_id[:8]}...")

# COMMAND ----------

# Step 0: Create Lakebase instance if it doesn't exist yet
from databricks.sdk.errors import NotFound
from databricks.sdk.service.database import DatabaseInstance

try:
    instance = w.database.get_database_instance(instance_name)
    print(f"Lakebase instance '{instance_name}' already exists — skipping creation")
except NotFound:
    print(f"Creating Lakebase instance '{instance_name}'...")
    # usage_policy_id is optional — only required if your workspace enforces billing policies.
    # If creation fails, find your policy ID with:
    #   for p in w.database.list_database_instance_usage_policies(): print(p)
    instance = w.database.create_database_instance(DatabaseInstance(name=instance_name, capacity="CU_1")).result()
    print(f"Created instance: {instance.read_write_dns}")

# COMMAND ----------

# Step 1: Grant CAN_USE on the Lakebase instance to the SPN
ws_token = w.tokens.create(lifetime_seconds=600).token_value
resp = requests.patch(
    f"{w.config.host}/api/2.0/permissions/database-instances/{instance_name}",
    headers={"Authorization": f"Bearer {ws_token}", "Content-Type": "application/json"},
    json={"access_control_list": [{"service_principal_name": client_id, "permission_level": "CAN_USE"}]},
)
resp.raise_for_status()
print(f"Granted CAN_USE on '{instance_name}' to SPN '{client_id}'")

# COMMAND ----------

# Step 2: Print Postgres SQL to run manually in the Lakebase SQL Editor.
# Open: Catalog → (your instance) → SQL Editor, then paste and run this.
lakebase_role_sql = f"""
CREATE EXTENSION IF NOT EXISTS databricks_auth;
SELECT databricks_create_role('{client_id}', 'SERVICE_PRINCIPAL');
GRANT CONNECT ON DATABASE databricks_postgres TO "{client_id}";
GRANT USAGE ON SCHEMA public TO "{client_id}";
GRANT SELECT, INSERT ON public.session_messages TO "{client_id}";
"""
print("Run the following SQL in the Lakebase SQL Editor:")
print(lakebase_role_sql)

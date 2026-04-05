# Databricks notebook source
# DO NOT RUN! For demonstration purposes!
from databricks.sdk import WorkspaceClient
from requests.auth import HTTPBasicAuth

w = WorkspaceClient()

client_id = "your client id"
client_secret = "your client secret"
account_id = "your account id"

w.secrets.create_scope(scope="admin1")
w.secrets.put_secret(scope="admin1", key="client_id", string_value=client_id)
w.secrets.put_secret(scope="admin1", key="client_secret", string_value=client_secret)
w.secrets.put_secret(scope="admin1", key="account_id", string_value=account_id)


# COMMAND ----------
import requests
from databricks.sdk import WorkspaceClient
from requests.auth import HTTPBasicAuth

w = WorkspaceClient()

# Admin credentials from secret scope
admin_client_id = dbutils.secrets.get("admin1", "client_id")
admin_client_secret = dbutils.secrets.get("admin1", "client_secret")
account_id = dbutils.secrets.get("admin1", "account_id")

account_host = "https://accounts.cloud.databricks.com"
instance_name = "arxiv-agent-instance"

# Get account-level token
token = requests.post(
    f"{account_host}/oidc/accounts/{account_id}/v1/token",
    auth=HTTPBasicAuth(admin_client_id, admin_client_secret),
    data={"grant_type": "client_credentials", "scope": "all-apis"}
).json()["access_token"]

# Step 1: Create service principal + OAuth secret
sp = w.service_principals.create(display_name="lakebase-sp-arxiv")
secret_resp = requests.post(
    f"{account_host}/api/2.0/accounts/{account_id}/servicePrincipals/{sp.id}/credentials/secrets",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
)
secret_resp.raise_for_status()
client_id = sp.application_id
client_secret = secret_resp.json()["secret"]

# Step 2: Store credentials in a secret scope
scope_name = "arxiv-agent-scope"
try:
    w.secrets.create_scope(scope=scope_name)
except Exception:
    pass  # scope already exists
w.secrets.put_secret(scope=scope_name, key="client_id", string_value=client_id)
w.secrets.put_secret(scope=scope_name, key="client_secret", string_value=client_secret)

# Step 3: Grant CAN_USE on database instance
ws_token = w.tokens.create(lifetime_seconds=600).token_value
requests.patch(
    f"{w.config.host}/api/2.0/permissions/database-instances/{instance_name}",
    headers={"Authorization": f"Bearer {ws_token}",
             "Content-Type": "application/json"},
    json={"access_control_list": [{"service_principal_name": client_id,
                                   "permission_level": "CAN_USE"}]}
).raise_for_status()

# Step 4: Postgres role SQL — run in Lakebase SQL Editor for 'arxiv-agent-instance'
lakebase_role_sql = f"""
CREATE EXTENSION IF NOT EXISTS databricks_auth;
SELECT databricks_create_role('{client_id}', 'SERVICE_PRINCIPAL');
GRANT CONNECT ON DATABASE databricks_postgres TO "{client_id}";
GRANT USAGE ON SCHEMA public TO "{client_id}";
GRANT SELECT, INSERT ON public.session_messages TO "{client_id}";
"""
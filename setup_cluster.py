# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Cluster Launch
# MAGIC We need to have libraries installed at a cluster level

# COMMAND ----------

%pip install -U databricks-sdk >= 0.9.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


w = WorkspaceClient(
  host  = db_host,
  token = db_token
)

# COMMAND ----------

# TODO - add in library installs
print("Attempting to create cluster. Please wait...")

c = w.clusters.create_and_wait(
  cluster_name             = 'hf_optimisation_cluster',
  spark_version            = '14.0.x-gpu-ml-scala2.12',
  node_type_id             = 'g5.12xlarge',
  autotermination_minutes = 45,
  num_workers              = 0,
  spark_conf               = {
        "spark.master": "local[*, 4]",
        "spark.databricks.cluster.profile": "singleNode"
        },
  custom_tags = {"ResourceClass": "SingleNode"}
  )

print(f"The cluster is now ready at " \
      f"{w.config.host}#setting/clusters/{c.cluster_id}/configuration\n")

# COMMAND ----------

# Install Libraries
from databricks.sdk.service.compute import Library

vllm_lib = Library().from_dict({'pypi': {'package': 'vllm==0.2.0'}})

w.libraries.install(c.cluster_id, [vllm_lib])

# COMMAND ----------

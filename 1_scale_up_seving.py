# Databricks notebook source
# MAGIC %md
# MAGIC # Serving Llama-2 with a cluster driver proxy app
# MAGIC We will use vLLM in order to scale up our serving
# MAGIC Note we haven't setup quantisation yet

# COMMAND ----------

%pip install ray[default]==2.6.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Setup
import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# setup home variables so that we don't run out of cache
import os
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

downloads_home = '/tmp/hf_ray_vllm/'
dbutils.fs.mkdirs(downloads_home)
dbfs_downloads_home = f'/dbfs{downloads_home}'

#os.environ['RAY_DLOAD_DIR'] = f'{dbfs_downloads_home}llama_2_gpu'

# setup model
## We need to download it for all the workers as well
from huggingface_hub import hf_hub_download, list_repo_files

repo_list = {'llama_2_7b': 'meta-llama/Llama-2-7b-chat-hf',
             'llama_2_13b': 'meta-llama/Llama-2-13b-chat-hf',
             'llama_2_70b': 'meta-llama/Llama-2-70b-chat-hf'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        # skip all the safetensors data as we aren't using it and it's time consuming to download
        if "safetensors" in name:
            continue
        target_path = os.path.join(dbfs_downloads_home, lib_name, name)
        if not os.path.exists(target_path):
            print(f"Downloading {name}")
            hf_hub_download(
                repo_list[lib_name],
                filename=name,
                local_dir=os.path.join(dbfs_downloads_home, lib_name),
                local_dir_use_symlinks=False,
            )

# COMMAND ----------

log_path = '/tmp/ray_logs_brian'

dbutils.fs.mkdirs(log_path)
dbfs_log_path = f'/dbfs{log_path}'

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

shutdown_ray_cluster()
# COMMAND ----------

setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=8,
  collect_log_to_path=dbfs_log_path
)

# COMMAND ----------

# MAGIC %md *NOTE* Manually set the right directory to load from for now
# MAGIC `/dbfs/tmp/hf_ray_vllm/llama_2_gpu_70b`
# MAGIC Set the tensor-parallel-size to num GPU needed

# COMMAND ----------

# MAGIC %sh

# MAGIC python -m vllm.entrypoints.api_server \
# MAGIC --host 0.0.0.0 \
# MAGIC --port 10101 \
# MAGIC --model <> \
# MAGIC --tokenizer <> \
# MAGIC --tensor-parallel-size 2 \
# MAGIC --dtype float \
# MAGIC --trust-remote-code

# COMMAND ----------

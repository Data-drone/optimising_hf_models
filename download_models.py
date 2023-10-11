# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Model Loading
# MAGIC To test out a lot of these optimisations we need to load and reload models

# COMMAND ----------

# DBTITLE 1,HF Credentials

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# setup home variables so that we don't run out of cache
import os
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

# setup model path
username = spark.sql("SELECT current_user()").first()['current_user()']

downloads_home = f'/home/{username}/hf_models'
dbutils.fs.mkdirs(downloads_home)
dbfs_downloads_home = f'/dbfs{downloads_home}'

# COMMAND ----------

from huggingface_hub import hf_hub_download, list_repo_files

repo_list = {'llama_2_7b': 'meta-llama/Llama-2-7b-chat-hf',
             'llama_2_13b': 'meta-llama/Llama-2-13b-chat-hf',
             'llama_2_13b_awq': 'TheBloke/Llama-2-13B-chat-AWQ'
             'llama_2_70b': 'meta-llama/Llama-2-70b-chat-hf',
             'llama_2_70b_awq': 'TheBloke/Llama-2-70B-chat-AWQ'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        # skip all the safetensors data as we aren't using it and it's time consuming to download
        # We do need safetensors for the awq models though
        if "safetensors" in name and 'awq' not in lib_name.split("_"):
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

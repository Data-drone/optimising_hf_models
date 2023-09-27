# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lets ping the vLLM server


# COMMAND ----------

import requests
from typing import Iterable, List
import json

# COMMAND ----------

# *Note* that we have embedded the generation config into the function for posting
# It is probably worthwhile to refactor and take these out or find better formats

def post_http_request(token: str,
                      prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    
    headers = {"User-Agent": "Test Client",
               "Content-Type": "application/json",
               "Authorization": "Bearer "+token}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": True,
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response

# These response functions might need editing as well

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

# COMMAND ----------

# Testing Prompt
system_prompt = 'As a helpful long island librarian, answer the questions provided in a succint and eloquent way.'

user_question = 'Explain to me like I am 5 LK-99'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Based on the below context:

LK-99 is a potential room-temperature superconductor with a gray‒black appearance.[2]: 8  It has a hexagonal structure slightly modified from lead‒apatite, by introducing small amounts of copper. A room-temperature superconductor is a material that is capable of exhibiting superconductivity at operating temperatures above 0 °C (273 K; 32 °F), that is, temperatures that can be reached and easily maintained in an everyday environment.

Provide an answer to the following:
{user_question}[/INST]
"""

# COMMAND ----------

token_temp = dbutils.secrets.get(scope='brian-hf', key='query-api')

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "10101"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}/generate"

#api_url = 'https://dbc-dp-5099015744649857.cloud.databricks.com/driver-proxy/o/5099015744649857/0927-034054-wqx8yh0g/8000/'
stream = False

print(f"Prompt: {prompt!r}\n", flush=True)
response = post_http_request(token_temp, prompt, driver_proxy_api, n, stream)

# COMMAND ----------

output = get_response(response)
for i, line in enumerate(output):
  print(f"Beam candidate {i}: {line!r}", flush=True)
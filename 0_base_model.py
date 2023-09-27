# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Inference on HuggingFace
# MAGIC *NOTE* this requires installing optimum which can only be done in cluster screen
# MAGIC this code was run with Optimum - 1.13.2

# COMMAND ----------

# MAGIC %pip install vllm

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

# COMMAND ----------

# Utility Function
import torch
from transformers import set_seed


def measure_latency_and_memory_use(model, inputs, nb_loops = 5):

  # define Events that measure start and end of the generate pass
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  # reset cuda memory stats and empty cache
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  # get the start time
  start_event.record()

  # actually generate
  for _ in range(nb_loops):
        # set seed for reproducibility
        set_seed(0)
        output = model(inputs, do_sample = True, temperature = 0.8)

  # get the end time
  end_event.record()
  torch.cuda.synchronize()

  # measure memory footprint and elapsed time
  max_memory = torch.cuda.max_memory_allocated()
  elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

  print('Execution time:', elapsed_time/nb_loops, 'seconds')
  print('Max memory footprint', max_memory*1e-9, ' GB')

  return output

# COMMAND ----------

# MAGIC %md # Setting up examples

# COMMAND ----------

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

# MAGIC %md # Model Selection

# COMMAND ----------

# Choose Model

from transformers import (
  AutoModelForCausalLM, pipeline, AutoTokenizer, 
) 

model_id = "meta-llama/Llama-2-7b-hf"
revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

model_id = "meta-llama/Llama-2-13b-chat-hf"
revision = "0ba94ac9b9e1d5a0037780667e8b219adde1908c"

# model_id = "meta-llama/Llama-2-70b-chat-hf"
# revision = "36d9a7388cc80e5f4b3e9701ca2f250d21a96c30"

n_loops = 10

# COMMAND ----------

# MAGIC %md # Base model

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        revision=revision,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

# COMMAND ----------

# Previous run for Llama 2 13b on 8xA10G:
# Execution time: 14.808284375 seconds
# Max memory footprint 5.862323712  GB

pipe = pipeline(
    "text-generation", model = model, tokenizer = tokenizer, max_new_tokens=125
)

with torch.inference_mode():
  text_output = measure_latency_and_memory_use(pipe, prompt, nb_loops = n_loops)

# COMMAND ----------

# MAGIC %md # Better Transformer
# MAGIC For more details on better transformer read: 

# COMMAND ----------

from optimum.bettertransformer import BetterTransformer

better_model = BetterTransformer.transform(model)

# COMMAND ----------

# Previous run for llama 2 13b on 8xA10G:
# Execution time: 24.311701562499998 seconds
# Max memory footprint 6.804478464000001  GB

better_pipeline = pipeline(
    "text-generation", model = better_model, tokenizer = tokenizer, max_new_tokens=125
)

with torch.inference_mode():
  text_output = measure_latency_and_memory_use(better_pipeline, prompt, nb_loops = n_loops)

# COMMAND ----------

# MAGIC %md # Adding bitsandbytes
# MAGIC Setting up bits and bytes is a bit more difficult
# MAGIC
# MAGIC We had to: 
# MAGIC - Add bitsandbytes
# MAGIC - Update Accelerate

# COMMAND ----------

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

compressed_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        revision=revision,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

# COMMAND ----------

# Previous Run llama 2 13b on 4x A10G
# Execution time: 8.38097578125 seconds
# Max memory footprint 2.3115688960000003  GB

compressed_pipeline = pipeline(
    "text-generation", model = compressed_model, tokenizer = tokenizer, max_new_tokens=125
)

with torch.inference_mode():
  text_output = measure_latency_and_memory_use(compressed_pipeline, prompt, nb_loops = n_loops)

# COMMAND ----------

# MAGIC %md
# MAGIC # Torch Compile test
# MAGIC MLR 14+ only or pip install on earlier

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

compile_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        revision=revision,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

compile_model = torch.compile(compile_model)

# COMMAND ----------

# Runtime with llama 2 13b
# Execution time: 14.446198437500001 seconds
# Max memory footprint 12.358254592000002  GB

compile_pipe = pipeline(
    "text-generation", model = compile_model, tokenizer = tokenizer, max_new_tokens=125
)

with torch.inference_mode():
  text_output = measure_latency_and_memory_use(compile_pipe, prompt, nb_loops = n_loops)

# COMMAND ----------

# MAGIC %md
# MAGIC # Flash Attention 2?
# MAGIC This requires installing from git for now

# COMMAND ----------

# MAGIC %md
# MAGIC # vLLM
# MAGIC This requires installing from git for now

# COMMAND ----------

from vllm import LLM, SamplingParams

os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

llm = LLM(model=model_id,
          tensor_parallel_size=4,
          dtype='bfloat16',
          #gpu_memory_utilization = 0.95,
          trust_remote_code=True)

sampling_params = SamplingParams(
                                temperature=0.8, 
                                # top_p=0.95,
                                top_k=10,
                                max_tokens=1024,
                                #  use_beam_search=True,
                                #  best_of=3
                                 )

# COMMAND ----------

# nax memory isn't import in this case
def vllm_measure_latency_and_memory_use(model, inputs, nb_loops = 5):

  # define Events that measure start and end of the generate pass
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  # reset cuda memory stats and empty cache
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  # get the start time
  start_event.record()

  # actually generate
  for _ in range(nb_loops):
        # set seed for reproducibility
        set_seed(0)
        output = model.generate(inputs, sampling_params)

  # get the end time
  end_event.record()
  torch.cuda.synchronize()

  # measure memory footprint and elapsed time
  max_memory = torch.cuda.max_memory_allocated()
  elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

  print('Execution time:', elapsed_time/nb_loops, 'seconds')
  print('Max memory footprint', max_memory*1e-9, ' GB')

  return output

# COMMAND ----------

# Run on llama 13b with 4x A10g
# Execution time: 6.35585546875 seconds
# Max memory footprint 0.0  GB

with torch.inference_mode():
  text_output = vllm_measure_latency_and_memory_use(llm, prompt, nb_loops = n_loops)

# COMMAND ----------
# %% [markdown]
# # core

# %%
#| default_exp core

# %%
#| hide
#| eval: false
import os
from nbdev.showdoc import *
from typing import Literal
from dataclasses import dataclass
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import time
import socket
from vllm import LLM, SamplingParams


# %%
@dataclass
class IndependentVariables:
    model_name: str
    dataset_name: str
    prompt: str
    prompt_type: Literal["yesno", "scale", "probe"]
    eval_size: int = 100
    threshold: float = 0.5

# %%
dep_exp1 = IndependentVariables(
    # model_name="TechxGenus/Meta-Llama-3-8B-GPTQ",
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="mtbench",
    prompt="Below is a prompt form a user: \n{prompt}\n\n\nAre you confident that you will be able to answer this prompt in a factually correct, helpful, informative, engaging, and creative way? Answer with a yes or no. Nothing else.",
    prompt_type="yesno",
    eval_size=100,
    threshold=0.5,
)

# %%
# Fetch the MTBench dataset

# %%
# Run the model
model = LLM(
    model=dep_exp1.model_name,
    dtype="auto",  # 'half',
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    swap_space=4,
    max_model_len=2048,
    kv_cache_dtype="fp8_e5m2",
    trust_remote_code=True,
    disable_custom_all_reduce=False,
    tensor_parallel_size=torch.cuda.device_count(),
    seed=42
)


# %%
# Test model with a sample query
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
# completion = client.completions.create(model=dep_exp1.model_name,
#                                       prompt="San Francisco is a")
# print("Completion result:", completion)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = model.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# %%
# Prepare prompt dataset of 100

# %%
# Run all the 100 prompts through the model and get confidence scores

# %%
# For confidence scores above threshold, let model answer and check the accuracy

# %%
# Repeat experiment by changing independent variables

# %%
#| hide
import nbdev; nbdev.nbdev_export()



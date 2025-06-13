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
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import socket
os.environ["OPENAI_API_KEY"] = "EMPTY"
from routellm.routers.routers import Router, ROUTER_CLS, NAME_TO_CLS
from confidence_based_routing.evaluate import evaluate
import datasets


# %%
@dataclass
class IndependentVariables:
    model_name: str
    dataset_name: str
    sys_prompt: str
    prompt_type: Literal["yesno_probs", "scale", "probe"]
    eval_size: int = 100
    threshold: float = 0.5
    max_new_tokens: int = 5
# %%
dep_exp2 = IndependentVariables(
    # model_name="TechxGenus/Meta-Llama-3-8B-GPTQ",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    dataset_name="mtbench",
    sys_prompt="You are a helpful assistant. You will get a prompt from a user. Are you confident that you will be able to answer this prompt in a factually correct, helpful, informative, engaging, and creative way? Answer with a yes or no. Nothing else.",
    prompt_type="yesno_probs",
    eval_size=100,
    threshold=0.5,
    max_new_tokens=1,
)

dep_exp3 = IndependentVariables(
    # model_name="TechxGenus/Meta-Llama-3-8B-GPTQ",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    dataset_name="mtbench",
    sys_prompt="You are a helpful assistant. You will get a prompt from a user. Are you confident that you will be able to answer this prompt in a factually correct, helpful, informative, engaging, and creative way? Rate your confidence on a scale of 1 to 10. Output only the number. Nothing else.",
    prompt_type="scale",
    eval_size=100,
    threshold=0.6,
    max_new_tokens=1,
)

# %%
# Run the model
model = AutoModelForCausalLM.from_pretrained(dep_exp1.model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(dep_exp1.model_name)

# %%
# Test model with a sample query
prompt = "What is the capital of France?"
conv = tokenizer.apply_chat_template([
    {"role": "system", "content": "\no_think You will get a prompt from a user. Are you confident that you will be able to answer this prompt in a factually correct, helpful, informative, engaging, and creative way? Answer with a yes or no. Nothing else.",},
    {"role": "user", "content": prompt}], 
    tokenize=False, add_generation_prompt=True)
input = tokenizer(conv, return_tensors="pt").to(model.device)
outputs = model.generate(**input, max_new_tokens=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

# %%
# Create a new router
class ConfidenceRouter(Router):
    NO_PARALLEL = True

    def __init__(self, model, tokenizer, indep_vars: IndependentVariables, verbose=False):
        self.indep_vars = indep_vars
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        
    def calculate_strong_win_rate(self, prompt):
        if self.verbose:
            print(f"Calculating strong win rate for prompt: {prompt}")
        conv = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.indep_vars.sys_prompt},
            {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        input = self.tokenizer(conv, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **input, 
            max_new_tokens=self.indep_vars.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )

        if self.indep_vars.prompt_type == "yesno_probs":
            tokens_of_interest = [self.tokenizer.encode("Yes"), self.tokenizer.encode("No")]
            scores = outputs.scores[-1][:, tokens_of_interest]
            probs = scores.softmax(dim=1)
            return probs[0, 0].item()

        if self.indep_vars.prompt_type == "scale":
            tokens_of_interest = [self.tokenizer.encode(str(i))[0] for i in range(1, 11)]
            scores = outputs.scores[-1][:, tokens_of_interest]
            probs = scores.softmax(dim=1)
            expected_value = (sum(p * (i+1) for i, p in enumerate(probs[0])) / 10).item()
            return expected_value
        
        raise ValueError(f"Prompt type {self.indep_vars.prompt_type} not supported")

# %%
router = ConfidenceRouter(model, tokenizer, indep_vars=dep_exp2, verbose=True)
router.calculate_strong_win_rate("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?")

# %%
router = ConfidenceRouter(model, tokenizer, indep_vars=dep_exp3, verbose=True)
router.calculate_strong_win_rate("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?")

# %%
dataset = datasets.load_dataset("openai/gsm8k", 'main', split="test")
list = []
for i in dataset.take(100):
    expected_value = router.calculate_strong_win_rate(i["question"])
    list.append(expected_value)

pd.Series(list).hist()
# %%
ROUTER_CLS["confidence_yesno_probs"] = ConfidenceRouter
ROUTER_CLS["confidence_scale"] = ConfidenceRouter
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}

# %%
import importlib
import confidence_based_routing.evaluate
importlib.reload(confidence_based_routing.evaluate)
import routellm
importlib.reload(routellm)
from confidence_based_routing.evaluate import evaluate

config = {
    "confidence_yesno_probs": {"model": model, "tokenizer": tokenizer, "indep_vars": dep_exp2}, 
    "confidence_scale": {"model": model, "tokenizer": tokenizer, "indep_vars": dep_exp3}, 
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}

evaluate(
    routers=["confidence_yesno_probs", "confidence_scale"], 
    benchmark="gsm8k", 
    config=config,
    overwrite_cache=[]
)

# %%
#| hide
import nbdev; nbdev.nbdev_export()

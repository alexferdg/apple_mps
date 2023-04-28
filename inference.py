from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
import torch

mps_device = torch.device("mps")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# model = AutoModelForCausalLM.from_pretrained("gpt2").to(mps_device)

pipe = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    torch_dtype=torch.float32,
    device=mps_device
)

generation = pipe("question: What is 42 ")
print(generation)

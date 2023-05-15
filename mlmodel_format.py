import coremltools as ct
"""
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
import torch

# Load a pre-trained model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# Set the model in evaluation mode.
model.eval()
# Trace the model with random data.
prompt: str = "Hello this is a trigger"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(input_ids)
traced_model = torch.jit.trace(model, input_ids)
out = traced_model(prompt)
"""
import torch
import torchvision

# Load a pre-trained version of MobileNetV2
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
# Set the model in evaluation mode.
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)

# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
# Save the converted model.
model.save("newmodel.mlpackage")
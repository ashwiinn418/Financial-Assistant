import torch
import bitsandbytes as bnb

print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0))
print("bitsandbytes version:", bnb.__version__)

# Try to create a quantized layer
temp_layer = bnb.nn.Linear4bit(10, 10)
print("Successfully created 4-bit layer")
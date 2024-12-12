import torch
import onnxruntime as ort


print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())


# Check if CUDA is available
print("CUDA is available:", torch.cuda.is_available())

# Check the GPU name
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("cuDNN is available:", torch.backends.cudnn.is_available())
else:
    print("No CUDA-enabled device detected.")

# Get available execution providers
providers = ort.get_available_providers()

# Check for CUDAExecutionProvider
if "CUDAExecutionProvider" in providers:
    print("CUDAExecutionProvider is available.")
else:
    print("CUDAExecutionProvider is not available.")
    
# Print all available providers
print("Available Providers:", providers)

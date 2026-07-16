# Create a virtual environment
python -m venv venv

# Activatation

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# Windows (Command Prompt)
# .venv\Scripts\activate.bat

python -m pip install --upgrade pip

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/
# Only change the wheel index (cu118, cu121, cpu, etc.)

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
    transformers \
    opencv-python-headless \
    websockets \
    onnxruntime \
    onnxscript \
    gradio

#onnxruntime-gpu in case of GPU
# for running inference on tensorrt do `pip install tensorrt`
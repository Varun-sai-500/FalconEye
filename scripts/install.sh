python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/
# Only change the wheel index (cu118, cu121, cpu, etc.)

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python -m pip install transformers/
opencv-python-headless /
websockets /
onnxruntime /
onnxscript /
gradio

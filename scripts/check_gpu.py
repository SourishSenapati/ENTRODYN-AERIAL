"""
GPU Availability Check Script.
Prints Torch version, CUDA availability, and Device name.
"""
import sys
import torch


def check_gpu():
    """Checks and prints GPU details."""
    print(f"Python Version: {sys.version}")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")


if __name__ == "__main__":
    check_gpu()

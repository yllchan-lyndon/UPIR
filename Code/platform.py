import platform
import sys
import torch

def print_system_info():
    """Prints essential software and hardware details for reproducibility."""

    print("--- General System Information ---")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU Architecture: {platform.machine()}")
    print(f"Processor Name (General): {platform.processor() if platform.processor() else 'N/A (Use terminal commands for specific model)'}")
    print(f"Python Version: {sys.version.split()[0]} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")

    # --- PyTorch and CUDA Info ---
    print("\n--- PyTorch and GPU Information ---")
    try:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            # Get CUDA version used by PyTorch build
            print(f"CUDA Compiler Version: {torch.version.cuda}")
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()
            print(f"GPU Count: {num_gpus}")

            # Print details for each GPU
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No CUDA-enabled GPU detected by PyTorch.")

    except ImportError:
        print("PyTorch is not installed or could not be imported.")
    except Exception as e:
        print(f"An error occurred while checking PyTorch/CUDA info: {e}")

if __name__ == "__main__":
    print_system_info()

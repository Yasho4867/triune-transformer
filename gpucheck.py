"""Compatibility entry point for GPU diagnostics."""

from triune.utils.device import device_diagnostic


def main() -> None:
    result = device_diagnostic()
    print(f"Python executable: {result['python']}")
    print(f"PyTorch version: {result['torch']}")
    print(f"CUDA available: {result['cuda_available']}")
    for index, gpu in enumerate(result["gpus"]):
        major, minor = gpu["capability"]
        print(f"GPU {index}: {gpu['name']} ({gpu['memory_gib']:.2f} GiB, SM{major}.{minor})")


if __name__ == "__main__":
    main()

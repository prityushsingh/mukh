# Installation

## Requirements

Mukh requires Python 3.7 or later. Make sure you have a compatible version installed:

```bash
python --version
```

## Installation via pip

The recommended way to install Mukh is using pip:

```bash
pip install mukh
```

## Development Installation

If you want to contribute to Mukh or use the latest development version, you can install it from source:

```bash
# Clone the repository
git clone https://github.com/ishandutta0098/mukh.git
cd mukh

# Install in development mode
pip install -e .
```

## Verify Installation

After installation, verify that Mukh is working correctly:

```python
import mukh
print(mukh.__version__)
```

## Dependencies

Mukh automatically installs the following dependencies:

- **OpenCV**: For image processing operations
- **NumPy**: For numerical computations
- **Pillow**: For image handling
- **MediaPipe**: For MediaPipe-based face detection
- **TensorFlow**: For deep learning models
- **PyTorch**: For PyTorch-based models

!!! note "GPU Support"
    For GPU acceleration, make sure you have the appropriate CUDA drivers installed for TensorFlow and PyTorch.

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'mukh'`

**Solution**: Make sure you've installed the package correctly:
```bash
pip list | grep mukh
```

**Issue**: GPU not detected

**Solution**: Install CUDA-compatible versions of TensorFlow and PyTorch:
```bash
# For TensorFlow GPU
pip install tensorflow-gpu

# For PyTorch GPU (check pytorch.org for your specific CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Getting Help

If you encounter any issues during installation:

1. Check the [GitHub Issues](https://github.com/ishandutta0098/mukh/issues)
2. Create a new issue with your system information and error details
3. Join our [community discussions](https://github.com/ishandutta0098/mukh/discussions)

## Next Steps

Once you have Mukh installed, check out the [Quick Start Guide](quick-start.md) to begin using the library. 
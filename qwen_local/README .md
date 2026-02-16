# Qwen 2.5 Coder 7B Instruct - Local Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python qwen_coder_test.py
```

## Hardware Requirements

### Minimum Configuration
- **RAM**: 16 GB system RAM
- **VRAM**: 8 GB GPU memory (FP16/BF16)
- **Storage**: ~15 GB free disk space
- **GPU**: CUDA-compatible (optional, but recommended)

### Recommended Configuration
- **RAM**: 32 GB system RAM
- **VRAM**: 16 GB GPU memory
- **GPU**: NVIDIA RTX 3090/4090, A4000, or better
- **Storage**: 20 GB SSD

### Memory-Saving Options

#### Option 1: 8-bit Quantization (~4 GB VRAM)
```python
USE_QUANTIZATION = '8bit'
```

#### Option 2: 4-bit Quantization (~2-3 GB VRAM)
```python
USE_QUANTIZATION = '4bit'
```

#### Option 3: CPU-only Mode (16+ GB RAM, slower)
The script will automatically fall back to CPU if no GPU is available.

## Configuration

Edit the `USE_QUANTIZATION` variable in `qwen_coder_test.py`:

```python
USE_QUANTIZATION = None    # Full precision (FP16/BF16)
USE_QUANTIZATION = '8bit'  # 8-bit quantization
USE_QUANTIZATION = '4bit'  # 4-bit quantization
```

## Features

- ✅ Automatic hardware detection
- ✅ Multiple quantization options
- ✅ Interactive chat interface
- ✅ Pre-configured test prompts
- ✅ Proper chat template formatting
- ✅ Error handling and troubleshooting

## Troubleshooting

### CUDA Out of Memory Error
- Try 8-bit quantization: `USE_QUANTIZATION = '8bit'`
- Try 4-bit quantization: `USE_QUANTIZATION = '4bit'`
- Reduce `max_new_tokens` to generate shorter responses
- Close other GPU-intensive applications

### Model Download Issues
- Ensure stable internet connection
- The model (~14 GB) downloads automatically on first run
- Cache location: `~/.cache/huggingface/hub/`

### ImportError for bitsandbytes
```bash
pip install bitsandbytes
```

Note: On Windows, bitsandbytes may require additional setup.

## Expected Performance

| Configuration | VRAM Usage | Speed (tokens/sec) |
|--------------|------------|-------------------|
| FP16         | ~14 GB     | ~30-50 (GPU)      |
| 8-bit        | ~4 GB      | ~20-35 (GPU)      |
| 4-bit        | ~2-3 GB    | ~15-25 (GPU)      |
| CPU          | 0 GB       | ~2-5 (CPU)        |

*Speed varies based on hardware and prompt complexity*

## Model Information

- **Model**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Parameters**: 7.61 billion
- **Context Length**: 128K tokens
- **Specialization**: Code generation and understanding
- **License**: Check Hugging Face model card

## Support

For issues with the model itself, visit:
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

# LoRA Attention Analyzer

A comprehensive tool for analyzing attention maps in Stable Diffusion models with LoRA (Low-Rank Adaptation) support. This package provides detailed visualization and analysis of how LoRA modifications affect token attention patterns in diffusion models.

## Features

- **Attention Map Analysis**: Extract and visualize attention maps from Stable Diffusion models
- **LoRA Comparison**: Compare attention patterns before and after applying LoRA weights
- **Comprehensive Visualization**: Multiple visualization modes including pixel-wise, grid-based, and token-specific analysis
- **Flexible LoRA Support**: Automatic LoRA loading with fallback to direct weight application
- **Token Analysis**: Detailed analysis of how different tokens affect image generation
- **Gallery Views**: Side-by-side comparison galleries for easy analysis

## Installation

### From PyPI (when published)
```bash
pip install lora-attention-analyzer
```

### From Source
```bash
git clone https://github.com/yourusername/lora-attention-analyzer.git
cd lora-attention-analyzer
pip install -e .
```

### Optional Dependencies
For advanced image processing capabilities:
```bash
pip install "lora-attention-analyzer[optional]"
```

For development:
```bash
pip install "lora-attention-analyzer[dev]"
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16+ GB VRAM for SDXL models

## Quick Start

### Command Line Interface

#### Basic Usage
```bash
# Run complete LoRA comparison analysis
lora-attention-analyzer compare \
  --model_id "/path/to/model.safetensors" \
  --lora_file "/path/to/lora.safetensors" \
  --prompt "your generation prompt" \
  --output_dir "./results"
```

#### Advanced Usage
```bash
# Detailed analysis with custom parameters
lora-attention-analyzer compare \
  --model_id "/path/to/model.safetensors" \
  --lora_file "/path/to/lora.safetensors" \
  --prompt "detailed prompt for generation" \
  --output_dir "./results" \
  --model_type "vpred" \
  --lora_scale 1.0 \
  --steps 50 \
  --guidance_scale 7.5 \
  --height 1024 \
  --width 1024 \
  --seed 42
```

#### Analyze LoRA File Structure
```bash
# Analyze LoRA file without running generation
lora-attention-analyzer analyze --lora_file "/path/to/lora.safetensors"
```

### Python API

```python
from lora_attention_analyzer import LoRAAttentionPipeline

# Initialize pipeline
pipeline = LoRAAttentionPipeline(
    model_id="/path/to/model.safetensors",
    model_type="vpred",  # or "epsilon"
    device="cuda"
)

# Run analysis
results = pipeline.run_comparison(
    lora_file="/path/to/lora.safetensors",
    prompt="your generation prompt",
    output_dir="./results",
    lora_scale=1.0,
    seed=42
)

# Access results
base_image = results['base']['image']
lora_image = results['lora']['image']
base_heat_map = results['base']['heat_map']
lora_heat_map = results['lora']['heat_map']
```

### Advanced API Usage

```python
from lora_attention_analyzer import AttentionExtractor, AttentionVisualizer

# Extract attention data
extractor = AttentionExtractor()
heat_data = extractor.extract_heat_data(token_heat_map, "token_name")
attention_scores = extractor.get_token_attention_scores(heat_map, tokens)

# Create custom visualizations
visualizer = AttentionVisualizer()
visualizer.visualize_pixel_dominant_tokens(heat_map, image, tokens, output_dir)
visualizer.create_token_attention_gallery(heat_map, image, tokens, output_dir)
```

## Output Structure

When running an analysis, the tool generates:

```
output_directory/
├── base_image.png                    # Generated image without LoRA
├── lora_image.png                    # Generated image with LoRA
├── image_comparison.png              # Side-by-side comparison
├── attention_comparison.png          # Token attention comparison chart
├── base_analysis/                    # Comprehensive base analysis
│   ├── pixel_dominant_tokens.png     # Pixel-wise dominant token map
│   ├── grid_analysis_4x4.png         # 4x4 grid analysis
│   ├── grid_analysis_8x8.png         # 8x8 grid analysis
│   ├── token_attention_distribution.png
│   └── token_attention_gallery.png   # All token overlays
└── lora_analysis/                    # Comprehensive LoRA analysis
    ├── pixel_dominant_tokens.png
    ├── grid_analysis_4x4.png
    ├── grid_analysis_8x8.png
    ├── token_attention_distribution.png
    └── token_attention_gallery.png
```

## Supported Models

- **Stable Diffusion XL** (SDXL) models
- **V-prediction** and **epsilon** parameterization
- Models in `.safetensors` format
- Standard LoRA files with UNet and Text Encoder weights

## Configuration Options

### Model Types
- `vpred`: V-prediction parameterization (common in newer models)
- `epsilon`: Epsilon parameterization (traditional)

### LoRA Scale
- Range: 0.0 to 2.0 (typically 0.5 to 1.5)
- 0.0: No LoRA effect
- 1.0: Full LoRA effect (recommended starting point)
- >1.0: Amplified LoRA effect

### Generation Parameters
- `steps`: Number of inference steps (20-100, default: 50)
- `guidance_scale`: CFG scale (5.0-15.0, default: 7.5)
- `height`/`width`: Image dimensions (512-1536, default: 1024)

## Advanced Features

### Custom Token Analysis
```python
# Specify custom tokens to analyze
tokens_to_analyze = ["character_name", "art_style", "composition"]

results = pipeline.run_comparison(
    # ... other parameters
    tokens_to_analyze=tokens_to_analyze
)
```

### Batch Processing
```python
# Process multiple LoRA files
lora_files = ["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"]

for i, lora_file in enumerate(lora_files):
    results = pipeline.run_comparison(
        lora_file=lora_file,
        output_dir=f"./results/comparison_{i}",
        # ... other parameters
    )
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image dimensions (`--height 768 --width 768`)
   - Use `enable_model_cpu_offload()` (enabled by default)
   - Close other GPU applications

2. **LoRA Loading Failed**
   - Ensure LoRA file is compatible with the base model
   - Check file integrity (corrupted downloads)
   - Try different `lora_scale` values

3. **Token Not Found**
   - Tokens are automatically extracted from prompts
   - Use simpler, single-word tokens for better results
   - Check token spelling and spacing

### Performance Tips

- Use half precision (`torch.float16`) for faster generation
- Enable xformers attention for memory efficiency
- Use shorter prompts for faster token analysis
- Consider using `steps=30` for faster iterations during testing

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/yourusername/lora-attention-analyzer.git
cd lora-attention-analyzer
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black lora_attention_analyzer/
isort lora_attention_analyzer/
flake8 lora_attention_analyzer/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{lora_attention_analyzer,
  title={LoRA Attention Analyzer: Visualizing Attention Maps in Stable Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lora-attention-analyzer}
}å
```

## Acknowledgments

- [DAAM](https://github.com/castorini/daam) for attention map computation
- [Diffusers](https://github.com/huggingface/diffusers) for Stable Diffusion implementation
- [Safetensors](https://github.com/huggingface/safetensors) for efficient tensor storage

## Related Projects

- [DAAM](https://github.com/castorini/daam) - Diffusion Attentive Attribution Maps
- [Diffusers](https://github.com/huggingface/diffusers) - Hugging Face Diffusers library
- [LoRA](https://github.com/microsoft/LoRA) - Low-Rank Adaptation
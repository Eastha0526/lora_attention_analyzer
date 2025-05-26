"""
Command-line interface for LoRA Attention Analyzer.
"""

import fire
import sys
from pathlib import Path
from typing import Optional, List

from ..core.pipeline import LoRAAttentionPipeline


def run_lora_comparison(
    model_id: str,
    lora_file: str,
    prompt: str,
    output_dir: str = './lora_analysis_output',
    model_type: str = 'vpred',
    negative_prompt: Optional[str] = None,
    lora_scale: float = 1.0,
    seed: int = 0,
    steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 1024,
    width: int = 1024,
    tokens_to_analyze: Optional[List[str]] = None,
    device: str = 'cuda'
) -> None:
    """
    Run comprehensive LoRA attention analysis comparison.
    
    This command generates images with and without LoRA, then creates detailed
    attention analysis visualizations to understand how LoRA affects the model.
    
    Args:
        model_id: Path to the base SDXL model file (.safetensors)
        lora_file: Path to the LoRA file (.safetensors)
        prompt: Text prompt for image generation
        output_dir: Directory to save analysis results (default: './lora_analysis_output')
        model_type: Model parameterization type ('vpred' or 'epsilon', default: 'vpred')
        negative_prompt: Negative prompt (uses sensible default if None)
        lora_scale: LoRA strength/scale factor (0.0-2.0, default: 1.0)
        seed: Random seed for reproducible generation (default: 0)
        steps: Number of inference steps (default: 50)
        guidance_scale: CFG guidance scale (default: 7.5)
        height: Image height in pixels (default: 1024)
        width: Image width in pixels (default: 1024)
        tokens_to_analyze: Specific tokens to analyze (auto-detected if None)
        device: Device to use for computation ('cuda' or 'cpu', default: 'cuda')
    
    Examples:
        # Basic usage
        lora-attention-analyzer compare \\
            --model_id "/path/to/model.safetensors" \\
            --lora_file "/path/to/lora.safetensors" \\
            --prompt "a cute cat sitting in a garden"
        
        # Advanced usage with custom parameters
        lora-attention-analyzer compare \\
            --model_id "/path/to/model.safetensors" \\
            --lora_file "/path/to/character_lora.safetensors" \\
            --prompt "anime girl, detailed face, blue eyes" \\
            --output_dir "./character_analysis" \\
            --lora_scale 1.2 \\
            --steps 50 \\
            --seed 42
    """
    print("LoRA Attention Analyzer")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"LoRA: {lora_file}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_dir}")
    print(f"LoRA Scale: {lora_scale}")
    print(f"Seed: {seed}")
    print(f"Steps: {steps}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Validate inputs
    try:
        _validate_inputs(model_id, lora_file, lora_scale, steps, guidance_scale, height, width)
    except ValueError as e:
        print(f"Input validation failed: {e}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        print("\nInitializing pipeline...")
        pipeline = LoRAAttentionPipeline(
            model_id=model_id,
            model_type=model_type,
            device=device
        )
        
        # Show pipeline info
        info = pipeline.get_pipeline_info()
        print(f"Model type: {info['model_type']}")
        print(f"Device: {info['device']}")
        print(f"Scheduler: {info['scheduler']}")
        print(f"UNet loaded: {info['unet_loaded']}")
        print(f"Text encoder loaded: {info['text_encoder_loaded']}")
        
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        print("\nTroubleshooting tips:")
        print("   • Check that the model file exists and is accessible")
        print("   • Ensure you have enough GPU memory (8GB+ recommended)")
        print("   • Try using --device cpu if GPU memory is insufficient")
        sys.exit(1)
    
    # Analyze LoRA file structure
    try:
        print("\nAnalyzing LoRA file structure...")
        lora_analysis = pipeline.analyze_lora_file(lora_file)
        
        if 'error' in lora_analysis:
            print(f"LoRA analysis failed: {lora_analysis['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"LoRA file analysis failed: {e}")
        print("\nTroubleshooting tips:")
        print("   • Check that the LoRA file exists and is accessible")
        print("   • Ensure the file is in .safetensors format")
        print("   • Verify the LoRA file is not corrupted")
        sys.exit(1)
    
    # Run the comparison analysis
    try:
        print("\n🎨 Starting LoRA comparison analysis...")
        results = pipeline.run_comparison(
            lora_file=lora_file,
            prompt=prompt,
            output_dir=output_dir,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            tokens_to_analyze=tokens_to_analyze
        )
        
        if 'error' not in results:
            _print_success_summary(output_dir, results)
        else:
            print(f"\nAnalysis failed: {results['error']}")
            _print_failure_tips()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        _print_failure_tips()
        sys.exit(1)


def analyze_lora_file(lora_file: str) -> None:
    """
    Analyze LoRA file structure and contents without running generation.
    
    This command examines a LoRA file and provides detailed information about
    its structure, including the number of keys for different components.
    
    Args:
        lora_file: Path to the LoRA file (.safetensors)
    
    Examples:
        # Analyze a LoRA file
        lora-attention-analyzer analyze --lora_file "/path/to/lora.safetensors"
    """
    print("LoRA File Analyzer")
    print("=" * 40)
    print(f"File: {lora_file}")
    print("=" * 40)
    
    # Validate file
    if not Path(lora_file).exists():
        print(f"LoRA file not found: {lora_file}")
        print("\nCheck the file path and try again.")
        sys.exit(1)
    
    # Analyze file
    try:
        from ..core.lora_utils import LoRAUtils
        
        lora_utils = LoRAUtils()
        analysis = lora_utils.analyze_lora_file(lora_file)
        
        if 'error' in analysis:
            print(f"Analysis failed: {analysis['error']}")
            sys.exit(1)
        
        # Print detailed results
        print("\nAnalysis Results:")
        print("=" * 30)
        print(f"Total keys: {analysis['total_keys']}")
        print(f"Text Encoder 1: {analysis['text_encoder_1_keys']} keys")
        print(f"Text Encoder 2: {analysis['text_encoder_2_keys']} keys")
        print(f"UNet: {analysis['unet_keys']} keys")
        print(f"LoRA pairs: {analysis['lora_pairs']}")
        print(f"Alpha keys: {analysis['alpha_keys']}")
        
        if analysis['sample_keys']:
            print(f"\nSample UNet Keys:")
            for sample in analysis['sample_keys']:
                size_mb = sample['size_mb']
                print(f"   {sample['key']}")
                print(f"     Shape: {sample['shape']}")
                print(f"     Type: {sample['dtype']}")
                print(f"     Size: {size_mb:.2f} MB")
                print()
        
        # Print key patterns
        if analysis['key_patterns']:
            print("Key Patterns:")
            for component, patterns in analysis['key_patterns'].items():
                if patterns:
                    print(f"   {component}: {len(patterns)} unique patterns")
                    for pattern in patterns[:3]:  # Show first 3 patterns
                        print(f"     • {pattern}")
                    if len(patterns) > 3:
                        print(f"     ... and {len(patterns) - 3} more")
                    print()
        
        print("LoRA file analysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("\nTroubleshooting tips:")
        print("   • Ensure the file is in .safetensors format")
        print("   • Check that the file is not corrupted")
        print("   • Verify you have read permissions for the file")
        sys.exit(1)


def _validate_inputs(
    model_id: str, 
    lora_file: str, 
    lora_scale: float, 
    steps: int, 
    guidance_scale: float, 
    height: int, 
    width: int
) -> None:
    """Validate command line inputs."""
    # Check file existence
    if not Path(model_id).exists():
        raise ValueError(f"Model file not found: {model_id}")
    
    if not Path(lora_file).exists():
        raise ValueError(f"LoRA file not found: {lora_file}")
    
    # Check file extensions
    if not model_id.lower().endswith('.safetensors'):
        raise ValueError(f"Model file must be .safetensors format: {model_id}")
    
    if not lora_file.lower().endswith('.safetensors'):
        raise ValueError(f"LoRA file must be .safetensors format: {lora_file}")
    
    # Check parameter ranges
    if not 0.0 <= lora_scale <= 2.0:
        raise ValueError(f"LoRA scale must be between 0.0 and 2.0, got: {lora_scale}")
    
    if not 1 <= steps <= 200:
        raise ValueError(f"Steps must be between 1 and 200, got: {steps}")
    
    if not 1.0 <= guidance_scale <= 30.0:
        raise ValueError(f"Guidance scale must be between 1.0 and 30.0, got: {guidance_scale}")
    
    if not 256 <= height <= 2048:
        raise ValueError(f"Height must be between 256 and 2048, got: {height}")
    
    if not 256 <= width <= 2048:
        raise ValueError(f"Width must be between 256 and 2048, got: {width}")
    
    # Check if dimensions are multiples of 8 (required for most models)
    if height % 8 != 0:
        raise ValueError(f"Height must be a multiple of 8, got: {height}")
    
    if width % 8 != 0:
        raise ValueError(f"Width must be a multiple of 8, got: {width}")


def _print_success_summary(output_dir: str, results: dict) -> None:
    """Print success summary with file listings."""
    print("\nLoRA Comparison Analysis Completed Successfully!")
    print("=" * 60)
    print(f"Results saved in: {output_dir}")
    print("\nGenerated Files:")
    print("┌─ Base Analysis:")
    print("│  ├── base_image.png (generated image without LoRA)")
    print("│  └── base_analysis/ (comprehensive attention analysis)")
    print("├─ LoRA Analysis:")
    print("│  ├── lora_image.png (generated image with LoRA)")
    print("│  └── lora_analysis/ (comprehensive attention analysis)")
    print("└─ Comparative Analysis:")
    print("   ├── image_comparison.png (side-by-side image comparison)")
    print("   ├── attention_comparison.png (attention score comparison)")
    print("   └── token_comparisons/ (individual token comparisons)")
    print("\nAnalysis Features:")
    print("   • Pixel-wise dominant token visualization")
    print("   • Grid-based attention analysis")
    print("   • Token attention distribution charts")
    print("   • Comprehensive token attention galleries")
    print("   • Before/after comparison visualizations")
    print("\nNext Steps:")
    print(f"   1. Open {output_dir} to view all generated visualizations")
    print("   2. Compare base_image.png and lora_image.png for visual differences")
    print("   3. Check attention_comparison.png to see how LoRA affects token attention")
    print("   4. Explore the analysis directories for detailed breakdowns")


def _print_failure_tips() -> None:
    """Print troubleshooting tips for failed analysis."""
    print("\nTroubleshooting Tips:")
    print("┌─ GPU Memory Issues:")
    print("│  • Try smaller image dimensions (--height 768 --width 768)")
    print("│  • Use CPU mode (--device cpu)")
    print("│  • Close other GPU applications")
    print("├─ LoRA Compatibility:")
    print("│  • Ensure LoRA is trained for the same model architecture")
    print("│  • Try different LoRA scale values (--lora_scale 0.8)")
    print("│  • Check LoRA file integrity")
    print("└─ General Issues:")
    print("   • Check file paths and permissions")
    print("   • Ensure all dependencies are installed")
    print("   • Try with a simpler prompt first")
    print("\nFor more help, check the documentation or create an issue on GitHub.")


def main() -> None:
    """
    Main CLI entry point using Fire for command routing.
    
    Available commands:
        compare: Run complete LoRA vs base model comparison analysis
        analyze: Analyze LoRA file structure without running generation
    
    Examples:
        lora-attention-analyzer compare --model_id model.safetensors --lora_file lora.safetensors --prompt "cute cat"
        lora-attention-analyzer analyze --lora_file lora.safetensors
        lora-attention-analyzer --help
    """
    try:
        # Use Fire to handle CLI routing
        fire.Fire({
            'compare': run_lora_comparison,
            'analyze': analyze_lora_file,
        })
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nIf this error persists, please report it as a bug.")
        sys.exit(1)


if __name__ == '__main__':
    main()
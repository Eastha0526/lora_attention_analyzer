"""
Basic usage examples for LoRA Attention Analyzer.
"""

from lora_attention_analyzer import LoRAAttentionPipeline
from pathlib import Path


def basic_comparison_example():
    """Basic LoRA comparison example."""
    
    # Configuration
    model_path = "/path/to/your/model.safetensors"
    lora_path = "/path/to/your/lora.safetensors"
    prompt = "a cute cat sitting in a garden, detailed, high quality"
    output_dir = "./output/basic_comparison"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please update the model_path variable with your model file path.")
        return
    
    if not Path(lora_path).exists():
        print(f"❌ LoRA file not found: {lora_path}")
        print("Please update the lora_path variable with your LoRA file path.")
        return
    
    print("🚀 Starting basic LoRA comparison...")
    
    # Initialize pipeline
    pipeline = LoRAAttentionPipeline(
        model_id=model_path,
        model_type="vpred",  # Change to "epsilon" if needed
        device="cuda"
    )
    
    # Run comparison
    results = pipeline.run_comparison(
        lora_file=lora_path,
        prompt=prompt,
        output_dir=output_dir,
        lora_scale=1.0,
        seed=42,
        steps=30  # Faster for testing
    )
    
    if 'error' not in results:
        print("✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
    else:
        print(f"❌ Analysis failed: {results['error']}")


def advanced_analysis_example():
    """Advanced analysis with custom parameters."""
    
    model_path = "/path/to/your/model.safetensors"
    lora_path = "/path/to/your/lora.safetensors"
    prompt = "anime girl, detailed face, blue eyes, long hair, school uniform"
    output_dir = "./output/advanced_analysis"
    
    if not Path(model_path).exists() or not Path(lora_path).exists():
        print("❌ Please update the file paths in the script.")
        return
    
    print("🔬 Starting advanced analysis...")
    
    # Initialize pipeline
    pipeline = LoRAAttentionPipeline(
        model_id=model_path,
        model_type="vpred",
        device="cuda"
    )
    
    # Custom tokens to analyze
    tokens_to_analyze = ["anime girl", "blue eyes", "long hair", "school uniform"]
    
    # Run analysis with custom parameters
    results = pipeline.run_comparison(
        lora_file=lora_path,
        prompt=prompt,
        output_dir=output_dir,
        negative_prompt="worst quality, low quality, blurry, bad anatomy",
        lora_scale=1.2,  # Slightly amplified
        seed=123,
        steps=50,
        guidance_scale=8.0,
        height=1024,
        width=768,  # Portrait orientation
        tokens_to_analyze=tokens_to_analyze
    )
    
    if 'error' not in results:
        print("✅ Advanced analysis completed!")
        print(f"📁 Results saved in: {output_dir}")
        
        # Access specific results
        base_image = results['base']['image']
        lora_image = results['lora']['image']
        print(f"📸 Base image size: {base_image.size}")
        print(f"📸 LoRA image size: {lora_image.size}")
    else:
        print(f"❌ Analysis failed: {results['error']}")


def lora_scale_comparison():
    """Compare different LoRA scales."""
    
    model_path = "/path/to/your/model.safetensors"
    lora_path = "/path/to/your/lora.safetensors"
    prompt = "portrait of a character, detailed"
    
    if not Path(model_path).exists() or not Path(lora_path).exists():
        print("❌ Please update the file paths in the script.")
        return
    
    print("⚖️ Comparing different LoRA scales...")
    
    # Initialize pipeline
    pipeline = LoRAAttentionPipeline(
        model_id=model_path,
        model_type="vpred",
        device="cuda"
    )
    
    # Test different scales
    scales = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    for scale in scales:
        print(f"\n📊 Testing LoRA scale: {scale}")
        
        output_dir = f"./output/scale_comparison/scale_{scale}"
        
        results = pipeline.run_comparison(
            lora_file=lora_path,
            prompt=prompt,
            output_dir=output_dir,
            lora_scale=scale,
            seed=42,  # Same seed for comparison
            steps=30
        )
        
        if 'error' not in results:
            print(f"✅ Scale {scale} completed!")
        else:
            print(f"❌ Scale {scale} failed: {results['error']}")


def analyze_lora_file_only():
    """Analyze LoRA file structure without generation."""
    
    lora_path = "/path/to/your/lora.safetensors"
    
    if not Path(lora_path).exists():
        print(f"❌ LoRA file not found: {lora_path}")
        print("Please update the lora_path variable.")
        return
    
    print("🔍 Analyzing LoRA file structure...")
    
    from lora_attention_analyzer.core.lora_utils import LoRAUtils
    
    lora_utils = LoRAUtils()
    analysis = lora_utils.analyze_lora_file(lora_path)
    
    print("\n📊 Analysis Results:")
    print(f"  📋 Total keys: {analysis['total_keys']}")
    print(f"  📝 Text Encoder 1 keys: {analysis['text_encoder_1_keys']}")
    print(f"  📝 Text Encoder 2 keys: {analysis['text_encoder_2_keys']}")
    print(f"  🧠 UNet keys: {analysis['unet_keys']}")
    
    if analysis['sample_unet_keys']:
        print("\n  🔗 Sample UNet key shapes:")
        for sample in analysis['sample_unet_keys']:
            print(f"    {sample['key']}: {sample['shape']}")


def custom_visualization_example():
    """Example of custom visualization using individual components."""
    
    print("🎨 Custom visualization example...")
    print("This example shows how to use individual components for custom analysis.")
    
    # This would require having heat_map and image objects from a previous analysis
    # Here's the structure you would use:
    
    """
    from lora_attention_analyzer import AttentionExtractor, AttentionVisualizer
    
    # Initialize components
    extractor = AttentionExtractor()
    visualizer = AttentionVisualizer()
    
    # Extract attention data
    tokens = ["your", "custom", "tokens"]
    attention_scores = extractor.get_token_attention_scores(heat_map, tokens)
    
    # Create custom visualizations
    output_dir = "./custom_analysis"
    visualizer.visualize_pixel_dominant_tokens(heat_map, image, tokens, output_dir)
    visualizer.create_token_attention_gallery(heat_map, image, tokens, output_dir)
    
    # Print attention scores
    print("Token attention scores:")
    for token, score in attention_scores.items():
        print(f"  {token}: {score:.4f}")
    """
    
    print("💡 To use this, you need heat_map and image objects from a previous analysis.")
    print("Run basic_comparison_example() first to generate these objects.")


if __name__ == "__main__":
    print("🎯 LoRA Attention Analyzer - Usage Examples")
    print("=" * 50)
    
    # Update these paths before running
    print("⚠️  Please update the file paths in this script before running!")
    print("   - Update model_path to your model file")
    print("   - Update lora_path to your LoRA file")
    print()
    
    # Choose which example to run
    examples = {
        "1": ("Basic Comparison", basic_comparison_example),
        "2": ("Advanced Analysis", advanced_analysis_example),
        "3": ("LoRA Scale Comparison", lora_scale_comparison),
        "4": ("Analyze LoRA File Only", analyze_lora_file_only),
        "5": ("Custom Visualization Info", custom_visualization_example),
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nTo run an example, modify this script and call the function directly.")
    print("Example: basic_comparison_example()")
    
    # Uncomment one of these to run:
    # basic_comparison_example()
    # advanced_analysis_example()
    # lora_scale_comparison()
    # analyze_lora_file_only()
    # custom_visualization_example()
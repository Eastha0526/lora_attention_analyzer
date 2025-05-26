"""
Main pipeline for LoRA attention analysis.
"""

import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from daam import trace, set_seed
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
import matplotlib.pyplot as plt

from .lora_utils import LoRAUtils
from ..analysis.attention_extractor import AttentionExtractor
from ..analysis.visualizer import AttentionVisualizer


class LoRAAttentionPipeline:
    """
    Main pipeline for analyzing attention maps with and without LoRA.
    
    This class orchestrates the entire workflow:
    1. Load and configure the diffusion pipeline
    2. Apply LoRA weights
    3. Generate images with attention tracking
    4. Create comprehensive analysis and visualizations
    """
    
    def __init__(
        self,
        model_id: str,
        model_type: str = 'vpred',
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the LoRA Attention Pipeline.
        
        Args:
            model_id: Path to the base model file (.safetensors)
            model_type: Model type ('vpred' for v-prediction, 'epsilon' for epsilon parameterization)
            device: Device to run on ('cuda' or 'cpu')
            torch_dtype: PyTorch data type for the model (float16 recommended for GPU)
        """
        self.model_id = model_id
        self.model_type = model_type
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize components
        self.pipe = None
        self.lora_utils = LoRAUtils()
        self.attention_extractor = AttentionExtractor()
        self.visualizer = AttentionVisualizer()
        
        # Default negative prompt
        self.default_negative_prompt = (
            "worst quality, low quality, lowers, low details, bad quality, poorly drawn, "
            "bad anatomy, multiple views, bad hands, blurry, artist sign, weibo username"
        )
        
        # Load the pipeline
        self._load_pipeline()
    
    def _load_pipeline(self) -> None:
        """Load and configure the diffusion pipeline."""
        print(f"Loading pipeline from: {self.model_id}")
        
        # Load base pipeline
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_id, 
            torch_dtype=self.torch_dtype, 
            use_safetensors=True, 
            variant='fp16'
        )
        
        # Configure scheduler based on model type
        if self.model_type == "vpred":
            scheduler_args = {
                "prediction_type": "v_prediction", 
                "rescale_betas_zero_snr": True
            }
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config, **scheduler_args
            )
        else:
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        
        # Enable memory optimization
        self.pipe.enable_xformers_memory_efficient_attention(
            attention_op=MemoryEfficientAttentionFlashAttentionOp
        )
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to(self.device)
        
        print("Pipeline loaded successfully!")
    
    def analyze_lora_file(self, lora_file: str) -> Dict[str, Any]:
        """
        Analyze the structure of a LoRA file.
        
        Args:
            lora_file: Path to the LoRA file
            
        Returns:
            Dictionary containing LoRA file analysis results
        """
        return self.lora_utils.analyze_lora_file(lora_file)
    
    def run_comparison(
        self,
        lora_file: str,
        prompt: str,
        output_dir: str,
        negative_prompt: Optional[str] = None,
        lora_scale: float = 1.0,
        seed: int = 0,
        steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        tokens_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete LoRA comparison analysis.
        
        Args:
            lora_file: Path to LoRA file (.safetensors)
            prompt: Generation prompt
            output_dir: Output directory for results
            negative_prompt: Negative prompt (uses default if None)
            lora_scale: LoRA scaling factor (0.0-2.0)
            seed: Random seed for reproducible generation
            steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            height: Image height in pixels
            width: Image width in pixels
            tokens_to_analyze: List of tokens to analyze (auto-detected if None)
            
        Returns:
            Dictionary containing analysis results with base and lora data
        """
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        
        if negative_prompt is None:
            negative_prompt = self.default_negative_prompt
        
        if tokens_to_analyze is None:
            tokens_to_analyze = self._extract_tokens_from_prompt(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Tokens to analyze: {tokens_to_analyze}")
        print(f"Output directory: {output_dir}")
        
        results = {}
        
        # Step 1: Generate base image (without LoRA)
        print("\nGENERATING BASE IMAGE (WITHOUT LORA)")
        print("=" * 50)
        base_results = self._generate_with_analysis(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
            tokens_to_analyze=tokens_to_analyze,
            output_prefix="base",
            output_dir=output_dir
        )
        results['base'] = base_results
        
        # Step 2: Apply LoRA
        print("\nApplying LoRA weights")
        print("=" * 50)
        lora_success = self._apply_lora(lora_file, lora_scale)
        
        if lora_success:
            # Step 3: Generate LoRA image
            print("\nGENERATING LORA IMAGE")
            print("=" * 50)
            lora_results = self._generate_with_analysis(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
                tokens_to_analyze=tokens_to_analyze,
                output_prefix="lora",
                output_dir=output_dir
            )
            results['lora'] = lora_results
            
            # Step 4: Create comparative analysis
            print("\nCREATING COMPARATIVE ANALYSIS")
            print("=" * 50)
            self._create_comparative_analysis(results, output_dir, tokens_to_analyze)
            
            print("\nLoRA comparison completed successfully!")
            print(f"Results saved in: {output_dir}")
        else:
            print("\nFailed to apply LoRA weights")
            results['error'] = "Failed to apply LoRA weights"
        
        return results
    
    def _generate_with_analysis(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        steps: int,
        seed: int,
        tokens_to_analyze: List[str],
        output_prefix: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Generate image with attention analysis."""
        gen = set_seed(seed)
        
        with torch.no_grad():
            with trace(self.pipe) as tc:
                # Generate image
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=gen
                )
                
                image = output.images[0]
                image_path = os.path.join(output_dir, f'{output_prefix}_image.png')
                image.save(image_path)
                print(f"Saved image: {image_path}")
                
                # Compute global heat map
                heat_map = tc.compute_global_heat_map()
                
                # Generate individual token visualizations
                print(f"Generating token visualizations for: {tokens_to_analyze}")
                for token in tokens_to_analyze:
                    try:
                        token_heat_map = heat_map.compute_word_heat_map(token)
                        token_heat_map.plot_overlay(image)
                        token_path = os.path.join(output_dir, f'{token}_{output_prefix}.png')
                        plt.savefig(token_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"{token}: saved to {token_path}")
                    except Exception as e:
                        print(f"{token}: failed - {e}")
                
                # Create comprehensive analysis
                analysis_dir = os.path.join(output_dir, f'{output_prefix}_analysis')
                print(f"Creating comprehensive analysis in: {analysis_dir}")
                self.visualizer.create_comprehensive_analysis(
                    heat_map, image, analysis_dir, tokens_to_analyze
                )
                
                return {
                    'image': image,
                    'heat_map': heat_map,
                    'analysis_dir': analysis_dir,
                    'image_path': image_path
                }
    
    def _apply_lora(self, lora_file: str, lora_scale: float) -> bool:
        """
        Apply LoRA weights to the pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        # Try using built-in LoRA loading first
        if hasattr(self.pipe, "load_lora_weights"):
            try:
                print("Attempting to use built-in load_lora_weights...")
                self.pipe.load_lora_weights(lora_file)
                
                adapter_names = list(self.pipe.get_active_adapters())
                print(f"Available adapters: {adapter_names}")
                
                if adapter_names:
                    self.pipe.set_adapters(
                        adapter_names, 
                        adapter_weights=[lora_scale] * len(adapter_names)
                    )
                    print(f"LoRA loaded successfully with scale {lora_scale}")
                    return True
            except Exception as e:
                print(f"Built-in loading failed: {e}")
        
        # Fallback to direct application
        print("🔄 Attempting direct LoRA weight application...")
        return self.lora_utils.apply_lora_directly(
            self.pipe, lora_file, lora_scale, self.device
        )
    
    def _extract_tokens_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract meaningful tokens from prompt for analysis.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            List of meaningful tokens for analysis
        """
        import re
        
        # Common stop words to filter out
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Split by common delimiters and clean up
        tokens = re.split(r'[,\s]+', prompt.lower())
        
        # Filter and clean tokens
        meaningful_tokens = []
        for token in tokens:
            token = token.strip('.,!?;:"()[]{}')
            if (token and 
                len(token) > 2 and 
                token not in stop_words and
                not token.isdigit()):
                meaningful_tokens.append(token)
        
        # Limit to reasonable number for analysis
        return meaningful_tokens[:8]
    
    def _create_comparative_analysis(
        self, 
        results: Dict[str, Any], 
        output_dir: str, 
        tokens_to_analyze: List[str]
    ) -> None:
        """Create comparative analysis between base and LoRA results."""
        print("Creating side-by-side comparisons...")
        
        try:
            self.visualizer.create_comparative_analysis(
                base_heat_map=results['base']['heat_map'],
                lora_heat_map=results['lora']['heat_map'],
                base_image=results['base']['image'],
                lora_image=results['lora']['image'],
                tokens=tokens_to_analyze,
                output_dir=output_dir
            )
            print("Comparative analysis completed")
        except Exception as e:
            print(f"Comparative analysis failed: {e}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the loaded pipeline."""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'scheduler': type(self.pipe.scheduler).__name__ if self.pipe else None,
            'unet_loaded': self.pipe is not None and hasattr(self.pipe, 'unet'),
            'text_encoder_loaded': self.pipe is not None and hasattr(self.pipe, 'text_encoder'),
        }
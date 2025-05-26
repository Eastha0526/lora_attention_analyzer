"""
Utilities for handling LoRA weights and application.
"""

import re
import torch
from typing import Dict, Tuple, List, Any
from safetensors import safe_open


class LoRAUtils:
    """
    Comprehensive utilities for LoRA file handling and application.
    
    This class provides functionality for:
    - Analyzing LoRA file structure and contents
    - Converting between different key naming conventions
    - Applying LoRA weights directly to pipeline models
    """
    
    def analyze_lora_file(self, lora_file: str) -> Dict[str, Any]:
        """
        Analyze the structure and contents of a LoRA file.
        
        Args:
            lora_file: Path to the LoRA file (.safetensors)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        print(f"Analyzing LoRA file: {lora_file}")
        
        analysis = {
            'file_path': lora_file,
            'total_keys': 0,
            'text_encoder_1_keys': 0,
            'text_encoder_2_keys': 0,
            'unet_keys': 0,
            'lora_pairs': 0,
            'alpha_keys': 0,
            'sample_keys': [],
            'key_patterns': {
                'te1_patterns': set(),
                'te2_patterns': set(),
                'unet_patterns': set()
            }
        }
        
        try:
            with safe_open(lora_file, framework="pt", device="cpu") as f:
                lora_keys = list(f.keys())
                analysis['total_keys'] = len(lora_keys)
                
                # Classify keys by component
                te1_keys = [k for k in lora_keys if k.startswith('lora_te1_')]
                te2_keys = [k for k in lora_keys if k.startswith('lora_te2_')]
                unet_keys = [k for k in lora_keys if not k.startswith(('lora_te1_', 'lora_te2_'))]
                alpha_keys = [k for k in lora_keys if k.endswith('.alpha')]
                
                analysis['text_encoder_1_keys'] = len(te1_keys)
                analysis['text_encoder_2_keys'] = len(te2_keys)
                analysis['unet_keys'] = len(unet_keys)
                analysis['alpha_keys'] = len(alpha_keys)
                
                # Find LoRA pairs
                analysis['lora_pairs'] = len(self._find_lora_pairs(lora_keys))
                
                # Analyze key patterns
                for key in te1_keys[:5]:
                    pattern = self._extract_key_pattern(key)
                    analysis['key_patterns']['te1_patterns'].add(pattern)
                
                for key in te2_keys[:5]:
                    pattern = self._extract_key_pattern(key)
                    analysis['key_patterns']['te2_patterns'].add(pattern)
                
                for key in unet_keys[:5]:
                    pattern = self._extract_key_pattern(key)
                    analysis['key_patterns']['unet_patterns'].add(pattern)
                
                # Sample key information with shapes
                for key in unet_keys[:5]:
                    tensor = f.get_tensor(key)
                    analysis['sample_keys'].append({
                        'key': key,
                        'shape': tuple(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024)
                    })
                
                # Convert pattern sets to lists for JSON serialization
                for component in analysis['key_patterns']:
                    analysis['key_patterns'][component] = list(analysis['key_patterns'][component])
        
        except Exception as e:
            analysis['error'] = str(e)
            print(f"Error analyzing LoRA file: {e}")
            return analysis
        
        # Print summary
        print(f"LoRA Analysis Results:")
        print(f"Total keys: {analysis['total_keys']}")
        print(f"Text Encoder 1: {analysis['text_encoder_1_keys']} keys")
        print(f"Text Encoder 2: {analysis['text_encoder_2_keys']} keys")  
        print(f"UNet: {analysis['unet_keys']} keys")
        print(f"LoRA pairs: {analysis['lora_pairs']}")
        print(f"Alpha keys: {analysis['alpha_keys']}")
        
        if analysis['sample_keys']:
            print(f"Sample UNet keys:")
            for sample in analysis['sample_keys']:
                print(f"{sample['key']}: {sample['shape']} ({sample['size_mb']:.2f}MB)")
        
        return analysis
    
    def apply_lora_directly(
        self, 
        pipe: Any, 
        lora_file: str, 
        lora_scale: float, 
        device: str
    ) -> bool:
        """
        Apply LoRA weights directly to the pipeline models.
        
        Args:
            pipe: Diffusion pipeline object
            lora_file: Path to LoRA file
            lora_scale: Scaling factor for LoRA application
            device: Device to use for computation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading LoRA tensors from: {lora_file}")
            
            # Load all LoRA tensors
            lora_tensors = {}
            with safe_open(lora_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lora_tensors[key] = f.get_tensor(key)
            
            print(f"Loaded {len(lora_tensors)} tensors")
            
            # Classify LoRA keys by target component
            te1_keys = [k for k in lora_tensors.keys() if k.startswith('lora_te1_')]
            te2_keys = [k for k in lora_tensors.keys() if k.startswith('lora_te2_')]
            unet_keys = [k for k in lora_tensors.keys() if not k.startswith(('lora_te1_', 'lora_te2_'))]

            modified_modules = 0

            # Apply UNet LoRA weights
            if unet_keys:
                print(f"Applying UNet LoRA ({len(unet_keys)} keys)...")
                unet_pairs = self._find_lora_pairs(unet_keys)
                print(f"Found {len(unet_pairs)} UNet LoRA pairs")
                modified_modules += self._apply_unet_lora(
                    pipe.unet, unet_pairs, lora_tensors, lora_scale, device
                )

            # Apply Text Encoder 1 LoRA weights
            if te1_keys and hasattr(pipe, 'text_encoder'):
                print(f"Applying Text Encoder 1 LoRA ({len(te1_keys)} keys)...")
                te1_pairs = self._find_lora_pairs(te1_keys)
                print(f"Found {len(te1_pairs)} TE1 LoRA pairs")
                modified_modules += self._apply_text_encoder_lora(
                    pipe.text_encoder, te1_pairs, lora_tensors, lora_scale, device, 'lora_te1_'
                )

            # Apply Text Encoder 2 LoRA weights
            if te2_keys and hasattr(pipe, 'text_encoder_2'):
                print(f"Applying Text Encoder 2 LoRA ({len(te2_keys)} keys)...")
                te2_pairs = self._find_lora_pairs(te2_keys)
                print(f"Found {len(te2_pairs)} TE2 LoRA pairs")
                modified_modules += self._apply_text_encoder_lora(
                    pipe.text_encoder_2, te2_pairs, lora_tensors, lora_scale, device, 'lora_te2_'
                )

            success = modified_modules > 0
            if success:
                print(f"Successfully modified {modified_modules} modules with LoRA weights (scale: {lora_scale})")
            else:
                print("No modules were modified")
            
            return success

        except Exception as e:
            print(f"Direct LoRA application failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _find_lora_pairs(self, keys: List[str]) -> Dict[str, Tuple[str, str]]:
        """
        Find matching up/down LoRA weight pairs.
        
        Args:
            keys: List of LoRA tensor keys
            
        Returns:
            Dictionary mapping base keys to (up_key, down_key) tuples
        """
        up_down_pairs = {}
        
        for key in keys:
            if '.lora_up.weight' in key:
                base_key = key.replace('.lora_up.weight', '')
                down_key = key.replace('.lora_up.weight', '.lora_down.weight')
                
                if down_key in keys:
                    up_down_pairs[base_key] = (key, down_key)
        
        return up_down_pairs
    
    def _apply_unet_lora(
        self, 
        unet: Any, 
        lora_pairs: Dict[str, Tuple[str, str]], 
        lora_tensors: Dict[str, torch.Tensor], 
        lora_scale: float, 
        device: str
    ) -> int:
        """Apply LoRA weights to UNet model."""
        modified_modules = 0
        unet_state_dict = unet.state_dict()
        
        for base_key, (up_key, down_key) in lora_pairs.items():
            # Try various UNet key conversion strategies
            possible_unet_keys = [
                base_key + '.weight',
                base_key,
                base_key.replace('_', '.') + '.weight',
                self._convert_sd_key_to_diffusers(base_key) + '.weight',
                self._convert_sd_key_to_diffusers(base_key)
            ]
            
            # Find matching key in UNet state dict
            unet_key = None
            for possible_key in possible_unet_keys:
                if possible_key in unet_state_dict:
                    unet_key = possible_key
                    break
            
            if unet_key:
                try:
                    # Get LoRA weights
                    lora_up = lora_tensors[up_key].to(device)
                    lora_down = lora_tensors[down_key].to(device)
                    
                    # Get alpha scaling factor
                    alpha_key = base_key + '.alpha'
                    if alpha_key in lora_tensors:
                        alpha = lora_tensors[alpha_key].to(device)
                        scale = alpha / lora_up.shape[0] if alpha.numel() == 1 else 1.0
                    else:
                        scale = 1.0
                    
                    # Calculate LoRA delta based on layer type
                    if len(lora_up.shape) == 4:  # Convolutional layer
                        lora_delta = torch.nn.functional.conv2d(
                            lora_down.permute(1, 0, 2, 3), 
                            lora_up
                        ).permute(1, 0, 2, 3)
                    else:  # Linear layer
                        lora_delta = lora_up @ lora_down
                    
                    # Apply LoRA modification
                    original_weight = unet_state_dict[unet_key]
                    new_weight = original_weight + (lora_scale * scale * lora_delta.to(original_weight.dtype))
                    
                    # Update model parameters
                    with torch.no_grad():
                        for name, param in unet.named_parameters():
                            if name == unet_key:
                                param.copy_(new_weight)
                                break
                    
                    modified_modules += 1
                    if modified_modules <= 3:  # Print first few for feedback
                        print(f"Applied to {unet_key} (alpha: {scale:.3f})")
                    
                except Exception as e:
                    if modified_modules < 5:  # Only print first few errors
                        print(f"Failed to apply to {unet_key}: {e}")
            else:
                if modified_modules < 5:  # Only print first few missing keys
                    print(f"UNet key not found for: {base_key}")
        
        return modified_modules
    
    def _apply_text_encoder_lora(
        self, 
        text_encoder: Any, 
        lora_pairs: Dict[str, Tuple[str, str]], 
        lora_tensors: Dict[str, torch.Tensor], 
        lora_scale: float, 
        device: str, 
        prefix: str
    ) -> int:
        """Apply LoRA weights to Text Encoder model."""
        modified_modules = 0
        te_state_dict = text_encoder.state_dict()
        
        for base_key, (up_key, down_key) in lora_pairs.items():
            # Remove prefix and convert key format
            clean_key = base_key.replace(prefix, '')
            possible_te_keys = [
                clean_key + '.weight',
                clean_key,
                clean_key.replace('_', '.') + '.weight',
                self._convert_te_key_to_diffusers(clean_key) + '.weight',
                self._convert_te_key_to_diffusers(clean_key)
            ]
            
            # Find matching key in Text Encoder state dict
            te_key = None
            for possible_key in possible_te_keys:
                if possible_key in te_state_dict:
                    te_key = possible_key
                    break
            
            if te_key:
                try:
                    # Get LoRA weights
                    lora_up = lora_tensors[up_key].to(device)
                    lora_down = lora_tensors[down_key].to(device)
                    
                    # Get alpha scaling factor
                    alpha_key = base_key + '.alpha'
                    if alpha_key in lora_tensors:
                        alpha = lora_tensors[alpha_key].to(device)
                        scale = alpha / lora_up.shape[0] if alpha.numel() == 1 else 1.0
                    else:
                        scale = 1.0
                    
                    # Calculate LoRA delta (text encoders use linear layers)
                    lora_delta = lora_up @ lora_down
                    
                    # Apply LoRA modification
                    original_weight = te_state_dict[te_key]
                    new_weight = original_weight + (lora_scale * scale * lora_delta.to(original_weight.dtype))
                    
                    # Update model parameters
                    with torch.no_grad():
                        for name, param in text_encoder.named_parameters():
                            if name == te_key:
                                param.copy_(new_weight)
                                break
                    
                    modified_modules += 1
                    if modified_modules <= 2:  # Print first few for feedback
                        print(f"Applied to {te_key} (alpha: {scale:.3f})")
                    
                except Exception as e:
                    if modified_modules < 3:  # Only print first few errors
                        print(f"Failed to apply to {te_key}: {e}")
        
        return modified_modules
    
    def _convert_sd_key_to_diffusers(self, sd_key: str) -> str:
        """Convert Stable Diffusion key naming to Diffusers format."""
        diffusers_key = sd_key
        
        # Convert block naming patterns
        diffusers_key = re.sub(r'input_blocks\.(\d+)\.(\d+)', r'down_blocks.\1.attentions.\2', diffusers_key)
        diffusers_key = re.sub(r'middle_block\.1', 'mid_block.attentions.0', diffusers_key)
        diffusers_key = re.sub(r'output_blocks\.(\d+)\.(\d+)', r'up_blocks.\1.attentions.\2', diffusers_key)
        
        # Convert attention layer naming
        diffusers_key = diffusers_key.replace('_transformer_blocks.', '.transformer_blocks.')
        diffusers_key = diffusers_key.replace('_attn1_', '.attn1.')
        diffusers_key = diffusers_key.replace('_attn2_', '.attn2.')
        diffusers_key = diffusers_key.replace('_to_', '.to_')
        diffusers_key = diffusers_key.replace('_proj_', '.proj_')
        diffusers_key = diffusers_key.replace('_ff_net.', '.ff.net.')
        diffusers_key = diffusers_key.replace('0_proj', '0.proj')
        
        return diffusers_key
    
    def _convert_te_key_to_diffusers(self, te_key: str) -> str:
        """Convert Text Encoder key naming to Diffusers format."""
        diffusers_key = te_key
        
        # Convert text encoder layer naming
        diffusers_key = diffusers_key.replace('text_model_encoder_layers_', 'text_model.encoder.layers.')
        diffusers_key = diffusers_key.replace('_mlp_', '.mlp.')
        diffusers_key = diffusers_key.replace('_self_attn_', '.self_attn.')
        diffusers_key = diffusers_key.replace('_', '.')
        
        return diffusers_key
    
    def _extract_key_pattern(self, key: str) -> str:
        """Extract general pattern from a LoRA key for analysis."""
        # Remove specific indices and weights to get pattern
        pattern = re.sub(r'\.\d+\.', '.N.', key)
        pattern = re.sub(r'\.lora_(up|down)\.weight', '.lora_X.weight', pattern)
        pattern = re.sub(r'\.alpha$', '.alpha', pattern)
        return pattern
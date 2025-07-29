"""
Comprehensive visualization utilities for attention analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from typing import List, Any, Optional, Dict

from .attention_extractor import AttentionExtractor


class AttentionVisualizer:
    """
    Comprehensive visualization tools for attention analysis.
    
    This class provides various visualization methods for analyzing
    attention patterns in diffusion models:
    - Pixel-wise dominant token visualization
    - Grid-based analysis
    - Attention distribution charts
    - Token attention galleries
    - Comparative analysis between models
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.attention_extractor = AttentionExtractor()
        
        # Default visualization settings
        self.default_figsize = (12, 8)
        self.default_dpi = 300
        self.color_palette = "Set3"
        
        # Set matplotlib style for better plots
        plt.style.use('default')
        sns.set_palette(self.color_palette)
    
    def create_comprehensive_analysis(
        self, 
        heat_map: Any, 
        image: Any, 
        output_dir: str, 
        tokens: List[str]
    ) -> None:
        """
        Create comprehensive token analysis and visualization suite.
        
        Args:
            heat_map: DAAM global heat map
            image: Generated image (PIL Image or numpy array)
            output_dir: Output directory for saving results
            tokens: List of tokens to analyze
        """
        try:
            print(f"Starting comprehensive visualization suite")
            print(f"Output directory: {output_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Validate tokens by testing heat map generation
            valid_tokens = self._validate_tokens(heat_map, tokens)
            
            if not valid_tokens:
                print("No valid tokens found for visualization")
                return
            
            print(f"Validated {len(valid_tokens)}/{len(tokens)} tokens: {valid_tokens}")

            # Create visualization suite
            visualizations = [
                ("Pixel Dominant Tokens", self.visualize_pixel_dominant_tokens),
                ("4x4 Grid Analysis", lambda hm, img, t, od: self.visualize_grid_analysis(hm, img, t, od, 4)),
                ("8x8 Grid Analysis", lambda hm, img, t, od: self.visualize_grid_analysis(hm, img, t, od, 8)),
                ("Token Attention Distribution", self.visualize_token_attention_distribution),
                ("Token Attention Gallery", self.create_token_attention_gallery),
            ]
            
            for viz_name, viz_func in visualizations:
                try:
                    print(f"Creating {viz_name}...")
                    viz_func(heat_map, image, valid_tokens, output_dir)
                    print(f"{viz_name} completed")
                except Exception as e:
                    print(f"{viz_name} failed: {e}")
            
            print("Comprehensive analysis completed!")
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_tokens(self, heat_map: Any, tokens: List[str]) -> List[str]:
        """Validate tokens by testing heat map generation."""
        valid_tokens = []
        
        for token in tokens:
            try:
                test_map = heat_map.compute_word_heat_map(token)
                # Try to extract data to ensure it's valid
                heat_data = self.attention_extractor.extract_heat_data(test_map, token)
                if heat_data is not None:
                    valid_tokens.append(token)
                    print(f"'{token}': valid")
                else:
                    print(f"'{token}': no extractable data")
            except Exception as e:
                print(f"'{token}': error - {e}")
        
        return valid_tokens
    
    def visualize_pixel_dominant_tokens(
        self, 
        heat_map: Any, 
        image: Any, 
        tokens: List[str], 
        output_dir: str
    ) -> None:
        """
        Visualize pixel-wise dominant tokens using color-coded maps.
        
        This creates a visualization showing which token has the highest
        attention value at each pixel location.
        """
        try:
            print(f"Processing {len(tokens)} tokens for pixel dominance")

            # Extract attention maps for all tokens
            token_maps = []
            valid_tokens = []
            
            for token in tokens:
                try:
                    token_heat_map = heat_map.compute_word_heat_map(token)
                    heat_data = self.attention_extractor.extract_heat_data(token_heat_map, token)
                    
                    if heat_data is not None:
                        token_maps.append(heat_data)
                        valid_tokens.append(token)
                        print(f"'{token}': {heat_data.shape}")
                    
                except Exception as e:
                    print(f"'{token}': {e}")
                    continue
            
            if len(token_maps) < 2:
                print(f"Need at least 2 valid tokens, got {len(token_maps)}")
                return
            
            # Normalize maps to same shape
            target_shape = token_maps[0].shape[:2]
            normalized_maps = self.attention_extractor.normalize_attention_maps(
                token_maps, target_shape
            )
            
            # Compute dominant tokens
            dominant_indices, stats = self.attention_extractor.compute_dominant_tokens(
                normalized_maps, valid_tokens
            )
            
            # Create visualization
            self._create_dominant_tokens_plot(
                image, dominant_indices, valid_tokens, stats, output_dir
            )
            
        except Exception as e:
            print(f"Pixel dominant visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_dominant_tokens_plot(
        self, 
        image: Any, 
        dominant_indices: np.ndarray, 
        tokens: List[str], 
        stats: Dict, 
        output_dir: str
    ) -> None:
        """Create the actual dominant tokens plot."""
        colors = sns.color_palette(self.color_palette, len(tokens))
        cmap = ListedColormap(colors)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Dominant token map
        im = axes[1].imshow(dominant_indices, cmap=cmap, vmin=0, vmax=len(tokens)-1)
        axes[1].set_title("Pixel-wise Dominant Tokens", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add legend
        legend_elements = [Patch(facecolor=colors[i], label=f"{tokens[i]} ({stats['dominant_percentages'][tokens[i]]:.1f}%)") 
                          for i in range(len(tokens))]
        axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Statistics bar chart
        percentages = [stats['dominant_percentages'][token] for token in tokens]
        bars = axes[2].bar(range(len(tokens)), percentages, color=colors)
        axes[2].set_title("Pixel Dominance Distribution", fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Tokens')
        axes[2].set_ylabel('Percentage of Pixels (%)')
        axes[2].set_xticks(range(len(tokens)))
        axes[2].set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, pct in zip(bars, percentages):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'pixel_dominant_tokens.png')
        plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def visualize_grid_analysis(
        self, 
        heat_map: Any, 
        image: Any, 
        tokens: List[str], 
        output_dir: str, 
        grid_size: int = 4
    ) -> None:
        """
        Visualize grid-based attention analysis with overlay.
        
        Args:
            grid_size: Size of the analysis grid (grid_size x grid_size)
        """
        try:
            print(f"Creating {grid_size}x{grid_size} grid analysis")
            
            # Get attention scores
            token_scores = self.attention_extractor.get_token_attention_scores(heat_map, tokens, run_label="grid")
            
            if len(token_scores) < 2:
                print(f"Need at least 2 valid tokens for grid analysis")
                return
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image with grid overlay
            axes[0].imshow(image)
            axes[0].set_title(f"Original Image with {grid_size}×{grid_size} Analysis Grid", fontsize=14, fontweight='bold')
            
            # Draw grid lines
            img_height, img_width = self._get_image_dimensions(image)
            self._draw_grid_lines(axes[0], img_width, img_height, grid_size)
            axes[0].axis('off')
            
            # Token attention distribution
            self._create_attention_distribution_chart(axes[1], token_scores, "Grid Analysis")
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'grid_analysis_{grid_size}x{grid_size}.png')
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Grid analysis failed: {e}")
    
    def _get_image_dimensions(self, image: Any) -> tuple:
        """Get image dimensions regardless of format (PIL or numpy)."""
        if hasattr(image, 'size'):  # PIL Image
            img_width, img_height = image.size
        else:  # numpy array
            img_height, img_width = image.shape[:2]
        return img_height, img_width
    
    def _draw_grid_lines(self, ax: plt.Axes, img_width: int, img_height: int, grid_size: int) -> None:
        """Draw grid lines on the image."""
        grid_h = img_height // grid_size
        grid_w = img_width // grid_size
        
        # Draw horizontal lines
        for i in range(1, grid_size):
            ax.axhline(y=i * grid_h, color='white', linewidth=2, alpha=0.8)
        
        # Draw vertical lines
        for i in range(1, grid_size):
            ax.axvline(x=i * grid_w, color='white', linewidth=2, alpha=0.8)
    
    def visualize_token_attention_distribution(
        self, 
        heat_map: Any, 
        tokens: List[str], 
        output_dir: str
    ) -> None:
        """
        Create comprehensive token attention distribution visualization.
        """
        try:
            print(f"Creating attention distribution for {len(tokens)} tokens")
            
            # Get attention scores
            token_scores = self.attention_extractor.get_token_attention_scores(heat_map, tokens, run_label="dist")
            
            if not token_scores:
                print(f"No valid attention scores found")
                return
            
            # Create comprehensive distribution plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Token Attention Analysis', fontsize=16, fontweight='bold')
            
            # 1. Raw scores
            self._create_attention_distribution_chart(axes[0, 0], token_scores, "Raw Attention Scores")
            
            # 2. Normalized scores
            total_attention = sum(token_scores.values())
            normalized_scores = {k: v/total_attention for k, v in token_scores.items()}
            self._create_attention_distribution_chart(axes[0, 1], normalized_scores, "Normalized Attention (%)", is_percentage=True)
            
            # 3. Sorted by importance
            sorted_tokens = sorted(token_scores.keys(), key=lambda x: token_scores[x], reverse=True)
            sorted_scores = {token: token_scores[token] for token in sorted_tokens}
            self._create_attention_distribution_chart(axes[1, 0], sorted_scores, "Ranked by Attention")
            
            # 4. Attention ratios (relative to max)
            max_attention = max(token_scores.values())
            ratio_scores = {k: v/max_attention for k, v in token_scores.items()}
            self._create_attention_distribution_chart(axes[1, 1], ratio_scores, "Relative to Maximum", is_ratio=True)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'token_attention_distribution.png')
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Attention distribution visualization failed: {e}")
    
    def _create_attention_distribution_chart(
        self, 
        ax: plt.Axes, 
        token_scores: Dict[str, float], 
        title: str,
        is_percentage: bool = False,
        is_ratio: bool = False
    ) -> None:
        """Create a single attention distribution chart."""
        tokens = list(token_scores.keys())
        scores = list(token_scores.values())
        
        colors = sns.color_palette(self.color_palette, len(tokens))
        bars = ax.bar(range(len(tokens)), scores, color=colors)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Tokens')
        
        if is_percentage:
            ax.set_ylabel('Attention Percentage')
            format_str = '{:.1%}'
        elif is_ratio:
            ax.set_ylabel('Attention Ratio')
            format_str = '{:.2f}'
        else:
            ax.set_ylabel('Attention Score')
            format_str = '{:.3f}'
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   format_str.format(score), ha='center', va='bottom', fontsize=9)
    
    def create_token_attention_gallery(
        self, 
        heat_map: Any, 
        image: Any, 
        tokens: List[str], 
        output_dir: str
    ) -> None:
        """
        Create a gallery showing all token attention maps as overlays.
        """
        try:
            print(f"Creating attention gallery for {len(tokens)} tokens")
            
            # Collect valid token maps
            valid_tokens = []
            token_maps = []
            
            for token in tokens:
                try:
                    token_heat_map = heat_map.compute_word_heat_map(token)
                    valid_tokens.append(token)
                    token_maps.append(token_heat_map)
                    print(f"Added '{token}' to gallery")
                except Exception as e:
                    print(f"'{token}' failed: {e}")
                    continue
            
            if not valid_tokens:
                print(f"No valid tokens for gallery")
                return
            
            # Calculate grid layout
            n_tokens = len(valid_tokens)
            cols = min(4, n_tokens)
            rows = (n_tokens + cols - 1) // cols
            
            print(f"Creating {rows}×{cols} gallery grid")
            
            # Create gallery plot
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            fig.suptitle('Token Attention Gallery', fontsize=16, fontweight='bold')
            
            # Ensure axes is always 2D array
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Plot each token
            for i, (token, token_map) in enumerate(zip(valid_tokens, token_maps)):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                try:
                    # Create attention overlay
                    if hasattr(token_map, 'plot_overlay'):
                        token_map.plot_overlay(image, ax=ax)
                    else:
                        # Fallback: manual overlay
                        ax.imshow(image)
                        heat_data = self.attention_extractor.extract_heat_data(token_map, token)
                        if heat_data is not None:
                            ax.imshow(heat_data, alpha=0.5, cmap='hot')
                    
                    ax.set_title(f"'{token}'", fontsize=12, fontweight='bold')
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"Failed to plot '{token}': {e}")
                    ax.text(0.5, 0.5, f"Error: {token}", ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            # Hide empty subplots
            for i in range(n_tokens, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'token_attention_gallery.png')
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Token gallery creation failed: {e}")
    
    def create_comparative_analysis(
        self,
        base_heat_map: Any,
        lora_heat_map: Any,
        base_image: Any,
        lora_image: Any,
        tokens: List[str],
        output_dir: str
    ) -> None:
        """
        Create comprehensive comparative analysis between base and LoRA results.
        """
        try:
            print("Creating comparative analysis...")
            
            # 1. Side-by-side image comparison
            self._create_image_comparison(base_image, lora_image, output_dir)
            
            # 2. Attention score comparison
            self._create_attention_comparison(base_heat_map, lora_heat_map, tokens, output_dir)
            
            # 3. Token-specific comparisons
            self._create_token_specific_comparisons(base_heat_map, lora_heat_map, base_image, lora_image, tokens, output_dir)
            
            print("Comparative analysis completed")
            
        except Exception as e:
            print(f"Comparative analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
    def create_comparative_analysis_multi(
        self,
        base_heat_map: Any,
        normal_heat_map: Any,
        adaptive_heat_map: Any,
        base_image: Any,
        normal_image: Any,
        adaptive_image: Any,
        tokens: List[str],
        output_dir: str
    ) -> None:
        """
        Create a three-way comparative analysis between base, normal LoRA, and adaptive LoRA.
        """
        print("Creating multi-LoRA comparative analysis...")

        try:
            output_path = os.path.join(output_dir, "token_comparisons_multi")
            os.makedirs(output_path, exist_ok=True)

            for token in tokens[:6]:  # Limit to a manageable number
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f"Token: '{token}' - Base vs Normal LoRA vs Adaptive LoRA", fontsize=16, fontweight='bold')

                # Row 1: Images
                images = [base_image, normal_image, adaptive_image]
                titles = ["Base Model", "Normal LoRA", "Adaptive LoRA"]
                for i in range(3):
                    axes[0, i].imshow(images[i])
                    axes[0, i].set_title(titles[i], fontsize=14)
                    axes[0, i].axis('off')

                # Row 2: Attention overlays
                maps = [
                    (base_heat_map, base_image),
                    (normal_heat_map, normal_image),
                    (adaptive_heat_map, adaptive_image)
                ]
                for i, (hm, img) in enumerate(maps):
                    try:
                        token_map = hm.compute_word_heat_map(token)
                        token_map.plot_overlay(img, ax=axes[1, i])
                        axes[1, i].set_title(f"{titles[i]} Attention", fontsize=12)
                    except Exception as e:
                        axes[1, i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[1, i].set_title(f"{titles[i]} Attention (Error)")
                    axes[1, i].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_path, f"{token}_triple_comparison.png")
                plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
                plt.close()
                print(f"Saved: {save_path}")

        except Exception as e:
            print(f"create_comparative_analysis_multi failed: {e}")
            import traceback
            traceback.print_exc()

    
    def _create_image_comparison(self, base_image: Any, lora_image: Any, output_dir: str) -> None:
        """Create side-by-side image comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(base_image)
        axes[0].set_title("Base Model (No LoRA)", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(lora_image)
        axes[1].set_title("With LoRA Applied", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'image_comparison.png')
        plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        print(f"Image comparison saved: {save_path}")
    
    def _create_attention_comparison(
        self, 
        base_heat_map: Any, 
        lora_heat_map: Any, 
        tokens: List[str], 
        output_dir: str
    ) -> None:
        """Create attention score comparison chart."""
        # Get attention scores for both models
        base_scores = self.attention_extractor.get_token_attention_scores(base_heat_map, tokens, run_label="base")
        lora_scores = self.attention_extractor.get_token_attention_scores(lora_heat_map, tokens, run_label="lora")
        
        # Find common tokens
        common_tokens = [token for token in tokens if token in base_scores and token in lora_scores]
        
        if not common_tokens:
            print("No common tokens found for attention comparison")
            return
        
        # Create comparison chart
        x_pos = np.arange(len(common_tokens))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute comparison
        base_values = [base_scores[token] for token in common_tokens]
        lora_values = [lora_scores[token] for token in common_tokens]
        
        bars1 = ax1.bar(x_pos - width/2, base_values, width, label='Base Model', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x_pos + width/2, lora_values, width, label='With LoRA', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Tokens')
        ax1.set_ylabel('Total Attention Score')
        ax1.set_title('Attention Score Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(common_tokens, rotation=45, ha='right')
        ax1.legend()
        
        # Relative change
        changes = [(lora_scores[token] - base_scores[token]) / base_scores[token] * 100 
                  for token in common_tokens if base_scores[token] > 0]
        valid_tokens = [token for token in common_tokens if base_scores[token] > 0]
        
        colors = ['green' if change > 0 else 'red' for change in changes]
        bars3 = ax2.bar(range(len(valid_tokens)), changes, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Tokens')
        ax2.set_ylabel('Percentage Change (%)')
        ax2.set_title('LoRA Effect on Attention (% Change)')
        ax2.set_xticks(range(len(valid_tokens)))
        ax2.set_xticklabels(valid_tokens, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, change in zip(bars3, changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{change:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'attention_comparison.png')
        plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        print(f"Attention comparison saved: {save_path}")
    
    def _create_token_specific_comparisons(
        self,
        base_heat_map: Any,
        lora_heat_map: Any,
        base_image: Any,
        lora_image: Any,
        tokens: List[str],
        output_dir: str
    ) -> None:
        """Create token-specific side-by-side comparisons."""
        print("Creating token-specific comparisons...")
        
        comparison_dir = os.path.join(output_dir, 'token_comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        for token in tokens[:4]:  # Limit to first 4 tokens to avoid too many files
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Token: '{token}' - Base vs LoRA Comparison", fontsize=14, fontweight='bold')
                
                # Base image
                axes[0, 0].imshow(base_image)
                axes[0, 0].set_title("Base Image")
                axes[0, 0].axis('off')
                
                # LoRA image
                axes[0, 1].imshow(lora_image)
                axes[0, 1].set_title("LoRA Image")
                axes[0, 1].axis('off')
                
                # Base attention
                try:
                    base_token_map = base_heat_map.compute_word_heat_map(token)
                    base_token_map.plot_overlay(base_image, ax=axes[1, 0])
                    axes[1, 0].set_title(f"Base Attention: '{token}'")
                except Exception as e:
                    axes[1, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title(f"Base Attention: '{token}' (Error)")
                axes[1, 0].axis('off')
                
                # LoRA attention
                try:
                    lora_token_map = lora_heat_map.compute_word_heat_map(token)
                    lora_token_map.plot_overlay(lora_image, ax=axes[1, 1])
                    axes[1, 1].set_title(f"LoRA Attention: '{token}'")
                except Exception as e:
                    axes[1, 1].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title(f"LoRA Attention: '{token}' (Error)")
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(comparison_dir, f'{token}_comparison.png')
                plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
                plt.close()
                
                print(f"'{token}': {save_path}")
                
            except Exception as e:
                print(f"'{token}': {e}")
        
        print(f"Token comparisons saved in: {comparison_dir}")
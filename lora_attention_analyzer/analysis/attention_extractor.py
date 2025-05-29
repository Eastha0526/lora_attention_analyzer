"""
Utilities for extracting and processing attention data from DAAM heat maps.
"""
import os
import json
import numpy as np
from datetime import datetime
import torch
from typing import Optional, Dict, Any, List, Tuple
import logging

class AttentionExtractor:
    """
    Comprehensive utilities for extracting attention data from DAAM heat maps.
    
    This class provides robust methods for:
    - Extracting numpy arrays from WordHeatMap objects
    - Computing attention scores for multiple tokens
    - Normalizing and processing attention maps
    - Computing pixel-wise dominant tokens
    """
    
    def __init__(
        self,
        log_dir:str = "./logs",
        log_file_prefix: str = "attention_extractor",
        also_write_txt: bool = False,
    ):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.json_path = os.path.join(log_dir, f"{log_file_prefix}_{ts}.json")
        self.txt_path = (
            os.path.join(log_dir, f"{log_file_prefix}_{ts}.txt") if also_write_txt else None
        )

        self.logger = logging.getLogger(f"AttentionExtractor_{ts}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

        if not self.logger.handlers:
            console_hdl = logging.StreamHandler()
            console_hdl.setFormatter(fmt)
            self.logger.addHandler(console_hdl)

            if self.txt_path:
                file_hdl = logging.FileHandler(self.txt_path)
                file_hdl.setFormatter(fmt)
                self.logger.addHandler(file_hdl)

        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(fmt)
        self.logger.addHandler(console_hdl)

        if self.txt_path:
            file_hdl = logging.FileHandler(self.txt_path)
            file_hdl.setFormatter(fmt)
            self.logger.addHandler(file_hdl)

        self._json_log: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
            "runs": [],
        }

    def extract_heat_data(self, token_heat_map: Any, token_name: Optional[str] = None, run_label:str = None) -> Optional[np.ndarray]:
        """
        Extract numpy array data safely from WordHeatMap objects.
        
        This method tries multiple strategies to extract attention data from DAAM's
        WordHeatMap objects, handling various data types and formats.
        
        Args:
            token_heat_map: WordHeatMap object from DAAM
            token_name: Name of token for debugging output (optional)
            
        Returns:
            Numpy array of heat data or None if extraction fails
        """
        if token_name:
            print(f"Extracting data for '{token_name}'")
        
        try:
            heat_data = None
            
            # Strategy 1: Check common attribute names
            common_attributes = ['heatmap', 'value', 'heat_map', 'data', 'attention_map']
            for attr_name in common_attributes:
                if hasattr(token_heat_map, attr_name):
                    heat_data = getattr(token_heat_map, attr_name)
                    if token_name:
                        print(f"Found data in '{attr_name}' attribute")
                    break
            
            # Strategy 2: If no data found, search all non-private attributes
            if heat_data is None:
                attrs = [attr for attr in dir(token_heat_map) 
                        if not attr.startswith('_') and not callable(getattr(token_heat_map, attr, None))]
                
                for attr_name in attrs:
                    try:
                        attr_value = getattr(token_heat_map, attr_name)
                        
                        # Check if it's a tensor or array with meaningful data
                        if self._is_valid_data(attr_value):
                            heat_data = attr_value
                            if token_name:
                                print(f"Found data in '{attr_name}' attribute")
                            break
                            
                    except Exception:
                        continue
            
            # Strategy 3: Process the found data
            if heat_data is not None:
                processed_data = self._process_heat_data(heat_data, token_name)
                if processed_data is not None:
                    if token_name:
                        print(f"Successfully extracted: shape={processed_data.shape}, dtype={processed_data.dtype}")
                    return processed_data
            
            # Strategy 4: Last resort - try to convert the object itself
            if token_name:
                print(f"Attempting direct conversion for '{token_name}'")
            
            try:
                direct_conversion = self._process_heat_data(token_heat_map, token_name)
                if direct_conversion is not None:
                    if token_name:
                        print(f"Direct conversion successful: shape={direct_conversion.shape}")
                    return direct_conversion
            except Exception:
                pass
            
            if token_name:
                print(f"Could not extract data for '{token_name}'")
            return None
            
        except Exception as e:
            if token_name:
                print(f"Error extracting data for '{token_name}': {str(e)}")
            return None
    
    def _is_valid_data(self, data: Any) -> bool:
        """Check if data appears to be valid tensor/array data."""
        # Check for PyTorch tensor
        if hasattr(data, 'detach') and hasattr(data, 'cpu'):
            return data.numel() > 1
        
        # Check for numpy array
        if isinstance(data, np.ndarray):
            return data.size > 1
        
        # Check for list/tuple that might contain numeric data
        if isinstance(data, (list, tuple)) and len(data) > 0:
            try:
                np.array(data)
                return True
            except:
                return False
        
        return False
    
    def _process_heat_data(self, heat_data: Any, token_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Process heat data into a standardized numpy array format.
        
        Args:
            heat_data: Raw heat data from various sources
            token_name: Token name for debugging
            
        Returns:
            Processed numpy array or None if processing fails
        """
        try:
            # Handle PyTorch Tensor
            if hasattr(heat_data, 'detach') and hasattr(heat_data, 'cpu'):
                if token_name:
                    print(f"Converting PyTorch tensor")
                
                # Handle tensor with gradients and device placement
                processed = heat_data.detach().cpu()
                
                # Convert to numpy
                if hasattr(processed, 'numpy'):
                    result = processed.numpy()
                else:
                    result = np.array(processed)
                
                return result if result.size > 0 else None
            
            # Handle numpy array
            elif isinstance(heat_data, np.ndarray):
                if token_name:
                    print(f"Data is already numpy array")
                return heat_data if heat_data.size > 0 else None
            
            # Handle other array-like objects
            else:
                if token_name:
                    print(f"Converting {type(heat_data)} to numpy")
                
                try:
                    result = np.array(heat_data)
                    return result if result.size > 0 else None
                except Exception:
                    return None
                    
        except Exception as e:
            if token_name:
                print(f"Processing failed: {str(e)}")
            return None
    
    def get_token_attention_scores(
        self, heat_map: Any, tokens: List[str], run_label: str = "run",
    ) -> Dict[str, float]:
        self.logger.info(f"{len(tokens)}개 토큰에 대한 attention 계산 시작")

        token_scores: Dict[str, float] = {}
        errors: Dict[str, str] = {}

        for idx, token in enumerate(tokens, 1):
            try:
                self.logger.info(f"{idx}/{len(tokens)} '{token}' 처리 중")

                # 토큰별 heat map 추출
                token_heat_map = heat_map.compute_word_heat_map(token)

                # numpy 데이터 추출
                heat_data = self.extract_heat_data(token_heat_map, token)

                if heat_data is not None:
                    total_attention = float(np.sum(heat_data))
                    token_scores[token] = total_attention
                    self.logger.info(f"    → '{token}': {total_attention:.4f}")
                else:
                    self.logger.warning(f"'{token}': 추출 실패 (데이터 없음)")
                    self._json_log["errors"][token] = "No data extracted"

            except Exception as e:
                msg = str(e)
                self.logger.error(f"'{token}': 오류 발생 - {msg}")
                self._json_log["errors"][token] = msg


        self.logger.info(
            f"attention 계산 완료: {len(token_scores)}/{len(tokens)}개 성공"
        )

        self._json_log["runs"].append(
            {
                "label": run_label,
                "timestamp": datetime.now().isoformat(),
                "token_scores": token_scores,
                "errors": errors,
            }
        )
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_log, f, indent=2, ensure_ascii=False)

        self.logger.info(f"로그 저장 완료 → {self.json_path}")

        if self.txt_path:
            self.logger.info(f"텍스트 로그 저장 완료 → {self.txt_path}")

        return token_scores
    
    def normalize_attention_maps(
        self, 
        token_maps: List[np.ndarray], 
        target_shape: Tuple[int, int],
        method: str = 'resize'
    ) -> List[np.ndarray]:
        """
        Normalize attention maps to the same shape using various methods.
        
        Args:
            token_maps: List of attention maps (numpy arrays)
            target_shape: Target shape (height, width) for normalization
            method: Normalization method ('resize', 'crop', 'pad')
            
        Returns:
            List of normalized attention maps
        """
        print(f"Normalizing {len(token_maps)} attention maps to {target_shape}")
        
        normalized_maps = []
        
        for i, token_map in enumerate(token_maps):
            original_shape = token_map.shape
            
            if original_shape[:2] == target_shape:
                # Already correct shape
                normalized_maps.append(token_map)
                continue
            
            print(f"Map {i+1}: {original_shape} -> {target_shape}")
            
            try:
                if method == 'resize':
                    normalized_map = self._resize_map(token_map, target_shape)
                elif method == 'crop':
                    normalized_map = self._crop_map(token_map, target_shape)
                elif method == 'pad':
                    normalized_map = self._pad_map(token_map, target_shape)
                else:
                    print(f"Unknown method '{method}', using resize")
                    normalized_map = self._resize_map(token_map, target_shape)
                
                normalized_maps.append(normalized_map)
                
            except Exception as e:
                print(f"Failed to normalize map {i+1}: {e}")
                # Fallback: use simple crop/pad
                normalized_map = self._simple_normalize(token_map, target_shape)
                normalized_maps.append(normalized_map)
        
        print(f"Normalized {len(normalized_maps)} maps successfully")
        return normalized_maps
    
    def _resize_map(self, token_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize map using available image processing libraries."""
        try:
            # Try scikit-image first (highest quality)
            from skimage.transform import resize
            return resize(token_map, target_shape, anti_aliasing=True, preserve_range=True)
        except ImportError:
            try:
                # Fallback to scipy
                from scipy.ndimage import zoom
                zoom_factors = (target_shape[0] / token_map.shape[0], 
                              target_shape[1] / token_map.shape[1])
                return zoom(token_map, zoom_factors)
            except ImportError:
                # Last resort: simple interpolation
                return self._simple_resize(token_map, target_shape)
    
    def _simple_resize(self, token_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Simple resize using numpy interpolation."""
        old_h, old_w = token_map.shape[:2]
        new_h, new_w = target_shape
        
        # Create coordinate grids
        y_coords = np.linspace(0, old_h - 1, new_h)
        x_coords = np.linspace(0, old_w - 1, new_w)
        
        # Simple nearest neighbor interpolation
        resized = np.zeros(target_shape, dtype=token_map.dtype)
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                resized[i, j] = token_map[int(round(y)), int(round(x))]
        
        return resized
    
    def _crop_map(self, token_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Crop map to target shape (center crop)."""
        old_h, old_w = token_map.shape[:2]
        new_h, new_w = target_shape
        
        # Calculate crop boundaries (center crop)
        start_h = max(0, (old_h - new_h) // 2)
        start_w = max(0, (old_w - new_w) // 2)
        end_h = min(old_h, start_h + new_h)
        end_w = min(old_w, start_w + new_w)
        
        cropped = token_map[start_h:end_h, start_w:end_w]
        
        # Pad if necessary
        if cropped.shape[:2] != target_shape:
            return self._pad_map(cropped, target_shape)
        
        return cropped
    
    def _pad_map(self, token_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Pad map to target shape."""
        old_h, old_w = token_map.shape[:2]
        new_h, new_w = target_shape
        
        if old_h >= new_h and old_w >= new_w:
            return self._crop_map(token_map, target_shape)
        
        # Calculate padding
        pad_h = max(0, new_h - old_h)
        pad_w = max(0, new_w - old_w)
        
        # Symmetric padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return np.pad(token_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    def _simple_normalize(self, token_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Simple normalization using crop and pad."""
        old_h, old_w = token_map.shape[:2]
        new_h, new_w = target_shape
        
        # Crop if larger
        if old_h > new_h or old_w > new_w:
            end_h = min(old_h, new_h)
            end_w = min(old_w, new_w)
            token_map = token_map[:end_h, :end_w]
        
        # Pad if smaller
        current_h, current_w = token_map.shape[:2]
        if current_h < new_h or current_w < new_w:
            pad_h = max(0, new_h - current_h)
            pad_w = max(0, new_w - current_w)
            token_map = np.pad(token_map, ((0, pad_h), (0, pad_w)), mode='constant')
        
        return token_map
    
    def compute_dominant_tokens(
        self, 
        token_maps: List[np.ndarray], 
        tokens: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute pixel-wise dominant tokens and related statistics.
        
        Args:
            token_maps: List of normalized attention maps
            tokens: List of corresponding token names
            
        Returns:
            Tuple of (dominant_indices_array, statistics_dict)
        """
        if len(token_maps) < 2:
            raise ValueError(f"Need at least 2 token maps for dominance analysis, got {len(token_maps)}")
        
        print(f"Computing dominant tokens for {len(token_maps)} maps")
        
        # Stack maps and compute dominance
        stacked_maps = np.stack(token_maps, axis=0)  # (num_tokens, height, width)
        dominant_indices = np.argmax(stacked_maps, axis=0)  # (height, width)
        
        # Compute statistics
        stats = {
            'total_pixels': dominant_indices.size,
            'dominant_counts': {},
            'dominant_percentages': {},
            'max_attention_values': {},
            'mean_attention_values': {}
        }
        
        for i, token in enumerate(tokens):
            # Count dominant pixels
            dominant_count = np.sum(dominant_indices == i)
            stats['dominant_counts'][token] = int(dominant_count)
            stats['dominant_percentages'][token] = float(dominant_count / dominant_indices.size * 100)
            
            # Attention statistics
            token_map = stacked_maps[i]
            stats['max_attention_values'][token] = float(np.max(token_map))
            stats['mean_attention_values'][token] = float(np.mean(token_map))
        
        print(f"Dominance analysis complete:")
        for token in tokens:
            pct = stats['dominant_percentages'][token]
            print(f"'{token}': {pct:.1f}% of pixels")
        
        return dominant_indices, stats
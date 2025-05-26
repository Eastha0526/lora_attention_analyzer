"""
Basic tests for LoRA Attention Analyzer.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from lora_attention_analyzer import LoRAAttentionPipeline, AttentionExtractor, AttentionVisualizer
from lora_attention_analyzer.core.lora_utils import LoRAUtils


class TestAttentionExtractor:
    """Test the AttentionExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = AttentionExtractor()
    
    def test_extract_heat_data_numpy_array(self):
        """Test extraction from numpy array."""
        # Mock WordHeatMap object with numpy array
        mock_heat_map = Mock()
        test_array = np.random.rand(64, 64)
        mock_heat_map.heatmap = test_array
        
        result = self.extractor.extract_heat_data(mock_heat_map, "test_token")
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, test_array)
    
    def test_extract_heat_data_torch_tensor(self):
        """Test extraction from PyTorch tensor."""
        mock_heat_map = Mock()
        test_tensor = torch.randn(64, 64)
        mock_heat_map.heatmap = test_tensor
        
        result = self.extractor.extract_heat_data(mock_heat_map, "test_token")
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)
    
    def test_extract_heat_data_empty_array(self):
        """Test handling of empty arrays."""
        mock_heat_map = Mock()
        mock_heat_map.heatmap = np.array([])
        
        result = self.extractor.extract_heat_data(mock_heat_map, "test_token")
        
        assert result is None
    
    def test_extract_heat_data_no_valid_attributes(self):
        """Test handling when no valid attributes are found."""
        mock_heat_map = Mock()
        # Remove all potential attributes
        mock_heat_map.configure_mock(**{attr: None for attr in ['heatmap', 'value', 'heat_map', 'data', 'attention_map']})
        
        result = self.extractor.extract_heat_data(mock_heat_map, "test_token")
        
        assert result is None
    
    def test_get_token_attention_scores(self):
        """Test getting attention scores for tokens."""
        # Mock heat map
        mock_heat_map = Mock()
        
        # Mock token heat maps
        mock_token_map1 = Mock()
        mock_token_map1.heatmap = np.ones((32, 32)) * 0.5
        
        mock_token_map2 = Mock()
        mock_token_map2.heatmap = np.ones((32, 32)) * 0.3
        
        mock_heat_map.compute_word_heat_map.side_effect = [mock_token_map1, mock_token_map2]
        
        tokens = ["token1", "token2"]
        scores = self.extractor.get_token_attention_scores(mock_heat_map, tokens)
        
        assert len(scores) == 2
        assert "token1" in scores
        assert "token2" in scores
        assert scores["token1"] > scores["token2"]  # First token has higher values
    
    def test_normalize_attention_maps(self):
        """Test attention map normalization."""
        # Create test maps with different shapes
        map1 = np.random.rand(32, 32)
        map2 = np.random.rand(48, 48)
        map3 = np.random.rand(64, 64)
        
        token_maps = [map1, map2, map3]
        target_shape = (64, 64)
        
        normalized_maps = self.extractor.normalize_attention_maps(token_maps, target_shape)
        
        assert len(normalized_maps) == 3
        for normalized_map in normalized_maps:
            assert normalized_map.shape == target_shape
    
    def test_compute_dominant_tokens(self):
        """Test computing dominant tokens."""
        # Create test maps where token 0 dominates top half, token 1 dominates bottom half
        map1 = np.zeros((64, 64))
        map1[:32, :] = 1.0  # Top half high
        
        map2 = np.zeros((64, 64))
        map2[32:, :] = 1.0  # Bottom half high
        
        token_maps = [map1, map2]
        tokens = ["token1", "token2"]
        
        dominant_indices = self.extractor.compute_dominant_tokens(token_maps, tokens)
        
        assert dominant_indices.shape == (64, 64)
        assert np.all(dominant_indices[:32, :] == 0)  # Top half should be token 0
        assert np.all(dominant_indices[32:, :] == 1)  # Bottom half should be token 1
    
    def test_compute_dominant_tokens_insufficient_maps(self):
        """Test error handling with insufficient token maps."""
        token_maps = [np.random.rand(32, 32)]  # Only one map
        tokens = ["token1"]
        
        with pytest.raises(ValueError, match="Need at least 2 token maps"):
            self.extractor.compute_dominant_tokens(token_maps, tokens)


class TestLoRAUtils:
    """Test the LoRAUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lora_utils = LoRAUtils()
    
    def test_find_lora_pairs(self):
        """Test finding LoRA up/down pairs."""
        keys = [
            "layer1.lora_up.weight",
            "layer1.lora_down.weight",
            "layer2.lora_up.weight",
            "layer2.lora_down.weight",
            "layer3.lora_up.weight",  # Missing down pair
        ]
        
        pairs = self.lora_utils._find_lora_pairs(keys)
        
        assert len(pairs) == 2
        assert "layer1" in pairs
        assert "layer2" in pairs
        assert "layer3" not in pairs  # Missing down pair
        
        assert pairs["layer1"] == ("layer1.lora_up.weight", "layer1.lora_down.weight")
        assert pairs["layer2"] == ("layer2.lora_up.weight", "layer2.lora_down.weight")
    
    def test_convert_sd_key_to_diffusers(self):
        """Test Stable Diffusion to Diffusers key conversion."""
        test_cases = [
            ("input_blocks.1.2", "down_blocks.1.attentions.2"),
            ("middle_block.1", "mid_block.attentions.0"),
            ("output_blocks.2.1", "up_blocks.2.attentions.1"),
        ]
        
        for sd_key, expected_diffusers_key in test_cases:
            result = self.lora_utils._convert_sd_key_to_diffusers(sd_key)
            assert expected_diffusers_key in result or result == expected_diffusers_key
    
    def test_convert_te_key_to_diffusers(self):
        """Test Text Encoder key conversion."""
        test_cases = [
            ("text_model_encoder_layers_0", "text_model.encoder.layers.0"),
            ("text_model_encoder_layers_5_mlp_fc1", "text_model.encoder.layers.5.mlp.fc1"),
        ]
        
        for te_key, expected_result in test_cases:
            result = self.lora_utils._convert_te_key_to_diffusers(te_key)
            assert expected_result in result


class TestAttentionVisualizer:
    """Test the AttentionVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = AttentionVisualizer()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('os.makedirs')
    def test_create_comprehensive_analysis(self, mock_makedirs, mock_close, mock_savefig):
        """Test comprehensive analysis creation."""
        # Mock heat map
        mock_heat_map = Mock()
        mock_token_map = Mock()
        mock_token_map.heatmap = np.random.rand(64, 64)
        mock_heat_map.compute_word_heat_map.return_value = mock_token_map
        
        # Mock image
        mock_image = Mock()
        mock_image.size = (512, 512)
        
        tokens = ["test_token1", "test_token2"]
        output_dir = "/tmp/test_output"
        
        # Should not raise any exceptions
        self.visualizer.create_comprehensive_analysis(
            mock_heat_map, mock_image, output_dir, tokens
        )
        
        # Verify directory creation was attempted
        mock_makedirs.assert_called()
        
        # Verify savefig was called multiple times (for different visualizations)
        assert mock_savefig.call_count >= 4  # At least 4 different visualizations


class TestLoRAAttentionPipeline:
    """Test the LoRAAttentionPipeline class."""
    
    @patch('lora_attention_analyzer.core.pipeline.StableDiffusionXLPipeline')
    def test_init(self, mock_pipeline_class):
        """Test pipeline initialization."""
        mock_pipe = Mock()
        mock_pipeline_class.from_single_file.return_value = mock_pipe
        
        pipeline = LoRAAttentionPipeline(
            model_id="/fake/model.safetensors",
            model_type="vpred",
            device="cpu"
        )
        
        assert pipeline.model_id == "/fake/model.safetensors"
        assert pipeline.model_type == "vpred"
        assert pipeline.device == "cpu"
        assert pipeline.pipe is not None
    
    def test_extract_tokens_from_prompt(self):
        """Test token extraction from prompt."""
        # We need to create a pipeline instance, but we can mock the heavy initialization
        with patch('lora_attention_analyzer.core.pipeline.StableDiffusionXLPipeline'):
            pipeline = LoRAAttentionPipeline(
                model_id="/fake/model.safetensors",
                device="cpu"
            )
        
        prompt = "a cute cat sitting in the garden, detailed artwork"
        tokens = pipeline._extract_tokens_from_prompt(prompt)
        
        # Should extract meaningful tokens and filter out stop words
        assert len(tokens) > 0
        assert "cute" in tokens or "cat" in tokens
        assert "sitting" in tokens or "garden" in tokens
        assert "the" not in tokens  # Stop word should be filtered
        assert "a" not in tokens  # Stop word should be filtered


class TestIntegration:
    """Integration tests."""
    
    def test_import_all_modules(self):
        """Test that all modules can be imported without errors."""
        from lora_attention_analyzer import LoRAAttentionPipeline, AttentionExtractor, AttentionVisualizer
        from lora_attention_analyzer.core.lora_utils import LoRAUtils
        from lora_attention_analyzer.cli.main import main, run_lora_comparison, analyze_lora_file
        
        # If we get here without ImportError, all imports work
        assert True
    
    def test_version_available(self):
        """Test that version information is available."""
        import lora_attention_analyzer
        
        assert hasattr(lora_attention_analyzer, '__version__')
        assert isinstance(lora_attention_analyzer.__version__, str)
        assert len(lora_attention_analyzer.__version__) > 0


# Utility functions for testing
def create_mock_safetensors_file(filepath, keys_data):
    """Create a mock safetensors file for testing."""
    # This would be used for more complex integration tests
    # For now, just create a placeholder
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()


def create_test_image():
    """Create a test image for testing."""
    from PIL import Image
    return Image.new('RGB', (512, 512), color='red')


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
"""
Tests for the synthetic review generation pipeline.
"""
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.utils.faker_utils import generate_product_info


class TestPipeline(unittest.TestCase):
    """Test cases for the generation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader("config.yaml")
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        config = self.config_loader.load()
        self.assertIsInstance(config, dict)
        self.assertIn("models", config)
    
    def test_product_info_generation(self):
        """Test product info generation."""
        product = generate_product_info(["electronics", "books"])
        self.assertIn("id", product)
        self.assertIn("name", product)
        self.assertIn("category", product)
        self.assertIn(product["category"], ["electronics", "books"])
    
    def test_directory_structure(self):
        """Test that required directories exist."""
        required_dirs = [
            "data/real_reviews",
            "data/generated",
            "logs/generation",
            "logs/quality",
            "logs/llm_judge"
        ]
        for dir_path in required_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Directory {dir_path} should exist")


if __name__ == "__main__":
    unittest.main()


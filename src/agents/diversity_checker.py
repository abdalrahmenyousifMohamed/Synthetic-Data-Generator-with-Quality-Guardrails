"""
Diversity checking agent for ensuring review diversity.
"""
from typing import Dict, Any, List
from src.quality.diversity_metrics import DiversityMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiversityChecker:
    """Agent responsible for checking review diversity."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diversity checker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("quality", {}).get("diversity", {})
        self.metrics = DiversityMetrics(config)
        self.min_unique_words = self.config.get("min_unique_words", 50)
        self.max_similarity = self.config.get("max_similarity_threshold", 0.8)
        self.review_history: List[str] = []
    
    def check(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check diversity of a review against previously generated reviews.
        
        Args:
            review: Review dictionary to check
            
        Returns:
            Diversity check results
        """
        review_text = review.get("review_text", "")
        results = {
            "passed": True,
            "metrics": {}
        }
        
        # Check unique word count
        unique_words = self.metrics.count_unique_words(review_text)
        results["metrics"]["unique_words"] = unique_words
        
        if unique_words < self.min_unique_words:
            results["passed"] = False
            results["reason"] = f"Insufficient unique words: {unique_words} < {self.min_unique_words}"
        
        # Check similarity to previous reviews
        if self.review_history:
            max_similarity = self.metrics.max_similarity(review_text, self.review_history)
            results["metrics"]["max_similarity"] = max_similarity
            
            if max_similarity > self.max_similarity:
                results["passed"] = False
                results["reason"] = f"Too similar to previous reviews: {max_similarity:.2f} > {self.max_similarity}"
        
        # Add to history
        self.review_history.append(review_text)
        
        return results
    
    def reset_history(self):
        """Reset the review history for a new generation session."""
        self.review_history = []


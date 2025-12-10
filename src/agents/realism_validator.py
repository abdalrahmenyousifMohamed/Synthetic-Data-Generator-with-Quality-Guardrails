"""
Realism validation agent for ensuring reviews appear realistic.
"""
from typing import Dict, Any
from src.quality.realism_validator import RealismValidator as QualityRealismValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealismValidator:
    """Agent responsible for validating review realism."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize realism validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("quality", {}).get("realism", {})
        self.validator = QualityRealismValidator(config)
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 500)
    
    def validate(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a review appears realistic.
        
        Args:
            review: Review dictionary to validate
            
        Returns:
            Realism validation results
        """
        review_text = review.get("review_text", "")
        results = {
            "passed": True,
            "checks": {}
        }
        
        # Length check
        length = len(review_text)
        results["checks"]["length"] = length
        
        if length < self.min_length:
            results["passed"] = False
            results["reason"] = f"Review too short: {length} < {self.min_length}"
        elif length > self.max_length:
            results["passed"] = False
            results["reason"] = f"Review too long: {length} > {self.max_length}"
        
        # Realism checks
        realism_score = self.validator.validate_realism(review_text)
        results["checks"]["realism_score"] = realism_score
        
        if realism_score < 0.5:
            results["passed"] = False
            if "reason" not in results:
                results["reason"] = f"Low realism score: {realism_score:.2f}"
        
        return results


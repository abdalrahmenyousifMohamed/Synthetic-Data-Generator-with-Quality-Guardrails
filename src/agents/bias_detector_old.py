"""
Bias detection agent for identifying potential biases in generated reviews.
"""
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BiasDetector:
    """Agent responsible for detecting biases in generated reviews."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bias detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("quality", {}).get("bias", {})
        self.check_sentiment_balance = self.config.get("check_sentiment_balance", True)
    
    def detect(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential biases in a review.
        
        Args:
            review: Review dictionary to analyze
            
        Returns:
            Bias detection results
        """
        review_text = review.get("review_text", "")
        results = {
            "passed": True,
            "issues": [],
            "sentiment": None
        }
        
        # Sentiment balance check
        if self.check_sentiment_balance:
            sentiment = self._analyze_sentiment(review_text)
            results["sentiment"] = sentiment
            
            # Check for extreme sentiment bias
            if sentiment in ["extremely_positive", "extremely_negative"]:
                results["issues"].append(f"Extreme sentiment detected: {sentiment}")
                results["passed"] = False
        
        # Check for demographic bias indicators
        bias_keywords = self._check_bias_keywords(review_text)
        if bias_keywords:
            results["issues"].extend(bias_keywords)
            results["passed"] = False
        
        return results
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis (can be enhanced with NLP libraries)."""
        positive_words = ["excellent", "amazing", "perfect", "love", "great", "wonderful", "fantastic"]
        negative_words = ["terrible", "awful", "horrible", "hate", "worst", "disappointed", "poor"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count + 3:
            return "extremely_positive"
        elif neg_count > pos_count + 3:
            return "extremely_negative"
        elif pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _check_bias_keywords(self, text: str) -> list:
        """Check for potentially biased language."""
        bias_patterns = [
            "all [group] are",
            "typical [group]",
            "always [group]",
        ]
        issues = []
        text_lower = text.lower()
        
        # This is a simplified check - can be enhanced
        for pattern in bias_patterns:
            if pattern.replace("[group]", "").replace("all ", "").replace("typical ", "").replace("always ", "") in text_lower:
                issues.append(f"Potential bias pattern detected: {pattern}")
        
        return issues


"""
Quality Validator Agent
Orchestrates all quality checks in a unified interface
"""

from typing import Dict, Any, List
from src.quality.diversity_metrics import DiversityMetrics
from src.quality.bias_detection import BiasDetector
from src.quality.realism_validator import RealismValidator


class QualityValidatorAgent:
    """
    Unified quality validation agent
    Combines diversity, bias, and realism checks
    """
    
    def __init__(
        self,
        diversity_metrics: DiversityMetrics,
        bias_detector: BiasDetector,
        realism_validator: RealismValidator
    ):
        self.diversity_metrics = diversity_metrics
        self.bias_detector = bias_detector
        self.realism_validator = realism_validator
    
    def validate_single_review(
        self,
        review_text: str,
        rating: int,
        existing_reviews: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a single review against all quality criteria
        """
        
        results = {
            "passed": True,
            "scores": {},
            "issues": []
        }
        
        # 1. Basic length and vocabulary check
        word_count = len(review_text.split())
        unique_words = len(set(review_text.lower().split()))
        
        results["scores"]["length"] = word_count
        results["scores"]["unique_words"] = unique_words
        
        if word_count < 50:
            results["passed"] = False
            results["issues"].append(f"Too short: {word_count} words")
        
        if word_count > 500:
            results["passed"] = False
            results["issues"].append(f"Too long: {word_count} words")
        
        if unique_words < 20:
            results["passed"] = False
            results["issues"].append(f"Low vocabulary: {unique_words} unique words")
        
        # 2. Diversity check (if we have existing reviews)
        if existing_reviews and len(existing_reviews) > 0:
            test_set = existing_reviews + [review_text]
            diversity_score = self.diversity_metrics.calculate_self_bleu(test_set)
            results["scores"]["diversity"] = diversity_score
            
            if diversity_score < 0.5:  # Too similar
                results["passed"] = False
                results["issues"].append(f"Too similar to existing reviews: {diversity_score:.2f}")
        else:
            results["scores"]["diversity"] = 1.0  # No comparison available
        
        # 3. Bias and realism check
        bias_result = self.bias_detector.detect_unrealistic_patterns(review_text, rating)
        results["scores"]["sentiment_alignment"] = bias_result["sentiment_mismatch"]["alignment_score"]
        results["scores"]["unrealistic_score"] = bias_result["unrealistic_score"]
        
        if bias_result["sentiment_mismatch"]["alignment_score"] < 0.7:
            results["passed"] = False
            results["issues"].append("Sentiment-rating mismatch")
        
        if bias_result["is_unrealistic"]:
            results["passed"] = False
            results["issues"].append("Unrealistic patterns detected")
        
        # 4. Realism validation
        realism_result = self.realism_validator.validate_single_review(review_text)
        results["scores"]["realism"] = realism_result["realism_score"]
        
        if realism_result["realism_score"] < 0.7:
            results["passed"] = False
            results["issues"].append(f"Low realism score: {realism_result['realism_score']:.2f}")
        
        return results
    
    def validate_dataset(
        self,
        reviews: List[Dict[str, Any]],
        target_distribution: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Validate entire dataset
        Returns comprehensive quality metrics
        """
        
        reviews_text = [r.get('review_text', r.get('text', '')) for r in reviews]
        ratings = [r.get('rating', 0) for r in reviews]
        
        # Overall diversity metrics
        diversity = self.diversity_metrics.calculate_overall_diversity(reviews_text)
        
        # Bias detection across dataset
        if target_distribution:
            bias_result = self.bias_detector.detect_sentiment_bias(
                reviews_text,
                ratings,
                target_distribution
            )
        else:
            bias_result = {"is_biased": False}
        
        # Individual review scores
        individual_scores = []
        for review in reviews:
            score = {
                "id": review.get('review_id', review.get('id', 'unknown')),
                "diversity": diversity["self_bleu_diversity"],
                "length": len(review.get('review_text', review.get('text', '')).split()),
                "rating": review.get('rating', 0)
            }
            individual_scores.append(score)
        
        return {
            "dataset_size": len(reviews),
            "overall_diversity": diversity,
            "bias_detection": bias_result,
            "individual_scores": individual_scores,
            "passed": not bias_result.get("is_biased", False) and diversity["overall_score"] > 0.6
        }
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import re


class RealismValidator:
    """Validate if synthetic reviews match characteristics of real reviews"""
    
    def __init__(self, real_reviews: List[str] = None):
        self.real_reviews = real_reviews or []
        self.real_stats = self._calculate_statistics(real_reviews) if real_reviews else {}
    
    def _calculate_statistics(self, reviews: List[str]) -> Dict[str, Any]:
        """Calculate statistical features of reviews"""
        if not reviews:
            return {}
        
        lengths = [len(review.split()) for review in reviews]
        unique_words_counts = [len(set(review.lower().split())) for review in reviews]
        sentence_counts = [len(review.split('.')) for review in reviews]
        
        # Word per sentence
        words_per_sentence = []
        for review in reviews:
            sentences = [s.strip() for s in review.split('.') if s.strip()]
            if sentences:
                avg_words = np.mean([len(s.split()) for s in sentences])
                words_per_sentence.append(avg_words)
        
        # Technical terms (simple heuristic: words with certain patterns)
        tech_term_pattern = re.compile(r'\b(?:API|SDK|UI|UX|integration|deployment|scalability|authentication)\b', re.IGNORECASE)
        tech_terms_per_review = [len(tech_term_pattern.findall(review)) for review in reviews]
        
        return {
            "avg_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "avg_unique_words": float(np.mean(unique_words_counts)),
            "avg_sentences": float(np.mean(sentence_counts)),
            "avg_words_per_sentence": float(np.mean(words_per_sentence)) if words_per_sentence else 0.0,
            "avg_tech_terms": float(np.mean(tech_terms_per_review)),
            "length_range": (int(min(lengths)), int(max(lengths)))
        }
    
    def compare_with_real(self, synthetic_reviews: List[str]) -> Dict[str, Any]:
        """Compare synthetic reviews with real reviews"""
        
        if not self.real_reviews:
            return {
                "realism_score": 0.5,
                "comparison": "No real reviews available for comparison"
            }
        
        synthetic_stats = self._calculate_statistics(synthetic_reviews)
        
        # Calculate differences
        length_diff = abs(synthetic_stats["avg_length"] - self.real_stats["avg_length"]) / self.real_stats["avg_length"]
        unique_words_diff = abs(synthetic_stats["avg_unique_words"] - self.real_stats["avg_unique_words"]) / self.real_stats["avg_unique_words"]
        sentence_diff = abs(synthetic_stats["avg_sentences"] - self.real_stats["avg_sentences"]) / self.real_stats["avg_sentences"]
        
        # Calculate realism score (lower difference = higher realism)
        # Target: within 20% of real reviews
        realism_score = 1.0 - min(1.0, (length_diff + unique_words_diff + sentence_diff) / 3)
        
        return {
            "realism_score": float(realism_score),
            "synthetic_stats": synthetic_stats,
            "real_stats": self.real_stats,
            "differences": {
                "length_diff_pct": float(length_diff * 100),
                "unique_words_diff_pct": float(unique_words_diff * 100),
                "sentence_diff_pct": float(sentence_diff * 100)
            },
            "comparison_quality": "excellent" if realism_score > 0.9 else "good" if realism_score > 0.8 else "moderate" if realism_score > 0.7 else "needs_improvement"
        }
    
    def validate_single_review(self, review: str) -> Dict[str, Any]:
        """Validate a single review for realism"""
        
        stats = self._calculate_statistics([review])
        
        # Check if within reasonable bounds
        checks = {
            "length_reasonable": 50 <= stats["avg_length"] <= 500,
            "has_variation": stats["avg_unique_words"] / stats["avg_length"] > 0.5 if stats["avg_length"] > 0 else False,
            "sentence_structure": 1 <= stats["avg_sentences"] <= 20
        }
        
        realism_score = sum(checks.values()) / len(checks)
        
        return {
            "realism_score": float(realism_score),
            "checks": checks,
            "stats": stats
        }
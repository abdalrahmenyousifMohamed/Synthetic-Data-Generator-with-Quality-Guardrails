"""
Production-Ready Bias Detection with Adaptive Thresholds and Circuit Breaker
FIXED: Ensure attempt_number parameter is properly defined
"""
from typing import Dict, Any, Optional
import re
from collections import defaultdict
import time


try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, using keyword-based sentiment")


class BiasDetector:
    """
    1. Adaptive thresholds based on rating
    2. Circuit breaker for stuck reviews
    3. Confidence-weighted scoring
    4. Graceful degradation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        
        self.alignment_thresholds = {
            1: 0.25,  
            2: 0.30,  
            3: 0.20,  
            4: 0.35,  
            5: 0.40   
        }
        
        
        self.failure_counts = defaultdict(int)
        self.last_reset = time.time()
        self.reset_interval = 300  
        
        
        self.sentiment_analyzer = None
        self.analyzer_type = "keyword"
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading sentiment model...")
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if torch.cuda.is_available() else -1,
                    truncation=True,
                    max_length=512
                )
                self.analyzer_type = "5-star"
                logger.info("✓ 5-star sentiment model loaded")
            except Exception as e:
                logger.warning(f"Transformer load failed: {e}. Using keywords.")
                self.sentiment_analyzer = None
    
    def detect_unrealistic_patterns(
        self, 
        review_text: str, 
        rating: int,
        attempt_number: int = 1  
    ) -> Dict[str, Any]:
        """
        
        Args:
            review_text: Review content
            rating: Star rating (1-5)
            attempt_number: Current attempt (for adaptive leniency) - DEFAULT=1
        
        Returns:
            Dict with detection results including adaptive thresholds
        """
        
        
        self._maybe_reset_circuit_breaker()
        
        
        sentiment_result = self._check_sentiment_alignment(
            review_text, 
            rating,
            attempt_number
        )
        
        
        generic_result = self._check_generic_patterns(review_text)
        
        
        unrealistic_score = 0.0
        
        
        if sentiment_result.get("is_mismatch", False):
            
            mismatch_weight = max(0.4, 0.6 - (attempt_number * 0.1))
            confidence_factor = sentiment_result.get("confidence", 0.5)
            unrealistic_score += mismatch_weight * confidence_factor
        
        
        if generic_result.get("is_generic", False):
            unrealistic_score += 0.2 * generic_result.get("generic_score", 0)
        
        
        if generic_result.get("repetition_score", 0) > 0.4:
            unrealistic_score += 0.1
        
        
        
        rejection_threshold = max(0.55 - (attempt_number * 0.05), 0.35)
        
        is_unrealistic = unrealistic_score > rejection_threshold
        
        
        if is_unrealistic:
            self.failure_counts[rating] += 1
            
            
            if self.failure_counts[rating] >= 5:
                logger.warning(
                    f"Circuit breaker: {self.failure_counts[rating]} failures for "
                    f"{rating}-star. Reducing strictness."
                )
                is_unrealistic = unrealistic_score > 0.70  
        
        return {
            "is_unrealistic": is_unrealistic,
            "unrealistic_score": min(unrealistic_score, 1.0),
            "sentiment_mismatch": sentiment_result,
            "generic_patterns": generic_result,
            "threshold_used": rejection_threshold,
            "circuit_breaker_active": self.failure_counts[rating] >= 5,
            "attempt_number": attempt_number  
        }
    
    def _check_sentiment_alignment(
        self, 
        review_text: str, 
        rating: int,
        attempt_number: int = 1
    ) -> Dict[str, Any]:
        
        
        base_threshold = self.alignment_thresholds.get(rating, 0.30)
        
        
        adjusted_threshold = max(base_threshold - (attempt_number * 0.05), 0.15)
        
        if self.sentiment_analyzer and self.analyzer_type == "5-star":
            return self._transformer_sentiment_5star(
                review_text, 
                rating, 
                adjusted_threshold
            )
        else:
            return self._keyword_sentiment(
                review_text, 
                rating, 
                adjusted_threshold
            )
    
    def _transformer_sentiment_5star(
        self, 
        review_text: str, 
        rating: int,
        threshold: float
    ) -> Dict[str, Any]:
        """Enhanced 5-star sentiment with  safeguards"""
        
        try:
            
            text = review_text[:512]
            
            
            result = self.sentiment_analyzer(text)[0]
            predicted_rating = int(result['label'].split()[0])
            confidence = result['score']
            
            
            predicted_sentiment = (
                "NEGATIVE" if predicted_rating <= 2 else
                "NEUTRAL" if predicted_rating == 3 else
                "POSITIVE"
            )
            
            expected_sentiment = (
                "NEGATIVE" if rating <= 2 else
                "NEUTRAL" if rating == 3 else
                "POSITIVE"
            )
            
            
            rating_difference = abs(predicted_rating - rating)
            
            
            if rating_difference == 0:
                alignment_score = confidence  
            elif rating_difference == 1:
                alignment_score = confidence * 0.7  
            elif rating_difference == 2:
                alignment_score = confidence * 0.4  
            else:
                alignment_score = confidence * 0.2  
            
            
            is_strong_mismatch = rating_difference >= 3 and confidence > 0.75
            is_moderate_mismatch = rating_difference == 2 and confidence > 0.85
            is_mismatch = is_strong_mismatch or is_moderate_mismatch
            
            
            if alignment_score >= threshold:
                is_mismatch = False
            
            return {
                "is_mismatch": is_mismatch,
                "sentiment": predicted_sentiment,
                "predicted_rating": predicted_rating,
                "confidence": confidence,
                "expected_sentiment": expected_sentiment,
                "alignment_score": alignment_score,
                "rating": rating,
                "rating_difference": rating_difference,
                "threshold_used": threshold,
                "method": "transformer-5star"
            }
            
        except Exception as e:
            logger.error(f"Transformer sentiment failed: {e}")
            return self._keyword_sentiment(review_text, rating, threshold)
    
    def _keyword_sentiment(
        self, 
        review_text: str, 
        rating: int,
        threshold: float
    ) -> Dict[str, Any]:
        """Enhanced keyword-based sentiment with better accuracy"""
        
        text_lower = review_text.lower()
        
        
        strong_positive = [
            "excellent", "amazing", "perfect", "love", "fantastic",
            "outstanding", "exceptional", "brilliant", "wonderful",
            "impressed", "exceeded", "superb", "flawless", "awesome"
        ]
        
        positive = [
            "great", "good", "helpful", "easy", "satisfied", "recommend",
            "useful", "efficient", "reliable", "solid", "happy", "pleased",
            "effective", "smooth", "clear", "fast", "simple"
        ]
        
        strong_negative = [
            "terrible", "awful", "horrible", "hate", "worst", "useless",
            "garbage", "pathetic", "disappointing", "frustrated", "regret",
            "nightmare", "disaster", "broken", "failure"
        ]
        
        negative = [
            "poor", "bad", "difficult", "confusing", "complicated", "slow",
            "buggy", "lacking", "limited", "issues", "problems", "annoying",
            "clunky", "frustrating", "missing", "weak"
        ]
        
        
        strong_pos = sum(2 for w in strong_positive if w in text_lower)
        pos = sum(1 for w in positive if w in text_lower)
        strong_neg = sum(2 for w in strong_negative if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        
        pos_score = strong_pos + pos
        neg_score = strong_neg + neg
        
        
        if pos_score > neg_score + 1:
            sentiment = "POSITIVE"
            confidence = min(0.5 + (pos_score - neg_score) * 0.08, 0.90)
        elif neg_score > pos_score + 1:
            sentiment = "NEGATIVE"
            confidence = min(0.5 + (neg_score - pos_score) * 0.08, 0.90)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.45
        
        
        expected_sentiment = (
            "NEGATIVE" if rating <= 2 else
            "NEUTRAL" if rating == 3 else
            "POSITIVE"
        )
        
        
        if sentiment == expected_sentiment:
            alignment_score = confidence
        elif expected_sentiment == "NEUTRAL":
            alignment_score = 0.5  
        else:
            alignment_score = 1 - confidence
        
        
        is_mismatch = False
        if rating >= 4 and sentiment == "NEGATIVE" and confidence > 0.70:
            is_mismatch = True
        elif rating <= 2 and sentiment == "POSITIVE" and confidence > 0.70:
            is_mismatch = True
        
        
        if alignment_score >= threshold:
            is_mismatch = False
        
        return {
            "is_mismatch": is_mismatch,
            "sentiment": sentiment,
            "confidence": confidence,
            "expected_sentiment": expected_sentiment,
            "alignment_score": alignment_score,
            "rating": rating,
            "threshold_used": threshold,
            "method": "keyword-based"
        }
    
    def _check_generic_patterns(self, review_text: str) -> Dict[str, Any]:
        """Lighter generic pattern detection"""
        
        text_lower = review_text.lower()
        
        
        generic_phrases = [
            "game changer", "life changing", "changed my life",
            "best thing ever", "total game changer", "absolutely perfect",
            "couldn't be happier", "exceeded all expectations"
        ]
        
        found = [p for p in generic_phrases if p in text_lower]
        generic_score = len(found) / max(len(generic_phrases), 1)
        
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', review_text) 
                     if len(s.strip()) > 10]
        
        repetition_score = 0.0
        if len(sentences) > 1:
            starts = [tuple(s.split()[:2]) for s in sentences 
                      if len(s.split()) >= 2]
            if len(starts) > 1:
                unique_starts = len(set(starts))
                repetition_score = 1 - (unique_starts / len(starts))
        
        
        exclaim_ratio = review_text.count('!') / max(len(sentences), 1)
        
        
        is_generic = (
            len(found) >= 2 or
            (repetition_score > 0.5 and exclaim_ratio > 0.4)
        )
        
        return {
            "is_generic": is_generic,
            "generic_score": generic_score,
            "found_generic_phrases": found,
            "repetition_score": repetition_score,
            "exclamation_ratio": exclaim_ratio
        }
    
    def _maybe_reset_circuit_breaker(self):
        """Reset failure counts periodically"""
        current_time = time.time()
        if current_time - self.last_reset > self.reset_interval:
            if any(count > 0 for count in self.failure_counts.values()):
                logger.info("Resetting circuit breaker failure counts")
            self.failure_counts.clear()
            self.last_reset = current_time
    
    def detect(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy compatibility method
        Maintains backward compatibility with older code
        """
        review_text = review.get("review_text", "")
        rating = review.get("rating", 3)
        attempt = review.get("regeneration_count", 0) + 1
        
        detection = self.detect_unrealistic_patterns(
            review_text, 
            rating, 
            attempt_number=attempt  
        )
        
        return {
            "passed": not detection["is_unrealistic"],
            "issues": [
                f"Sentiment mismatch (score: {detection['sentiment_mismatch']['alignment_score']:.2f})"
            ] if detection["is_unrealistic"] else [],
            "sentiment": detection["sentiment_mismatch"].get("sentiment"),
            "details": detection
        }
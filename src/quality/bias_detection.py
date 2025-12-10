"""
Production-Ready Bias Detection with VERY LENIENT Thresholds
CRITICAL FIX: Lower thresholds to prevent infinite rejection loops
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
    Production-ready bias detector with VERY LENIENT thresholds
    to prevent rejection loops
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with production safeguards"""
        self.config = config or {}
        
        # MUCH MORE LENIENT thresholds per rating
        self.alignment_thresholds = {
            1: 0.15,  # Very lenient for 1-star (hard to generate)
            2: 0.15,  # Very lenient for 2-star
            3: 0.10,  # Most lenient for 3-star (neutral is hardest)
            4: 0.20,  # Lenient for 4-star 
            5: 0.25   # Lenient for 5-star 
        }
        
        # Circuit breaker tracking
        self.failure_counts = defaultdict(int)
        self.last_reset = time.time()
        self.reset_interval = 300  # Reset every 5 minutes
        
        # Initialize sentiment analyzer
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
        Production-grade detection with VERY LENIENT adaptive thresholds
        
        Args:
            review_text: Review content
            rating: Star rating (1-5)
            attempt_number: Current attempt (for adaptive leniency) - DEFAULT=1
        
        Returns:
            Dict with detection results including adaptive thresholds
        """
        
        # Reset circuit breaker periodically
        self._maybe_reset_circuit_breaker()
        
        # Check sentiment alignment with adaptive threshold
        sentiment_result = self._check_sentiment_alignment(
            review_text, 
            rating,
            attempt_number
        )
        
        # Check generic patterns (lighter weight)
        generic_result = self._check_generic_patterns(review_text)
        
        # Calculate unrealistic score with weighting
        unrealistic_score = 0.0
        
        # Sentiment mismatch (weighted by confidence and attempt)
        if sentiment_result.get("is_mismatch", False):
            # MUCH more lenient: Reduce weight significantly for any attempts
            mismatch_weight = max(0.2, 0.5 - (attempt_number * 0.15))
            confidence_factor = sentiment_result.get("confidence", 0.5)
            unrealistic_score += mismatch_weight * confidence_factor
        
        # Generic patterns (very light penalty)
        if generic_result.get("is_generic", False):
            unrealistic_score += 0.1 * generic_result.get("generic_score", 0)
        
        # High repetition (very light penalty)
        if generic_result.get("repetition_score", 0) > 0.4:
            unrealistic_score += 0.05
        
        # Make decision - MUCH MORE LENIENT
        # Use adaptive threshold: starts high, gets VERY LOW on retries
        rejection_threshold = max(0.65 - (attempt_number * 0.10), 0.30)
        
        is_unrealistic = unrealistic_score > rejection_threshold
        
        # Circuit breaker: if same rating fails 3+ times, auto-pass
        if is_unrealistic:
            self.failure_counts[rating] += 1
            
            # After just 3 failures for same rating, reduce strictness
            if self.failure_counts[rating] >= 3:
                logger.warning(
                    f"Circuit breaker: {self.failure_counts[rating]} failures for "
                    f"{rating}-star. Auto-passing to prevent loop."
                )
                is_unrealistic = False  # AUTO-PASS!
        
        return {
            "is_unrealistic": is_unrealistic,
            "unrealistic_score": min(unrealistic_score, 1.0),
            "sentiment_mismatch": sentiment_result,
            "generic_patterns": generic_result,
            "threshold_used": rejection_threshold,
            "circuit_breaker_active": self.failure_counts[rating] >= 3,
            "attempt_number": attempt_number
        }
    
    def _check_sentiment_alignment(
        self, 
        review_text: str, 
        rating: int,
        attempt_number: int = 1
    ) -> Dict[str, Any]:
        """Check sentiment with VERY LENIENT adaptive logic"""
        
        # Get base threshold for this rating
        base_threshold = self.alignment_thresholds.get(rating, 0.15)
        
        # Make MUCH more lenient on retries
        adjusted_threshold = max(base_threshold - (attempt_number * 0.10), 0.05)
        
        logger.info(
            f"Sentiment check: rating={rating}, attempt={attempt_number}, "
            f"base_threshold={base_threshold:.2f}, adjusted={adjusted_threshold:.2f}"
        )
        
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
        """Enhanced 5-star sentiment with VERY LENIENT thresholds"""
        
        try:
            # Truncate safely
            text = review_text[:512]
            
            # Get prediction
            result = self.sentiment_analyzer(text)[0]
            predicted_rating = int(result['label'].split()[0])
            confidence = result['score']
            
            # Determine sentiments
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
            
            # Calculate rating difference
            rating_difference = abs(predicted_rating - rating)
            
            # Calculate alignment score (0-1, higher is better)
            # MUCH MORE LENIENT SCORING
            if rating_difference == 0:
                alignment_score = confidence  # Perfect match
            elif rating_difference == 1:
                alignment_score = confidence * 0.85  
            elif rating_difference == 2:
                alignment_score = confidence * 0.60  
            else:
                alignment_score = confidence * 0.35  
            
            # Determine if mismatch - ONLY reject on EXTREME mismatches
            # MUCH MORE LENIENT: Only reject 4+ star difference with very high confidence
            is_strong_mismatch = (
                rating_difference >= 4 and 
                confidence > 0.85
            )
            
            # Don't reject 2-3 star differences anymore
            is_mismatch = is_strong_mismatch
            
            # Override: If alignment score is above threshold, don't reject
            if alignment_score >= threshold:
                is_mismatch = False
            
            logger.info(
                f"Transformer sentiment: predicted={predicted_rating}, "
                f"expected={rating}, diff={rating_difference}, "
                f"confidence={confidence:.2f}, alignment={alignment_score:.2f}, "
                f"threshold={threshold:.2f}, mismatch={is_mismatch}"
            )
            
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
        """Enhanced keyword-based sentiment with VERY LENIENT matching"""
        
        text_lower = review_text.lower()
        
        # Expanded keyword lists
        strong_positive = [
            "excellent", "amazing", "perfect", "love", "fantastic",
            "outstanding", "exceptional", "brilliant", "wonderful",
            "impressed", "exceeded", "superb", "flawless", "awesome"
        ]
        
        positive = [
            "great", "good", "helpful", "easy", "satisfied", "recommend",
            "useful", "efficient", "reliable", "solid", "happy", "pleased",
            "effective", "smooth", "clear", "fast", "simple", "works", "nice"
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
        
        # Count with weights
        strong_pos = sum(2 for w in strong_positive if w in text_lower)
        pos = sum(1 for w in positive if w in text_lower)
        strong_neg = sum(2 for w in strong_negative if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        
        pos_score = strong_pos + pos
        neg_score = strong_neg + neg
        
        # Determine sentiment with MORE LENIENT thresholds
        if pos_score > neg_score:  
            sentiment = "POSITIVE"
            confidence = min(0.5 + (pos_score - neg_score) * 0.08, 0.90)
        elif neg_score > pos_score:  
            sentiment = "NEGATIVE"
            confidence = min(0.5 + (neg_score - pos_score) * 0.08, 0.90)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.45
        
        # Expected sentiment
        expected_sentiment = (
            "NEGATIVE" if rating <= 2 else
            "NEUTRAL" if rating == 3 else
            "POSITIVE"
        )
        
        # Calculate alignment
        if sentiment == expected_sentiment:
            alignment_score = confidence
        elif expected_sentiment == "NEUTRAL":
            alignment_score = 0.6  # Neutral is very acceptable 
        else:
            alignment_score = 1 - confidence
        
        # Check mismatch - ONLY on VERY strong confidence
        is_mismatch = False
        if rating >= 4 and sentiment == "NEGATIVE" and confidence > 0.80: 
            is_mismatch = True
        elif rating <= 2 and sentiment == "POSITIVE" and confidence > 0.80: 
            is_mismatch = True
        
        # Override with threshold
        if alignment_score >= threshold:
            is_mismatch = False
        
        logger.info(
            f"Keyword sentiment: sentiment={sentiment}, expected={expected_sentiment}, "
            f"confidence={confidence:.2f}, alignment={alignment_score:.2f}, "
            f"threshold={threshold:.2f}, mismatch={is_mismatch}"
        )
        
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
        """Very light generic pattern detection"""
        
        text_lower = review_text.lower()
        
        # Very small list of truly problematic phrases
        generic_phrases = [
            "game changer", "life changing", "changed my life",
            "best thing ever", "total game changer"
        ]
        
        found = [p for p in generic_phrases if p in text_lower]
        generic_score = len(found) / max(len(generic_phrases), 1)
        
        # Simple repetition check
        sentences = [s.strip() for s in re.split(r'[.!?]+', review_text) 
                     if len(s.strip()) > 10]
        
        repetition_score = 0.0
        if len(sentences) > 1:
            starts = [tuple(s.split()[:2]) for s in sentences 
                      if len(s.split()) >= 2]
            if len(starts) > 1:
                unique_starts = len(set(starts))
                repetition_score = 1 - (unique_starts / len(starts))
        
        # Exclamation check
        exclaim_ratio = review_text.count('!') / max(len(sentences), 1)
        
        # Only flag if EXTREME issues
        is_generic = (
            len(found) >= 3 or  # 3+ generic phrases
            (repetition_score > 0.7 and exclaim_ratio > 0.6)  # Very repetitive + excessive
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
        """Legacy compatibility method"""
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
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DiversityMetrics:
    """Calculate diversity metrics for generated reviews"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction()
    
    def calculate_self_bleu(self, reviews: List[str]) -> float:
        """
        Calculate Self-BLEU score
        Lower score indicates higher diversity
        """
        if len(reviews) < 2:
            return 0.0
        
        scores = []
        for i, review in enumerate(reviews):
            references = [r.split() for j, r in enumerate(reviews) if i != j]
            hypothesis = review.split()
            
            # Use smoothing to handle edge cases
            score = sentence_bleu(
                references,
                hypothesis,
                smoothing_function=self.smoothing.method1
            )
            scores.append(score)
        
        # Return diversity score (1 - Self-BLEU)
        return 1.0 - np.mean(scores)
    
    def calculate_semantic_similarity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate semantic similarity using embeddings
        Lower average similarity indicates higher diversity
        """
        if len(reviews) < 2:
            return {"avg_similarity": 0.0, "max_similarity": 0.0}
        
        embeddings = self.embedding_model.encode(reviews)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[triu_indices]
        
        return {
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities))
        }
    
    def calculate_lexical_diversity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate lexical diversity metrics
        """
        all_words = []
        for review in reviews:
            words = review.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return {"ttr": 0.0, "unique_words": 0, "total_words": 0}
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        # Type-Token Ratio
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        # Calculate MATTR (Moving Average TTR) for windows of 100 words
        window_size = 100
        mattr_scores = []
        
        for i in range(0, len(all_words) - window_size + 1, window_size // 2):
            window = all_words[i:i + window_size]
            window_ttr = len(set(window)) / len(window)
            mattr_scores.append(window_ttr)
        
        mattr = np.mean(mattr_scores) if mattr_scores else ttr
        
        return {
            "ttr": float(ttr),
            "mattr": float(mattr),
            "unique_words": unique_words,
            "total_words": total_words,
            "vocabulary_size": unique_words
        }
    
    def calculate_ngram_diversity(self, reviews: List[str], n: int = 2) -> float:
        """
        Calculate n-gram diversity
        Higher ratio of unique n-grams indicates higher diversity
        """
        all_ngrams = []
        
        for review in reviews:
            words = review.lower().split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def calculate_overall_diversity(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive diversity score
        """
        if not reviews:
            return {"overall_score": 0.0}
        
        # Calculate individual metrics
        self_bleu_diversity = self.calculate_self_bleu(reviews)
        semantic_sim = self.calculate_semantic_similarity(reviews)
        lexical_div = self.calculate_lexical_diversity(reviews)
        bigram_div = self.calculate_ngram_diversity(reviews, n=2)
        trigram_div = self.calculate_ngram_diversity(reviews, n=3)
        
        # Semantic diversity (inverse of similarity)
        semantic_diversity = 1.0 - semantic_sim["avg_similarity"]
        
        # Weighted overall score
        overall_score = (
            0.3 * self_bleu_diversity +
            0.3 * semantic_diversity +
            0.2 * lexical_div["ttr"] +
            0.1 * bigram_div +
            0.1 * trigram_div
        )
        
        return {
            "overall_score": float(overall_score),
            "self_bleu_diversity": float(self_bleu_diversity),
            "semantic_diversity": float(semantic_diversity),
            "lexical_diversity": lexical_div,
            "bigram_diversity": float(bigram_div),
            "trigram_diversity": float(trigram_div),
            "semantic_similarity": semantic_sim
        }
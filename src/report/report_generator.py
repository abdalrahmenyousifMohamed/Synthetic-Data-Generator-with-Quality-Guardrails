from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime
from src.utils.metrics_tracker import PipelineMetrics
from src.quality.diversity_metrics import DiversityMetrics
from src.quality.realism_validator import RealismValidator


class QualityReportGenerator:
    """Generate comprehensive quality reports"""
    
    def __init__(
        self,
        metrics: PipelineMetrics,
        diversity_metrics: DiversityMetrics,
        realism_validator: RealismValidator,
        generated_reviews: List[Dict[str, Any]]
    ):
        self.metrics = metrics
        self.diversity_metrics = diversity_metrics
        self.realism_validator = realism_validator
        self.generated_reviews = generated_reviews
    
    def generate_markdown_report(self, output_path: str):
        """Generate comprehensive Markdown quality report"""
        
        reviews_text = [r['review_text'] for r in self.generated_reviews]
        ratings = [r['rating'] for r in self.generated_reviews]
        
        # Calculate diversity metrics
        diversity = self.diversity_metrics.calculate_overall_diversity(reviews_text)
        
        # Calculate realism comparison
        realism = self.realism_validator.compare_with_real(reviews_text)
        
        # Build report
        report = self._build_report_content(diversity, realism, ratings)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
    
    def _build_report_content(
        self,
        diversity: Dict[str, Any],
        realism: Dict[str, Any],
        ratings: List[int]
    ) -> str:
        """Build the complete report content"""
        
        report = f"""# Synthetic Review Dataset Quality Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Generation Summary

- **Target Samples**: {self.metrics.target_samples}
- **Samples Generated**: {self.metrics.total_generated}
- **Samples Accepted**: {self.metrics.total_accepted} ({self.metrics.success_rate*100:.1f}% success rate)
- **Samples Rejected**: {self.metrics.total_rejected}
- **Total Regenerations**: {self.metrics.total_regenerations}
- **Generation Time**: {self.metrics.total_time/60:.1f} minutes
- **Total Cost**: ${self.metrics.total_cost:.2f}
- **LLM Judge Evaluations**: {self.metrics.llm_judge_evaluations}

## Model Performance Comparison

| Model | Samples | Accepted | Accept Rate | Avg Quality | LLM Judge Score | Avg Time | Cost | Cost/Sample |
|-------|---------|----------|-------------|-------------|-----------------|----------|------|-------------|
"""
        
        for model_name, model_metrics in self.metrics.model_metrics.items():
            report += f"| {model_name} | {model_metrics.samples_generated} | {model_metrics.samples_accepted} | {model_metrics.acceptance_rate*100:.1f}% | {model_metrics.avg_quality_score:.2f} | {model_metrics.avg_llm_judge_score:.2f} | {model_metrics.avg_time_per_sample:.2f}s | ${model_metrics.total_cost:.2f} | ${model_metrics.cost_per_sample:.3f} |\n"
        
        report += f"""
## Diversity Metrics

- **Overall Diversity Score**: {diversity['overall_score']:.3f} {'✓' if diversity['overall_score'] > 0.6 else '✗'}
- **Self-BLEU Diversity**: {diversity['self_bleu_diversity']:.3f} (higher is better)
- **Semantic Diversity**: {diversity['semantic_diversity']:.3f} (higher is better)
- **Vocabulary Size**: {diversity['lexical_diversity']['vocabulary_size']} unique tokens
- **Type-Token Ratio**: {diversity['lexical_diversity']['ttr']:.3f}
- **MATTR**: {diversity['lexical_diversity']['mattr']:.3f}
- **Bigram Diversity**: {diversity['bigram_diversity']:.3f}
- **Trigram Diversity**: {diversity['trigram_diversity']:.3f}

### Semantic Similarity Analysis

- **Average Similarity**: {diversity['semantic_similarity']['avg_similarity']:.3f} {'✓' if diversity['semantic_similarity']['avg_similarity'] < 0.7 else '✗'}
- **Max Similarity**: {diversity['semantic_similarity']['max_similarity']:.3f}
- **Min Similarity**: {diversity['semantic_similarity']['min_similarity']:.3f}

## LLM-as-a-Judge Evaluation Results

**PRIMARY JUDGE**: Gemini 2.0 Flash Exp (as specified in requirements)

### Overall Judge Metrics

- **Average Overall Score**: {self.metrics.llm_judge_pass_rate:.2f}
- **Pass Rate**: {self.metrics.llm_judge_pass_rate*100:.1f}%
- **Judge Agreement Rate**: {self.metrics.judge_agreement_rate*100:.1f}%

### Judge Performance Insights

The LLM judge system (Gemini 2.0 Flash Exp as primary) evaluated all {self.metrics.llm_judge_evaluations} reviews across 4 dimensions:
- Authenticity (natural language, specific details, balanced perspective)
- Alignment (sentiment-rating match)
- Expertise (domain knowledge, technical accuracy)
- Uniqueness (originality, non-templated structure)

## Rating Distribution

"""
        
        # Rating distribution
        from collections import Counter
        rating_counts = Counter(ratings)
        total = len(ratings)
        
        for rating in [1, 2, 3, 4, 5]:
            count = rating_counts.get(rating, 0)
            pct = (count / total * 100) if total > 0 else 0
            report += f"- **{rating}-Star**: {count} ({pct:.1f}%)\n"
        
        report += f"""
## Synthetic vs Real Comparison

**Realism Score**: {realism.get('realism_score', 0):.2f}/1.00 {'✓' if realism.get('realism_score', 0) > 0.7 else '✗'}

| Metric | Synthetic | Real | Difference |
|--------|-----------|------|------------|
"""
        
        if 'synthetic_stats' in realism and 'real_stats' in realism:
            syn_stats = realism['synthetic_stats']
            real_stats = realism['real_stats']
            
            report += f"| Avg Length (words) | {syn_stats['avg_length']:.1f} | {real_stats['avg_length']:.1f} | {realism['differences']['length_diff_pct']:.1f}% |\n"
            report += f"| Avg Unique Words | {syn_stats['avg_unique_words']:.1f} | {real_stats['avg_unique_words']:.1f} | {realism['differences']['unique_words_diff_pct']:.1f}% |\n"
            report += f"| Avg Sentences | {syn_stats['avg_sentences']:.1f} | {real_stats['avg_sentences']:.1f} | {realism['differences']['sentence_diff_pct']:.1f}% |\n"
        
        report += f"""
**Comparison Quality**: {realism.get('comparison_quality', 'N/A')}

## Rejection Analysis

Total Rejections: {self.metrics.total_rejected}

### Top Rejection Reasons

"""
        
        # Sort rejection reasons by count
        sorted_reasons = sorted(
            self.metrics.rejection_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons[:10]:
            pct = (count / self.metrics.total_rejected * 100) if self.metrics.total_rejected > 0 else 0
            report += f"- {reason}: {count} ({pct:.1f}%)\n"
        
        report += f"""
## Cost & Performance Analysis

### Generation Costs

"""
        
        for model_name, model_metrics in self.metrics.model_metrics.items():
            report += f"- **{model_name}**: ${model_metrics.total_cost:.2f} ({model_metrics.samples_accepted} samples) = ${model_metrics.cost_per_sample:.4f}/sample\n"
        
        report += f"""
**Total Generation Cost**: ${self.metrics.total_cost:.2f}

### Time Performance

- **Total Pipeline Time**: {self.metrics.total_time/60:.1f} minutes
- **Throughput**: {self.metrics.total_accepted/(self.metrics.total_time/60):.1f} samples/minute
- **Average Time per Accepted Sample**: {self.metrics.total_time/self.metrics.total_accepted:.1f}s

## Recommendations

1. **Quality Improvements**:
   - Current diversity score: {diversity['overall_score']:.2f}
   - {"Strong diversity across reviews" if diversity['overall_score'] > 0.7 else "Consider increasing seed word variety"}

2. **Cost Optimization**:
"""
        
        # Find most cost-effective model
        best_model = min(
            self.metrics.model_metrics.items(),
            key=lambda x: x[1].cost_per_sample if x[1].samples_accepted > 0 else float('inf')
        )
        
        report += f"   - Most cost-effective: {best_model[0]} at ${best_model[1].cost_per_sample:.4f}/sample\n"
        
        report += f"""
3. **Performance Optimization**:
   - Current throughput: {self.metrics.total_accepted/(self.metrics.total_time/60):.1f} samples/minute
   - {"Efficient generation speed" if self.metrics.total_accepted/(self.metrics.total_time/60) > 5 else "Consider parallel processing for faster generation"}

4. **LLM Judge Insights**:
   - Pass rate: {self.metrics.llm_judge_pass_rate*100:.1f}%
   - Judge agreement: {self.metrics.judge_agreement_rate*100:.1f}%
   - Using Gemini 2.0 Flash Exp as primary judge provides fast, cost-effective evaluation

---

*Report generated by Synthetic Review Generator v1.0*
"""
        
        return report
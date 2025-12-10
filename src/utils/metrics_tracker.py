from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class ModelMetrics:
    """Track metrics per model"""
    model_name: str
    samples_generated: int = 0
    samples_accepted: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    avg_quality_score: float = 0.0
    avg_llm_judge_score: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    llm_judge_scores: List[float] = field(default_factory=list)
    
    @property
    def acceptance_rate(self) -> float:
        if self.samples_generated == 0:
            return 0.0
        return self.samples_accepted / self.samples_generated
    
    @property
    def avg_time_per_sample(self) -> float:
        if self.samples_generated == 0:
            return 0.0
        return self.total_time / self.samples_generated
    
    @property
    def cost_per_sample(self) -> float:
        if self.samples_generated == 0:
            return 0.0
        return self.total_cost / self.samples_generated


@dataclass
class PipelineMetrics:
    """Track overall pipeline metrics"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    target_samples: int = 0
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_regenerations: int = 0
    
    model_metrics: Dict[str, ModelMetrics] = field(default_factory=dict)
    
    # LLM Judge metrics
    llm_judge_evaluations: int = 0
    llm_judge_pass_rate: float = 0.0
    judge_agreement_rate: float = 0.0
    
    # Quality metrics aggregates - FIX: Initialize properly
    avg_diversity_score: float = 0.0
    avg_self_bleu: float = 0.0
    avg_sentiment_alignment: float = 0.0
    avg_realism_score: float = 0.0
    
    # Track aggregated scores
    _diversity_scores: List[float] = field(default_factory=list)
    _self_bleu_scores: List[float] = field(default_factory=list)
    _sentiment_scores: List[float] = field(default_factory=list)
    _realism_scores: List[float] = field(default_factory=list)
    
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    
    @property
    def total_time(self) -> float:
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_accepted / self.total_generated
    
    @property
    def total_cost(self) -> float:
        return sum(m.total_cost for m in self.model_metrics.values())
    
    def add_rejection_reason(self, reason: str):
        """Track rejection reasons"""
        if reason not in self.rejection_reasons:
            self.rejection_reasons[reason] = 0
        self.rejection_reasons[reason] += 1
    
    def update_aggregate_scores(self, statistical_metrics: Dict[str, Any]):
        """Update aggregate quality scores from statistical metrics"""
        if 'self_bleu_diversity' in statistical_metrics:
            self._diversity_scores.append(statistical_metrics['self_bleu_diversity'])
            self.avg_diversity_score = sum(self._diversity_scores) / len(self._diversity_scores)
            self.avg_self_bleu = self.avg_diversity_score  # Same value, different name
        
        if 'sentiment_alignment' in statistical_metrics:
            self._sentiment_scores.append(statistical_metrics['sentiment_alignment'])
            self.avg_sentiment_alignment = sum(self._sentiment_scores) / len(self._sentiment_scores)
        
        # Note: Realism score comes from realism validator, not always available
        if 'realism_score' in statistical_metrics:
            self._realism_scores.append(statistical_metrics['realism_score'])
            self.avg_realism_score = sum(self._realism_scores) / len(self._realism_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        data['total_time'] = self.total_time
        data['success_rate'] = self.success_rate
        data['total_cost'] = self.total_cost
        
        # Remove internal tracking lists from export
        data.pop('_diversity_scores', None)
        data.pop('_self_bleu_scores', None)
        data.pop('_sentiment_scores', None)
        data.pop('_realism_scores', None)
        
        return data
    
    def save(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsTracker:
    """Central metrics tracking system"""
    
    def __init__(self, target_samples: int):
        self.metrics = PipelineMetrics(target_samples=target_samples)
    
    def init_model(self, model_name: str):
        """Initialize tracking for a model"""
        if model_name not in self.metrics.model_metrics:
            self.metrics.model_metrics[model_name] = ModelMetrics(model_name=model_name)
    
    def record_generation(
        self,
        model_name: str,
        accepted: bool,
        time_taken: float,
        cost: float,
        quality_score: float = None,
        llm_judge_score: float = None,
        statistical_metrics: Dict[str, Any] = None
    ):
        """Record a generation attempt"""
        self.init_model(model_name)
        model_metrics = self.metrics.model_metrics[model_name]
        
        model_metrics.samples_generated += 1
        self.metrics.total_generated += 1
        
        if accepted:
            model_metrics.samples_accepted += 1
            self.metrics.total_accepted += 1
            
            if quality_score is not None:
                model_metrics.quality_scores.append(quality_score)
            
            if llm_judge_score is not None:
                model_metrics.llm_judge_scores.append(llm_judge_score)
            
            # Update aggregate scores - FIX: Actually calculate them
            if statistical_metrics:
                self.metrics.update_aggregate_scores(statistical_metrics)
        else:
            self.metrics.total_rejected += 1
        
        model_metrics.total_time += time_taken
        model_metrics.total_cost += cost
    
    def record_llm_judge(self, passed: bool, judges_agreed: bool):
        """Record LLM judge evaluation"""
        self.metrics.llm_judge_evaluations += 1
        
        if passed:
            # Update pass rate
            pass_count = int(self.metrics.llm_judge_pass_rate * (self.metrics.llm_judge_evaluations - 1))
            pass_count += 1
            self.metrics.llm_judge_pass_rate = pass_count / self.metrics.llm_judge_evaluations
        
        if judges_agreed:
            # Update agreement rate
            agree_count = int(self.metrics.judge_agreement_rate * (self.metrics.llm_judge_evaluations - 1))
            agree_count += 1
            self.metrics.judge_agreement_rate = agree_count / self.metrics.llm_judge_evaluations
    
    def finalize(self):
        """Finalize metrics calculation"""
        self.metrics.end_time = datetime.now()
        
        # Calculate average scores for each model
        for model_metrics in self.metrics.model_metrics.values():
            if model_metrics.quality_scores:
                model_metrics.avg_quality_score = sum(model_metrics.quality_scores) / len(model_metrics.quality_scores)
            
            if model_metrics.llm_judge_scores:
                model_metrics.avg_llm_judge_score = sum(model_metrics.llm_judge_scores) / len(model_metrics.llm_judge_scores)
    
    def get_metrics(self) -> PipelineMetrics:
        """Get current metrics"""
        return self.metrics
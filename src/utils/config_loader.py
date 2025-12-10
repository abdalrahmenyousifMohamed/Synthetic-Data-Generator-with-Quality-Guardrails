import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LocalModelConfig(BaseModel):
    """Configuration for local models"""
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0", etc.
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_memory: Optional[Dict[int, str]] = None
    

class ModelConfig(BaseModel):
    provider: str  # "openai", "google", "local"
    model: str
    weight: float
    temperature: float = 1.0
    max_tokens: int = 1024
    api_key_env: Optional[str] = None  # Not needed for local models
    local_config: Optional[Dict[str, Any]] = None  # For local model settings


class PersonaConfig(BaseModel):
    name: str
    characteristics: str
    experience: str
    focus: list[str]
    weight: float


class LLMJudgeConfig(BaseModel):
    primary_model: str
    secondary_model: str
    temperature: float = 0.2
    enable_multi_judge: bool = True
    parallel_execution: bool = True
    max_workers: int = 10
    evaluation_criteria: list[str]


class QualityThresholds(BaseModel):
    min_length: int = 50
    max_length: int = 500
    min_unique_words: int = 20
    max_self_bleu: float = 0.5
    min_sentiment_alignment: float = 0.7
    min_diversity_score: float = 0.6
    min_realism_score: float = 0.7
    min_llm_judge_score: float = 0.7
    min_authenticity_score: float = 0.7
    min_alignment_score: float = 0.6
    min_expertise_score: float = 0.6
    min_uniqueness_score: float = 0.6
    max_regeneration_attempts: int = 3


class GenerationConfig(BaseModel):
    target_samples: int
    batch_size: int
    temperature: float
    seed_word_count: int
    domain: str


class Config(BaseModel):
    generation: GenerationConfig
    models: list[ModelConfig]
    llm_judge: LLMJudgeConfig
    personas: list[PersonaConfig]
    rating_distribution: Dict[str, float]
    review_characteristics: Dict[str, Any]
    quality_thresholds: QualityThresholds
    logging: Dict[str, Any]


def load_config(config_path: str = "config.yaml") -> Config:
    """Load and validate configuration from YAML file"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Load API keys from environment for API-based models
    for model_config in config_data['models']:
        if model_config['provider'] in ['openai', 'google']:
            api_key_env = model_config.get('api_key_env')
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if not api_key:
                    raise ValueError(f"API key not found in environment: {api_key_env}")
    
    config = Config(**config_data)
    
    # Validate rating distribution sums to 1.0
    rating_sum = sum(config.rating_distribution.values())
    if not 0.99 <= rating_sum <= 1.01:
        raise ValueError(f"Rating distribution must sum to 1.0, got {rating_sum}")
    
    # Validate persona weights sum to 1.0
    persona_weight_sum = sum(p.weight for p in config.personas)
    if not 0.99 <= persona_weight_sum <= 1.01:
        raise ValueError(f"Persona weights must sum to 1.0, got {persona_weight_sum}")
    
    return config


def get_api_key(env_var: str) -> str:
    """Get API key from environment"""
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"API key not found: {env_var}")
    return api_key
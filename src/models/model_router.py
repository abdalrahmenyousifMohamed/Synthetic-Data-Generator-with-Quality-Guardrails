from typing import Dict, Any, Optional
import random
from src.models.openai_client import OpenAIClient
from src.models.gemini_client import GeminiClient
from src.models.local_client import LocalModelClient
from src.models.openrouter_client import OpenRouterClient
from src.models.groq_client import GroqClient  # ✅ Add Groq import
from src.utils.config_loader import ModelConfig


class ModelRouter:
    """Route generation requests to appropriate models based on weights"""
    
    def __init__(self, model_configs: list[ModelConfig]):
        self.models = {}
        self.weights = {}
        self.local_models = {}  # Cache local models
        
        for config in model_configs:
            if config.provider == "openai":
                api_key = self._get_api_key(config.api_key_env)
                client = OpenAIClient(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature
                )
            
            elif config.provider == "google":
                api_key = self._get_api_key(config.api_key_env)
                client = GeminiClient(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature
                )
            
            elif config.provider == "openrouter":
                api_key = self._get_api_key(config.api_key_env)
                client = OpenRouterClient(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature
                )
            
            elif config.provider == "groq":  # ✅ Add Groq support
                api_key = self._get_api_key(config.api_key_env)
                client = GroqClient(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature
                )
            
            elif config.provider == "local":
                # Local models are loaded on-demand or pre-loaded
                client = self._get_or_load_local_model(config)
            
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
            
            model_key = f"{config.provider}:{config.model}"
            self.models[model_key] = client
            self.weights[model_key] = config.weight
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not 0.99 <= weight_sum <= 1.01:
            raise ValueError(f"Model weights must sum to 1.0, got {weight_sum}")
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment"""
        import os
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found: {env_var}")
        return api_key
    
    def _get_or_load_local_model(self, config: ModelConfig) -> LocalModelClient:
        """Get or load local model"""
        model_key = f"{config.provider}:{config.model}"
        
        # Check if already loaded
        if model_key in self.local_models:
            return self.local_models[model_key]
        
        # Load new local model
        local_config = config.local_config or {}
        
        client = LocalModelClient(
            model_name=config.model,
            temperature=config.temperature,
            device=local_config.get("device", "mps"),
        )
        
        self.local_models[model_key] = client
        return client
    
    def select_model(self) -> str:
        """Select a model based on weights"""
        models = list(self.weights.keys())
        weights = list(self.weights.values())
        return random.choices(models, weights=weights, k=1)[0]
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using selected or random model"""
        
        if model_key is None:
            model_key = self.select_model()
        
        if model_key not in self.models:
            raise ValueError(f"Model not found: {model_key}")
        
        client = self.models[model_key]
        result = client.generate(system_prompt, user_prompt, **kwargs)
        result["model_key"] = model_key
        
        return result
    
    def get_all_models(self) -> list[str]:
        """Get list of all available models"""
        return list(self.models.keys())
    
    def unload_local_models(self):
        """Unload all local models from memory"""
        for model_key, client in self.local_models.items():
            if isinstance(client, LocalModelClient):
                client.unload()
        self.local_models.clear()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.unload_local_models()
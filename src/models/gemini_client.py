import google.generativeai as genai
import os
import time
from typing import Dict, Any, Optional


class GeminiClient:
    """Google Gemini API client for review generation"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 1.0):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name=model)
        
        # Pricing estimates (per 1M tokens)
        self.pricing = {
            "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
            "gemini-pro": {"input": 0.5, "output": 1.5}
        }
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate text using Gemini API"""
        
        start_time = time.time()
        
        generation_config = {
            "temperature": temperature or self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_tokens,
        }
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            print(f"🔍 GEMINI RAW RESPONSE: {response.text[:500]}") 
            
            generation_time = time.time() - start_time
            
            # Estimate tokens (rough approximation)
            input_tokens = len(full_prompt.split()) * 1.3
            output_tokens = len(response.text.split()) * 1.3
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return {
                "text": response.text,
                "model": self.model_name,
                "time": generation_time,
                "cost": cost,
                "tokens": {
                    "input": int(input_tokens),
                    "output": int(output_tokens),
                    "total": int(input_tokens + output_tokens)
                },
                "success": True
            }
        
        except Exception as e:
            return {
                "text": None,
                "model": self.model_name,
                "time": time.time() - start_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost"""
        model_pricing = self.pricing.get(
            self.model_name, 
            {"input": 0.5, "output": 1.5}
        )
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
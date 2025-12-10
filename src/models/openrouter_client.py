import requests
import time
from typing import Dict, Any, Optional


class OpenRouterClient:
    """
    Client for OpenRouter API - access hundreds of models through one API
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",  # Default to free model
        temperature: float = 1.0,
        site_url: str = "https://github.com/yourusername/synthetic-reviews",
        site_name: str = "Synthetic Review Generator"
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.site_url = site_url
        self.site_name = site_name
        
        # Pricing (per 1M tokens) - will be fetched from API response
        self.pricing = {
            "input": 0.0,
            "output": 0.0
        }
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate text using OpenRouter API"""
        
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            generation_time = time.time() - start_time
            
            # Extract generated text
            generated_text = data['choices'][0]['message']['content']
            
            # Calculate token usage and cost
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
            
            # Calculate cost from usage
            # OpenRouter returns cost in the response
            cost = 0.0
            if 'usage' in data and 'total_cost' in data['usage']:
                cost = data['usage']['total_cost']
            else:
                # Fallback: estimate from model pricing
                # Free models have 0 cost
                if ':free' in self.model:
                    cost = 0.0
                else:
                    # Default rough estimate if not provided
                    cost = (input_tokens * 0.0001 + output_tokens * 0.0003) / 1000
            
            return {
                "text": generated_text.strip(),
                "model": self.model,
                "time": generation_time,
                "cost": cost,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens
                },
                "success": True
            }
        
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e.response, 'text'):
                error_msg = f"{error_msg} - Response: {e.response.text}"
            
            return {
                "text": None,
                "model": self.model,
                "time": time.time() - start_time,
                "cost": 0.0,
                "error": error_msg,
                "success": False
            }
        
        except Exception as e:
            return {
                "text": None,
                "model": self.model,
                "time": time.time() - start_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }
    
    def get_available_models(self) -> list:
        """Fetch list of available models from OpenRouter"""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
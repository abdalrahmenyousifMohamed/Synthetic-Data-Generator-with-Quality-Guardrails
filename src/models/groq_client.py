import requests
import time
import json
from typing import Dict, Any, Optional
from collections import deque


class GroqClient:
    """
    Client for Groq API with built-in rate limiting and retry logic
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 1.0,
        debug: bool = False,
        requests_per_minute: int = 25,  # Conservative limit (30 is max)
        tokens_per_minute: int = 10000,  # Conservative limit (12000 is max)
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        
        # Track requests and tokens in the last minute
        self.request_times = deque(maxlen=requests_per_minute)
        self.token_usage = deque(maxlen=100)  # Track last 100 requests
        
    def _wait_for_rate_limit(self):
        """Wait if we're hitting rate limits"""
        now = time.time()
        
        # Check request rate limit
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = self.request_times[0]
            time_since_oldest = now - oldest_request
            
            if time_since_oldest < 60:
                wait_time = 60 - time_since_oldest + 0.5  # Add buffer
                if self.debug:
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s for request limit...")
                time.sleep(wait_time)
        
        # Check token rate limit (last minute)
        recent_tokens = sum(
            tokens for timestamp, tokens in self.token_usage
            if now - timestamp < 60
        )
        
        if recent_tokens > self.tokens_per_minute * 0.8:  # 80% threshold
            if self.debug:
                print(f"[RATE LIMIT] Token usage high ({recent_tokens}/{self.tokens_per_minute}), slowing down...")
            time.sleep(2)  # Brief pause
        
        # Record this request
        self.request_times.append(now)
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate text using Groq API with rate limiting and retries"""
        
        for attempt in range(self.max_retries):
            try:
                # Wait if needed for rate limits
                self._wait_for_rate_limit()
                
                start_time = time.time()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
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
                
                if self.debug:
                    print(f"\n[DEBUG] Groq Request (attempt {attempt + 1}/{self.max_retries}):")
                    print(f"  Model: {self.model}")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Handle rate limit errors with exponential backoff
                if response.status_code == 429:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', '')
                    
                    # Extract wait time from error message if available
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    
                    if 'Please try again in' in error_msg:
                        # Try to extract the exact wait time
                        try:
                            # Extract milliseconds from message
                            if 'ms' in error_msg:
                                ms = float(error_msg.split('Please try again in ')[1].split('ms')[0].replace(',', ''))
                                wait_time = (ms / 1000) + 0.5  # Add buffer
                            elif 's.' in error_msg:
                                wait_time = float(error_msg.split('Please try again in ')[1].split('s.')[0]) + 0.5
                        except:
                            pass
                    
                    if attempt < self.max_retries - 1:
                        if self.debug:
                            print(f"[RATE LIMIT] 429 error, waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "text": None,
                            "model": self.model,
                            "time": time.time() - start_time,
                            "cost": 0.0,
                            "error": f"Rate limit exceeded after {self.max_retries} retries: {error_msg}",
                            "success": False
                        }
                
                response.raise_for_status()
                data = response.json()
                
                generation_time = time.time() - start_time
                
                # Validate response structure
                if 'choices' not in data or len(data['choices']) == 0:
                    error_msg = f"Invalid response structure: {json.dumps(data)[:200]}"
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        "text": None,
                        "model": self.model,
                        "time": generation_time,
                        "cost": 0.0,
                        "error": error_msg,
                        "success": False
                    }
                
                choice = data['choices'][0]
                if 'message' not in choice or 'content' not in choice['message']:
                    error_msg = f"Invalid choice structure: {json.dumps(choice)[:200]}"
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        "text": None,
                        "model": self.model,
                        "time": generation_time,
                        "cost": 0.0,
                        "error": error_msg,
                        "success": False
                    }
                
                generated_text = choice['message']['content']
                
                # Clean the generated text
                generated_text = self._clean_text(generated_text)
                
                if not generated_text or generated_text.strip() == "":
                    error_msg = "Generated text is empty"
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        "text": None,
                        "model": self.model,
                        "time": generation_time,
                        "cost": 0.0,
                        "error": error_msg,
                        "success": False
                    }
                
                # Calculate token usage
                usage = data.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                
                # Track token usage for rate limiting
                self.token_usage.append((time.time(), total_tokens))
                
                # Calculate cost
                cost = (input_tokens * 0.0000001 + output_tokens * 0.0000001)
                
                if self.debug:
                    print(f"[DEBUG] Success! {len(generated_text.split())} words, {total_tokens} tokens")
                
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
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_msg = f"{error_msg} - {error_data.get('error', {}).get('message', e.response.text)}"
                    except:
                        error_msg = f"{error_msg} - Response: {e.response.text}"
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    if self.debug:
                        print(f"[ERROR] Request failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                return {
                    "text": None,
                    "model": self.model,
                    "time": time.time() - start_time,
                    "cost": 0.0,
                    "error": error_msg,
                    "success": False
                }
            
            except Exception as e:
                error_msg = str(e)
                if self.debug:
                    print(f"[ERROR] Exception: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                
                return {
                    "text": None,
                    "model": self.model,
                    "time": time.time() - start_time,
                    "cost": 0.0,
                    "error": error_msg,
                    "success": False
                }
        
        # Should never reach here
        return {
            "text": None,
            "model": self.model,
            "time": 0,
            "cost": 0.0,
            "error": "Max retries exceeded",
            "success": False
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean generated text by removing unwanted formatting"""
        if not text:
            return text
        
        text = text.strip()
        
        # Remove wrapping quotation marks
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Remove markdown code blocks
        if text.startswith('```') and text.endswith('```'):
            lines = text.split('\n')
            if len(lines) > 2:
                text = '\n'.join(lines[1:-1]).strip()
        
        return text
    
    def get_available_models(self) -> list:
        """Fetch list of available models from Groq"""
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
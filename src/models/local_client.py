import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from typing import Dict, Any, Optional
import gc


class LocalModelClient:
    """
    Local model client for running models like Phi-3.5-mini-instruct
    Supports quantization for reduced memory usage
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        temperature: float = 1.0,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.device = device
        
        print(f"Loading local model: {model_name}...")
        print(f"Device: {device}, 8-bit: {load_in_8bit}, 4-bit: {load_in_4bit}")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif device == "auto":
            model_kwargs["device_map"] = "auto"
        
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if device != "auto" and not quantization_config:
            self.model.to(device)
        
        self.model.eval()
        
        # Calculate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        self.model_size_gb = param_count * 2 / (1024**3)  # Assuming float16
        
        print(f"✓ Model loaded: {param_count/1e9:.2f}B parameters (~{self.model_size_gb:.2f}GB)")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 256  # ✅ Reduced from 1024 - reviews don't need that many tokens
    ) -> Dict[str, Any]:
        """Generate text using local model"""
        
        start_time = time.time()
        
        # Format prompt based on model's chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback formatting
                formatted_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Store input length for later
            input_length = inputs['input_ids'].shape[1]
            
            # Show generation is starting
            print(f"🔄 Generating with {self.model_name}... (this may take 30-60s on MPS)")
            
            # ✅ FIX: Disable cache to avoid DynamicCache compatibility issues
            # This is slower but works with all transformers versions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature or self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # ✅ CRITICAL FIX: Disable cache to avoid seen_tokens error
                    repetition_penalty=1.1,  # Add slight repetition penalty
                )
            
            # Decode ONLY the new tokens (skip input prompt)
            generated_ids = outputs[0][input_length:]
            
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            generation_time = time.time() - start_time
            
            # Calculate token counts
            output_tokens = len(generated_ids)
            
            # 🔍 DEBUG: Print what was generated
            if output_tokens < 10:  # Only print warning if very short
                print(f"⚠️ WARNING: Only generated {output_tokens} tokens")
                print(f"  Generated text: '{generated_text}'")
            
            # ⚠️ SAFETY CHECK: If empty, return error
            if not generated_text.strip():
                print(f"❌ ERROR: Model generated empty text!")
                print(f"  Input length: {input_length}")
                print(f"  Output length: {outputs.shape[1]}")
                print(f"  New tokens: {output_tokens}")
                return {
                    "text": None,
                    "model": self.model_name,
                    "time": generation_time,
                    "cost": 0.0,
                    "error": "Model generated empty response",
                    "success": False
                }
            
            # Local models have no API cost
            cost = 0.0
            
            # Clean up
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()  # For Mac M1/M2
            
            return {
                "text": generated_text.strip(),
                "model": self.model_name,
                "time": generation_time,
                "cost": cost,
                "tokens": {
                    "input": input_length,
                    "output": output_tokens,
                    "total": input_length + output_tokens
                },
                "success": True
            }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ ERROR in generate(): {error_details}")
            
            return {
                "text": None,
                "model": self.model_name,
                "time": time.time() - start_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }
    
    def unload(self):
        """Unload model from memory"""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"✓ Model {self.model_name} unloaded from memory")
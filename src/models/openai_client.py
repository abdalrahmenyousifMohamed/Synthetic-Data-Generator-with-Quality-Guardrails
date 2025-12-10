import openai
import os
import time
from typing import Dict, Any, Optional, List
import json
import tempfile
from pathlib import Path


class OpenAIClient:
    """OpenAI API client for review generation with Batch API support"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 1.0):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        # Pricing per 1K tokens
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        }
        
        # Batch API offers 50% discount
        self.batch_discount = 0.5
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API (synchronous)"""
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens
            )
            
            generation_time = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model,
                "time": generation_time,
                "cost": cost,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": response.usage.total_tokens
                },
                "success": True
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
    
    def batch_generate(
        self, 
        requests: List[Dict[str, str]], 
        use_batch_api: bool = False,
        description: str = "Review generation batch"
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reviews
        
        Args:
            requests: List of dicts with 'system_prompt' and 'user_prompt'
            use_batch_api: If True, use async Batch API (50% cheaper, slower)
            description: Description for the batch job
        
        Returns:
            List of generation results
        """
        if not use_batch_api:
            # Synchronous generation (old behavior)
            results = []
            for req in requests:
                result = self.generate(
                    system_prompt=req["system_prompt"],
                    user_prompt=req["user_prompt"],
                    temperature=req.get("temperature"),
                    max_tokens=req.get("max_tokens", 1024)
                )
                results.append(result)
            return results
        else:
            # Use Batch API
            return self._batch_api_generate(requests, description)
    
    def _batch_api_generate(
        self, 
        requests: List[Dict[str, str]], 
        description: str
    ) -> List[Dict[str, Any]]:
        """
        Use OpenAI Batch API for asynchronous generation
        
        This creates a batch job and waits for completion.
        Note: Batch jobs typically complete within 24 hours
        """
        
        try:
            # Step 1: Create JSONL file with batch requests
            batch_file_path = self._create_batch_file(requests)
            
            # Step 2: Upload the file
            print(f"Uploading batch file with {len(requests)} requests...")
            with open(batch_file_path, 'rb') as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            # Step 3: Create batch job
            print(f"Creating batch job...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description
                }
            )
            
            print(f"Batch job created: {batch_job.id}")
            print(f"Status: {batch_job.status}")
            
            # Step 4: Poll for completion
            results = self._wait_for_batch_completion(batch_job.id, requests)
            
            # Cleanup
            os.unlink(batch_file_path)
            
            return results
            
        except Exception as e:
            print(f"Batch API error: {str(e)}")
            # Fallback to synchronous generation
            print("Falling back to synchronous generation...")
            return self.batch_generate(requests, use_batch_api=False)
    
    def _create_batch_file(self, requests: List[Dict[str, str]]) -> str:
        """Create JSONL file for batch requests"""
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False
        )
        
        for i, req in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": req["system_prompt"]},
                        {"role": "user", "content": req["user_prompt"]}
                    ],
                    "temperature": req.get("temperature", self.temperature),
                    "max_tokens": req.get("max_tokens", 1024)
                }
            }
            temp_file.write(json.dumps(batch_request) + '\n')
        
        temp_file.close()
        return temp_file.name
    
    def _wait_for_batch_completion(
        self, 
        batch_id: str, 
        original_requests: List[Dict[str, str]],
        poll_interval: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Poll batch job until completion
        
        Args:
            batch_id: The batch job ID
            original_requests: Original request list (for error handling)
            poll_interval: Seconds between status checks
        """
        
        print(f"Waiting for batch completion (checking every {poll_interval}s)...")
        
        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status
            
            print(f"Status: {status} | Completed: {batch_job.request_counts.completed}/{batch_job.request_counts.total}")
            
            if status == "completed":
                print("Batch completed! Retrieving results...")
                return self._retrieve_batch_results(batch_job, original_requests)
            
            elif status == "failed":
                print(f"Batch failed: {batch_job.errors}")
                # Fallback to sync
                return self.batch_generate(original_requests, use_batch_api=False)
            
            elif status in ["expired", "cancelled"]:
                print(f"Batch {status}")
                return self.batch_generate(original_requests, use_batch_api=False)
            
            # Still processing
            time.sleep(poll_interval)
    
    def _retrieve_batch_results(
        self, 
        batch_job, 
        original_requests: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Download and parse batch results"""
        
        # Download output file
        result_file_id = batch_job.output_file_id
        file_response = self.client.files.content(result_file_id)
        
        # Parse JSONL results
        results_by_id = {}
        for line in file_response.text.strip().split('\n'):
            result = json.loads(line)
            custom_id = result['custom_id']
            results_by_id[custom_id] = result
        
        # Map back to original order
        formatted_results = []
        for i, req in enumerate(original_requests):
            custom_id = f"request-{i}"
            
            if custom_id in results_by_id:
                result = results_by_id[custom_id]
                
                if result['response']['status_code'] == 200:
                    body = result['response']['body']
                    choice = body['choices'][0]
                    usage = body['usage']
                    
                    # Calculate cost with batch discount
                    cost = self._calculate_cost(
                        usage['prompt_tokens'], 
                        usage['completion_tokens'],
                        is_batch=True
                    )
                    
                    formatted_results.append({
                        "text": choice['message']['content'],
                        "model": self.model,
                        "time": 0,  # Batch doesn't track individual times
                        "cost": cost,
                        "tokens": {
                            "input": usage['prompt_tokens'],
                            "output": usage['completion_tokens'],
                            "total": usage['total_tokens']
                        },
                        "success": True,
                        "batch_mode": True
                    })
                else:
                    # Request failed
                    formatted_results.append({
                        "text": None,
                        "model": self.model,
                        "time": 0,
                        "cost": 0.0,
                        "error": result['response']['body'].get('error', 'Unknown error'),
                        "success": False,
                        "batch_mode": True
                    })
            else:
                # Missing result
                formatted_results.append({
                    "text": None,
                    "model": self.model,
                    "time": 0,
                    "cost": 0.0,
                    "error": "Result not found in batch output",
                    "success": False,
                    "batch_mode": True
                })
        
        return formatted_results
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check status of a batch job"""
        batch_job = self.client.batches.retrieve(batch_id)
        return {
            "id": batch_job.id,
            "status": batch_job.status,
            "created_at": batch_job.created_at,
            "completed_at": batch_job.completed_at,
            "request_counts": {
                "total": batch_job.request_counts.total,
                "completed": batch_job.request_counts.completed,
                "failed": batch_job.request_counts.failed
            },
            "metadata": batch_job.metadata
        }
    
    def list_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent batch jobs"""
        batches = self.client.batches.list(limit=limit)
        return [
            {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "metadata": batch.metadata
            }
            for batch in batches.data
        ]
    
    def _calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        is_batch: bool = False
    ) -> float:
        """Calculate API cost"""
        model_pricing = self.pricing.get(
            self.model, 
            {"input": 0.01, "output": 0.03}
        )
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        total_cost = input_cost + output_cost
        
        # Apply batch discount
        if is_batch:
            total_cost *= self.batch_discount
        
        return total_cost
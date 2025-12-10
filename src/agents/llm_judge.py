"""
LLM-as-a-Judge agent for evaluating review quality.
"""
from typing import Dict, Any, List
from src.models.model_router import ModelRouter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMJudge:
    """Agent that uses LLM to judge review quality."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("quality", {}).get("llm_judge", {})
        judge_config = config.copy()
        judge_config["models"]["primary"] = self.config.get("model", "gpt-4")
        self.model_router = ModelRouter(judge_config)
        self.criteria = self.config.get("evaluation_criteria", ["realism", "relevance", "coherence"])
    
    def evaluate(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a review using LLM-as-a-Judge.
        
        Args:
            review: Review dictionary to evaluate
            
        Returns:
            Evaluation results with scores and feedback
        """
        prompt = self._build_evaluation_prompt(review)
        
        try:
            response = self.model_router.generate(prompt)
            evaluation = self._parse_evaluation(response, review)
            return evaluation
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            return {
                "passed": False,
                "error": str(e),
                "scores": {}
            }
    
    def evaluate_batch(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of reviews.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            List of evaluation results
        """
        return [self.evaluate(review) for review in reviews]
    
    def _build_evaluation_prompt(self, review: Dict[str, Any]) -> str:
        """Build the evaluation prompt for LLM judge."""
        review_text = review.get("review_text", "")
        product_name = review.get("product_name", "product")
        category = review.get("category", "general")
        
        criteria_list = ", ".join(self.criteria)
        
        prompt = f"""You are an expert evaluator of customer reviews. Evaluate the following review for a {category} product named "{product_name}".

Review:
{review_text}

Evaluate this review on the following criteria (provide scores 0-1 and brief feedback for each):
{criteria_list}

Provide your evaluation in the following JSON format:
{{
    "realism": {{"score": 0.0-1.0, "feedback": "..."}},
    "relevance": {{"score": 0.0-1.0, "feedback": "..."}},
    "coherence": {{"score": 0.0-1.0, "feedback": "..."}},
    "overall_score": 0.0-1.0,
    "passed": true/false
}}

Evaluation:"""
        return prompt
    
    def _parse_evaluation(self, response: str, review: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured evaluation format."""
        import json
        import re
        
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                evaluation = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM judge response as JSON")
            evaluation = {
                "passed": False,
                "error": "Failed to parse evaluation",
                "scores": {}
            }
        
        evaluation["review_id"] = review.get("product_id")
        return evaluation


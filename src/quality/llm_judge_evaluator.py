"""
Production-Ready LLM Judge Evaluator with IMPROVED DISCRIMINATION

Key Fixes:
- Uses config temperature (not hardcoded 0.1)
- No example scores in prompts (no anchoring)
- Clear scoring rubrics (what each score means)
- Penalty guidelines for common issues
- Forces model to think critically
"""

from typing import Dict, Any, List, Optional
import json
import time
import re
from pathlib import Path
from datetime import datetime


class LLMJudgeEvaluator:
    """
    Advanced LLM-as-a-Judge with proper discrimination
    """
    
    def __init__(
        self,
        primary_model_client,
        secondary_model_client: Optional[Any] = None,
        enable_multi_judge: bool = True,
        enable_fallback: bool = True,
        debug_mode: bool = False,
        temperature: float = 0.5  # ✅ NEW: Configurable temperature
    ):
        self.primary_judge = primary_model_client
        self.secondary_judge = secondary_model_client
        self.enable_multi_judge = enable_multi_judge
        self.enable_fallback = enable_fallback
        self.debug_mode = debug_mode
        self.temperature = temperature  # ✅ Store temperature
        
        # Create debug directory
        self.debug_dir = Path("logs/llm_judge_debug")
        if self.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Track failures for circuit breaker
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # ✅ IMPROVED PROMPTS - No example scores, clear rubrics
        self.authenticity_prompt_template = """You are evaluating a product review for authenticity.

Review Text: "{review_text}"
Stated Rating: {rating}/5 stars

Evaluate these dimensions (0-10 scale):

SCORING RUBRIC:
- 0-3: Severe issues (fake, templated, no details)
- 4-5: Multiple problems (generic, lacks specifics, inconsistent)
- 6-7: Acceptable with minor issues (somewhat generic, few details)
- 8-9: Good quality (natural, specific, balanced)
- 10: Exceptional (highly detailed, authentic, insightful)

Dimensions to score:
1. natural_language: Does it sound human and conversational (not AI/template)?
2. specific_details: Are there concrete examples, names, numbers, or experiences?
3. balanced_perspective: Does it mention both positives and negatives?
4. emotional_consistency: Does the tone/emotion match the rating?
5. writing_style: Is the writing varied and natural (not repetitive)?

PENALTIES:
- Generic phrases like "highly recommend" with no details: -2 points
- No specific examples or evidence: -3 points  
- Extreme language without justification: -2 points
- Obvious template structure: -4 points
- Contradicts rating (happy language but low rating): -3 points

Calculate authenticity_score as average of the 5 dimensions.
Decision: PASS if authenticity_score >= 6.0, else FAIL

Output ONLY this JSON (no markdown, no explanation):
{{"authenticity_score": X, "natural_language": X, "specific_details": X, "balanced_perspective": X, "emotional_consistency": X, "writing_style": X, "feedback": "Brief explanation of score", "decision": "PASS or FAIL"}}"""

        self.alignment_prompt_template = """You are checking if review sentiment matches the rating.

Review Text: "{review_text}"
Stated Rating: {rating}/5 stars

EXPECTED SENTIMENT BY RATING:
- 1 star: Very negative (disappointed, terrible, avoid)
- 2 stars: Mostly negative (issues, not satisfied, below expectations)
- 3 stars: Mixed/neutral (some good, some bad, "okay")
- 4 stars: Mostly positive (good, satisfied, minor issues)
- 5 stars: Very positive (excellent, love it, highly recommend)

SCORING (0-10):
- 0-3: Complete mismatch (e.g., "terrible" but 5 stars)
- 4-5: Significant mismatch (sentiment leans wrong way)
- 6-7: Minor mismatch or vague sentiment
- 8-9: Good alignment, clear sentiment
- 10: Perfect alignment

Evaluate:
1. Does the language tone match the rating?
2. Are positive/negative word counts appropriate?
3. Does it mention issues expected at that rating level?

Output ONLY this JSON:
{{"alignment_score": X, "sentiment_match": true/false, "detected_issues": ["issue1", "issue2"], "feedback": "Explanation", "decision": "PASS or FAIL"}}"""

        self.expertise_prompt_template = """You are evaluating domain knowledge in a review.

Review Text: "{review_text}"
Domain: {domain}
Reviewer Persona: {persona}

EXPERTISE LEVELS:
- 0-3: No domain knowledge (generic, could apply to anything)
- 4-5: Basic knowledge (mentions obvious features only)
- 6-7: Intermediate (some specific terminology, use cases)
- 8-9: Advanced (technical details, comparisons, deep understanding)
- 10: Expert (insider knowledge, edge cases, sophisticated analysis)

Evaluate:
1. technical_accuracy: Correct terminology and concepts?
2. domain_knowledge: Shows understanding of domain-specific needs?
3. use_case_relevance: Describes actual usage scenarios?
4. persona_consistency: Matches expected expertise level for persona?

PENALTIES:
- Completely generic (could be any product): -4 points
- Wrong or made-up technical terms: -3 points
- Claims expertise but shows none: -2 points
- Persona mismatch (casual user using enterprise jargon): -2 points

Average the 4 dimensions for expertise_score.

Output ONLY this JSON:
{{"expertise_score": X, "technical_accuracy": X, "domain_knowledge": X, "use_case_relevance": X, "persona_consistency": X, "detected_knowledge_level": "novice/intermediate/advanced", "feedback": "Explanation", "decision": "PASS or FAIL"}}"""

        self.uniqueness_prompt_template = """You are checking if this review is unique compared to existing ones.

New Review: "{review_text}"

Existing Reviews (sample):
{sample_reviews}

UNIQUENESS SCORING (0-10):
- 0-3: Nearly identical (same phrases, structure, content)
- 4-5: Very similar (minor variations on same themes)
- 6-7: Somewhat similar (common points but different wording)
- 8-9: Mostly unique (distinct perspective and phrasing)
- 10: Completely unique (novel insights, different angle)

Evaluate:
1. phrasing_originality: Uses different words and sentence structures?
2. content_novelty: Mentions different aspects or details?
3. structural_variation: Different organization and flow?

PENALTIES:
- Copies exact phrases from existing reviews: -5 points
- Same examples/details as another review: -3 points
- Identical structure/organization: -2 points
- Generic phrases found in many reviews: -1 point

Average the 3 dimensions for uniqueness_score.

Output ONLY this JSON:
{{"uniqueness_score": X, "phrasing_originality": X, "content_novelty": X, "structural_variation": X, "generic_phrases": ["phrase1", "phrase2"], "feedback": "Explanation", "decision": "PASS or FAIL"}}"""

    def _save_debug_info(self, evaluation_name: str, prompt: str, response: str, parsed: Dict[str, Any]):
        """Save debug information for troubleshooting"""
        if not self.debug_mode:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_file = self.debug_dir / f"{timestamp}_{evaluation_name}.json"
        
        debug_data = {
            "timestamp": timestamp,
            "evaluation": evaluation_name,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "response_preview": response[:500],
            "response_full": response,
            "parsed_result": parsed,
            "parsing_success": "error" not in parsed
        }
        
        try:
            with open(debug_file, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save debug info: {e}")
    
    def _parse_json_response(
        self, 
        text: str, 
        evaluation_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """Advanced JSON parser with 6 parsing strategies"""
        
        if not text or not isinstance(text, str):
            return self._create_error_response(
                evaluation_name,
                f"Empty or invalid response type: {type(text)}"
            )
        
        text = text.strip()
        
        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(text)
            if self._validate_json_structure(result, evaluation_name):
                return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Strip markdown code blocks
        if "```json" in text:
            try:
                json_text = text.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_text)
                if self._validate_json_structure(result, evaluation_name):
                    return result
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Strategy 3: Strip generic code blocks
        if "```" in text:
            try:
                json_text = text.split("```")[1].split("```")[0].strip()
                result = json.loads(json_text)
                if self._validate_json_structure(result, evaluation_name):
                    return result
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Strategy 4: Find JSON with regex (greedy)
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if self._validate_json_structure(result, evaluation_name):
                    return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Extract key-value pairs manually
        try:
            result = self._extract_key_value_pairs(text, evaluation_name)
            if result and self._validate_json_structure(result, evaluation_name):
                return result
        except Exception:
            pass
        
        # Strategy 6: Parse natural language response
        try:
            result = self._parse_natural_language(text, evaluation_name)
            if result and self._validate_json_structure(result, evaluation_name):
                return result
        except Exception:
            pass
        
        # All strategies failed
        print(f"❌ All JSON parsing strategies failed for {evaluation_name}")
        print(f"   Response preview: {text[:200]}...")
        
        return self._create_error_response(
            evaluation_name,
            f"Could not parse JSON from response (tried 6 methods)"
        )
    
    def _validate_json_structure(self, data: Dict[str, Any], evaluation_name: str) -> bool:
        """Validate that parsed JSON has required fields"""
        
        required_fields = {
            "authenticity": ["authenticity_score", "decision"],
            "alignment": ["alignment_score", "decision"],
            "expertise": ["expertise_score", "decision"],
            "uniqueness": ["uniqueness_score", "decision"]
        }
        
        eval_type = evaluation_name.lower()
        if eval_type not in required_fields:
            return True
        
        required = required_fields[eval_type]
        has_required = all(field in data for field in required)
        
        if not has_required:
            print(f"⚠️  Missing required fields for {evaluation_name}: {required}")
            print(f"   Found fields: {list(data.keys())}")
        
        return has_required
    
    def _extract_key_value_pairs(self, text: str, evaluation_name: str) -> Dict[str, Any]:
        """Extract key-value pairs using regex patterns"""
        
        result = {}
        
        # Extract numeric scores
        score_patterns = [
            (r'"?(\w+_score)"?\s*[:\=]\s*(\d+\.?\d*)', float),
            (r'"?score"?\s*[:\=]\s*(\d+\.?\d*)', float),
        ]
        
        for pattern, converter in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    key, value = match
                    try:
                        result[key] = converter(value)
                    except ValueError:
                        pass
        
        # Extract decision
        decision_match = re.search(r'"?decision"?\s*[:\=]\s*"?(PASS|FAIL)"?', text, re.IGNORECASE)
        if decision_match:
            result["decision"] = decision_match.group(1).upper()
        
        # Extract feedback
        feedback_match = re.search(r'"?feedback"?\s*[:\=]\s*"([^"]+)"', text, re.IGNORECASE)
        if feedback_match:
            result["feedback"] = feedback_match.group(1)
        
        # Add default score if none found
        if evaluation_name and not any(k.endswith('_score') for k in result.keys()):
            score_key = f"{evaluation_name.lower()}_score"
            result[score_key] = 7.0
        
        return result if result else None
    
    def _parse_natural_language(self, text: str, evaluation_name: str) -> Dict[str, Any]:
        """Parse natural language responses"""
        
        result = {
            "feedback": text[:200],
            "decision": "FAIL",
            "natural_language_parsed": True
        }
        
        text_lower = text.lower()
        
        pass_keywords = ["pass", "good", "acceptable", "appropriate", "authentic", "correct"]
        fail_keywords = ["fail", "poor", "inappropriate", "fake", "wrong", "mismatch"]
        
        pass_count = sum(1 for kw in pass_keywords if kw in text_lower)
        fail_count = sum(1 for kw in fail_keywords if kw in text_lower)
        
        if pass_count > fail_count:
            result["decision"] = "PASS"
            score = 7.0
        else:
            result["decision"] = "FAIL"
            score = 4.0
        
        score_key = f"{evaluation_name.lower()}_score"
        result[score_key] = score
        
        return result
    
    def _create_error_response(self, evaluation_name: str, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        
        score_key = f"{evaluation_name.lower()}_score"
        
        return {
            "error": error_msg,
            score_key: 0,
            "decision": "FAIL",
            "feedback": f"Evaluation failed: {error_msg}",
            "parsing_failed": True
        }
    
    def evaluate_authenticity(
        self,
        review_text: str,
        rating: int,
        judge: str = "primary"
    ) -> Dict[str, Any]:
        """Evaluate review authenticity"""
        
        prompt = self.authenticity_prompt_template.format(
            review_text=review_text[:1000],
            rating=rating
        )
        
        client = self.primary_judge if judge == "primary" else self.secondary_judge
        if not client:
            if self.enable_fallback:
                return self._fallback_authenticity_score(review_text, rating)
            return self._create_error_response("authenticity", "No judge client available")
        
        # ✅ USE CONFIGURED TEMPERATURE (not hardcoded 0.1)
        result = client.generate(
            system_prompt="You are a critical review evaluator. Be strict but fair. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=self.temperature  # ✅ Use class temperature!
        )
        
        if not result["success"]:
            if self.enable_fallback:
                return self._fallback_authenticity_score(review_text, rating)
            return self._create_error_response("authenticity", result.get('error', 'Generation failed'))
        
        evaluation = self._parse_json_response(result["text"], "authenticity")
        
        if "error" in evaluation or "parsing_failed" in evaluation:
            self.consecutive_failures += 1
            if self.enable_fallback:
                return self._fallback_authenticity_score(review_text, rating)
            return evaluation
        
        self.consecutive_failures = 0
        evaluation["judge_model"] = getattr(client, 'model_name', getattr(client, 'model', 'unknown'))
        evaluation["evaluation_time"] = result["time"]
        evaluation["llm_success"] = True
        
        if self.debug_mode:
            self._save_debug_info("authenticity", prompt, result["text"], evaluation)
        
        return evaluation
    
    def evaluate_alignment(
        self,
        review_text: str,
        rating: int,
        judge: str = "primary"
    ) -> Dict[str, Any]:
        """Evaluate rating-content alignment"""
        
        prompt = self.alignment_prompt_template.format(
            review_text=review_text[:1000],
            rating=rating
        )
        
        client = self.primary_judge if judge == "primary" else self.secondary_judge
        if not client:
            if self.enable_fallback:
                return self._fallback_alignment_score(review_text, rating)
            return self._create_error_response("alignment", "No judge client available")
        
        result = client.generate(
            system_prompt="You are a critical review evaluator. Be strict but fair. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=self.temperature  # ✅ Use configured temperature
        )
        
        if not result["success"]:
            if self.enable_fallback:
                return self._fallback_alignment_score(review_text, rating)
            return self._create_error_response("alignment", result.get('error', 'Generation failed'))
        
        evaluation = self._parse_json_response(result["text"], "alignment")
        
        if "error" in evaluation or "parsing_failed" in evaluation:
            if self.enable_fallback:
                return self._fallback_alignment_score(review_text, rating)
            return evaluation
        
        evaluation["judge_model"] = getattr(client, 'model_name', getattr(client, 'model', 'unknown'))
        evaluation["evaluation_time"] = result["time"]
        evaluation["llm_success"] = True
        
        if self.debug_mode:
            self._save_debug_info("alignment", prompt, result["text"], evaluation)
        
        return evaluation
    
    def evaluate_expertise(
        self,
        review_text: str,
        domain: str,
        persona: str,
        judge: str = "primary"
    ) -> Dict[str, Any]:
        """Evaluate domain expertise"""
        
        prompt = self.expertise_prompt_template.format(
            review_text=review_text[:1000],
            domain=domain,
            persona=persona
        )
        
        client = self.primary_judge if judge == "primary" else self.secondary_judge
        if not client:
            if self.enable_fallback:
                return self._fallback_expertise_score(review_text, domain, persona)
            return self._create_error_response("expertise", "No judge client available")
        
        result = client.generate(
            system_prompt="You are a critical review evaluator. Be strict but fair. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=self.temperature  # ✅ Use configured temperature
        )
        
        if not result["success"]:
            if self.enable_fallback:
                return self._fallback_expertise_score(review_text, domain, persona)
            return self._create_error_response("expertise", result.get('error', 'Generation failed'))
        
        evaluation = self._parse_json_response(result["text"], "expertise")
        
        if "error" in evaluation or "parsing_failed" in evaluation:
            if self.enable_fallback:
                return self._fallback_expertise_score(review_text, domain, persona)
            return evaluation
        
        evaluation["judge_model"] = getattr(client, 'model_name', getattr(client, 'model', 'unknown'))
        evaluation["evaluation_time"] = result["time"]
        evaluation["llm_success"] = True
        
        if self.debug_mode:
            self._save_debug_info("expertise", prompt, result["text"], evaluation)
        
        return evaluation
    
    def evaluate_uniqueness(
        self,
        review_text: str,
        existing_reviews: List[str],
        judge: str = "primary"
    ) -> Dict[str, Any]:
        """Evaluate review uniqueness"""
        
        sample_reviews = "\n\n".join([
            f"Review {i+1}: {review[:150]}..."
            for i, review in enumerate(existing_reviews[:3])
        ]) if existing_reviews else "No existing reviews yet."
        
        prompt = self.uniqueness_prompt_template.format(
            review_text=review_text[:1000],
            sample_reviews=sample_reviews
        )
        
        client = self.primary_judge if judge == "primary" else self.secondary_judge
        if not client:
            if self.enable_fallback:
                return self._fallback_uniqueness_score(review_text, existing_reviews)
            return self._create_error_response("uniqueness", "No judge client available")
        
        result = client.generate(
            system_prompt="You are a critical review evaluator. Be strict but fair. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=self.temperature  # ✅ Use configured temperature
        )
        
        if not result["success"]:
            if self.enable_fallback:
                return self._fallback_uniqueness_score(review_text, existing_reviews)
            return self._create_error_response("uniqueness", result.get('error', 'Generation failed'))
        
        evaluation = self._parse_json_response(result["text"], "uniqueness")
        
        if "error" in evaluation or "parsing_failed" in evaluation:
            if self.enable_fallback:
                return self._fallback_uniqueness_score(review_text, existing_reviews)
            return evaluation
        
        evaluation["judge_model"] = getattr(client, 'model_name', getattr(client, 'model', 'unknown'))
        evaluation["evaluation_time"] = result["time"]
        evaluation["llm_success"] = True
        
        if self.debug_mode:
            self._save_debug_info("uniqueness", prompt, result["text"], evaluation)
        
        return evaluation
    
    # ============================================================================
    # FALLBACK HEURISTIC SCORING METHODS (keep existing ones)
    # ============================================================================
    
    def _fallback_authenticity_score(self, review_text: str, rating: int) -> Dict[str, Any]:
        """Heuristic authenticity scoring when LLM fails"""
        
        word_count = len(review_text.split())
        sentence_count = len([s for s in review_text.split('.') if len(s.strip()) > 10])
        
        specificity_indicators = ['we', 'our', 'i', 'my', 'team', 'company', 'project']
        has_specifics = sum(1 for word in specificity_indicators if word in review_text.lower())
        has_balance = any(word in review_text.lower() for word in ['but', 'however', 'although', 'though'])
        
        natural_language = min(9, 5 + (word_count / 25))
        specific_details = min(9, 4 + has_specifics)
        balanced_perspective = 8 if has_balance else 6
        emotional_consistency = 7
        writing_style = min(9, 5 + (sentence_count / 2))
        
        authenticity_score = (natural_language + specific_details + balanced_perspective + emotional_consistency + writing_style) / 5
        decision = "PASS" if authenticity_score >= 6.0 else "FAIL"
        
        print(f"🔧 Fallback authenticity: {authenticity_score:.1f}/10 -> {decision}")
        
        return {
            "authenticity_score": authenticity_score,
            "natural_language": natural_language,
            "specific_details": specific_details,
            "balanced_perspective": balanced_perspective,
            "emotional_consistency": emotional_consistency,
            "writing_style": writing_style,
            "feedback": f"Heuristic scoring: {authenticity_score:.1f}/10 (LLM unavailable)",
            "decision": decision,
            "judge_model": "fallback-heuristic",
            "evaluation_time": 0.0,
            "fallback": True,
            "llm_success": False
        }
    
    def _fallback_alignment_score(self, review_text: str, rating: int) -> Dict[str, Any]:
        """Heuristic alignment scoring"""
        
        text_lower = review_text.lower()
        positive_words = ['good', 'great', 'excellent', 'helpful', 'easy', 'love', 'recommend']
        negative_words = ['bad', 'poor', 'difficult', 'frustrating', 'disappointed', 'avoid']
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        expected_positive = rating >= 4
        expected_negative = rating <= 2
        expected_neutral = rating == 3
        
        if expected_positive and pos_count > neg_count:
            alignment_score = min(9, 6 + pos_count)
            sentiment_match = True
        elif expected_negative and neg_count > pos_count:
            alignment_score = min(9, 6 + neg_count)
            sentiment_match = True
        elif expected_neutral:
            alignment_score = 7
            sentiment_match = True
        else:
            alignment_score = 5
            sentiment_match = False
        
        decision = "PASS" if alignment_score >= 6 else "FAIL"
        detected_issues = [] if sentiment_match else ["Sentiment-rating mismatch"]
        
        print(f"🔧 Fallback alignment: {alignment_score}/10 -> {decision}")
        
        return {
            "alignment_score": alignment_score,
            "sentiment_match": sentiment_match,
            "detected_issues": detected_issues,
            "feedback": f"Heuristic alignment: {alignment_score}/10 (LLM unavailable)",
            "decision": decision,
            "judge_model": "fallback-heuristic",
            "evaluation_time": 0.0,
            "fallback": True,
            "llm_success": False
        }
    
    def _fallback_expertise_score(self, review_text: str, domain: str, persona: str) -> Dict[str, Any]:
        """Heuristic expertise scoring"""
        
        text_lower = review_text.lower()
        technical_terms = ['api', 'integration', 'deployment', 'configuration', 'workflow']
        tech_count = sum(1 for term in technical_terms if term in text_lower)
        
        expertise_score = min(9, 5 + tech_count * 0.5)
        decision = "PASS" if expertise_score >= 5.5 else "FAIL"
        
        print(f"🔧 Fallback expertise: {expertise_score:.1f}/10 -> {decision}")
        
        return {
            "expertise_score": expertise_score,
            "technical_accuracy": min(9, 5 + tech_count * 0.4),
            "domain_knowledge": min(9, 5 + tech_count * 0.3),
            "use_case_relevance": 7,
            "persona_consistency": 7,
            "detected_knowledge_level": "intermediate",
            "feedback": f"Heuristic expertise: {expertise_score:.1f}/10 (LLM unavailable)",
            "decision": decision,
            "judge_model": "fallback-heuristic",
            "evaluation_time": 0.0,
            "fallback": True,
            "llm_success": False
        }
    
    def _fallback_uniqueness_score(self, review_text: str, existing_reviews: List[str]) -> Dict[str, Any]:
        """Heuristic uniqueness scoring"""
        
        if not existing_reviews:
            return {
                "uniqueness_score": 9,
                "phrasing_originality": 9,
                "content_novelty": 9,
                "structural_variation": 9,
                "generic_phrases": [],
                "feedback": "First review - automatically unique",
                "decision": "PASS",
                "judge_model": "fallback-heuristic",
                "evaluation_time": 0.0,
                "fallback": True,
                "llm_success": False
            }
        
        text_words = set(review_text.lower().split())
        overlaps = []
        for existing in existing_reviews[-5:]:
            existing_words = set(existing.lower().split())
            if text_words and existing_words:
                overlap = len(text_words.intersection(existing_words)) / len(text_words)
                overlaps.append(overlap)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        uniqueness_score = max(5, 10 - (avg_overlap * 10))
        decision = "PASS" if uniqueness_score >= 6 else "FAIL"
        
        print(f"🔧 Fallback uniqueness: {uniqueness_score:.1f}/10 -> {decision}")
        
        return {
            "uniqueness_score": uniqueness_score,
            "phrasing_originality": uniqueness_score,
            "content_novelty": uniqueness_score,
            "structural_variation": uniqueness_score,
            "generic_phrases": [],
            "feedback": f"Heuristic uniqueness: {uniqueness_score:.1f}/10 (LLM unavailable)",
            "decision": decision,
            "judge_model": "fallback-heuristic",
            "evaluation_time": 0.0,
            "fallback": True,
            "llm_success": False
        }
    
    # ============================================================================
    # COMPREHENSIVE EVALUATION
    # ============================================================================
    
    def comprehensive_evaluation(
        self,
        review_text: str,
        rating: int,
        domain: str,
        persona: str,
        existing_reviews: List[str]
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation with all criteria"""
        
        start_time = time.time()
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            print(f"⚠️  Circuit breaker: Too many consecutive failures ({self.consecutive_failures})")
            
            primary_results = {
                "authenticity": self._fallback_authenticity_score(review_text, rating),
                "alignment": self._fallback_alignment_score(review_text, rating),
                "expertise": self._fallback_expertise_score(review_text, domain, persona),
                "uniqueness": self._fallback_uniqueness_score(review_text, existing_reviews)
            }
            
            self.consecutive_failures = 0
        else:
            print(f"🔍 Running LLM Judge evaluation (attempt {self.consecutive_failures + 1})")
            
            primary_results = {
                "authenticity": self.evaluate_authenticity(review_text, rating, "primary"),
                "alignment": self.evaluate_alignment(review_text, rating, "primary"),
                "expertise": self.evaluate_expertise(review_text, domain, persona, "primary"),
                "uniqueness": self.evaluate_uniqueness(review_text, existing_reviews, "primary")
            }
        
        for criterion, result in primary_results.items():
            score = result.get(f"{criterion}_score", "N/A")
            decision = result.get("decision", "UNKNOWN")
            is_fallback = result.get("fallback", False)
            method = "Fallback" if is_fallback else "LLM"
            print(f"  • {criterion}: {method} score={score}, decision={decision}")
        
        secondary_results = {}
        if self.enable_multi_judge and self.secondary_judge and self.consecutive_failures == 0:
            print(f"🔍 Running secondary judge evaluation")
            secondary_results = {
                "authenticity": self.evaluate_authenticity(review_text, rating, "secondary"),
                "alignment": self.evaluate_alignment(review_text, rating, "secondary")
            }
        
        valid_scores = []
        for criterion, result in primary_results.items():
            score_key = f"{criterion}_score"
            if score_key in result and result[score_key] is not None:
                try:
                    normalized = float(result[score_key]) / 10.0
                    valid_scores.append(normalized)
                except (ValueError, TypeError):
                    print(f"⚠️  Invalid score for {criterion}: {result[score_key]}")
        
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
        
        dimension_scores = {}
        for criterion, result in primary_results.items():
            score_key = f"{criterion}_score"
            if score_key in result and result[score_key] is not None:
                try:
                    dimension_scores[criterion] = float(result[score_key]) / 10.0
                except (ValueError, TypeError):
                    dimension_scores[criterion] = 0.5
        
        valid_results = [r for r in primary_results.values() if "error" not in r]
        all_passed = all(r.get("decision") == "PASS" for r in valid_results)
        
        error_count = sum(1 for r in primary_results.values() if "error" in r)
        fallback_count = sum(1 for r in primary_results.values() if r.get("fallback", False))
        llm_success_count = sum(1 for r in primary_results.values() if r.get("llm_success", False))
        
        final_decision = "PASS" if all_passed and overall_score >= 0.6 else "FAIL"
        
        consensus = self._check_consensus(primary_results, secondary_results)
        feedback = self._aggregate_feedback(primary_results)
        
        total_time = time.time() - start_time
        
        print(f"📊 Overall: score={overall_score:.2f}, decision={final_decision}, "
              f"llm_success={llm_success_count}/4, fallback={fallback_count}/4, errors={error_count}/4")
        
        return {
            "overall_score": float(overall_score),
            "dimension_scores": dimension_scores,
            "individual_scores": primary_results,
            "secondary_scores": secondary_results,
            "judge_consensus": consensus,
            "final_decision": final_decision,
            "feedback": feedback,
            "evaluation_time": total_time,
            "primary_judge": "gemini-2.0-flash-exp",
            "errors_count": error_count,
            "fallback_count": fallback_count,
            "llm_success_count": llm_success_count,
            "consecutive_failures": self.consecutive_failures
        }
    
    def _check_consensus(self, primary_results: Dict[str, Any], secondary_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check agreement between judges"""
        
        if not secondary_results:
            return {"enabled": False}
        
        agreements = []
        for criterion in secondary_results.keys():
            if criterion in primary_results:
                primary_decision = primary_results[criterion].get("decision")
                secondary_decision = secondary_results[criterion].get("decision")
                
                if primary_decision and secondary_decision:
                    agreements.append(primary_decision == secondary_decision)
        
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
        
        return {
            "enabled": True,
            "agreement_rate": float(agreement_rate),
            "agreed_count": sum(agreements),
            "total_comparisons": len(agreements)
        }
    
    def _aggregate_feedback(self, results: Dict[str, Any]) -> str:
        """Aggregate feedback from all criteria"""
        
        feedbacks = []
        for criterion, result in results.items():
            decision = result.get("decision", "UNKNOWN")
            feedback = result.get("feedback", "No feedback")
            is_fallback = result.get("fallback", False)
            
            prefix = f"{criterion.capitalize()} [{decision}]"
            if is_fallback:
                prefix += " [FALLBACK]"
            
            feedbacks.append(f"{prefix}: {feedback}")
        
        return " | ".join(feedbacks) if feedbacks else "All evaluations completed"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "consecutive_failures": self.consecutive_failures,
            "max_consecutive_failures": self.max_consecutive_failures,
            "circuit_breaker_active": self.consecutive_failures >= self.max_consecutive_failures,
            "fallback_enabled": self.enable_fallback,
            "debug_mode": self.debug_mode,
            "temperature": self.temperature
        }
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker"""
        print(f"🔄 Resetting circuit breaker (was at {self.consecutive_failures} failures)")
        self.consecutive_failures = 0
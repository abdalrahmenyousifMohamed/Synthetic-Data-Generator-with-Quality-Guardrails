from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
import time
from src.agents.generator import ReviewGeneratorAgent
from src.quality.diversity_metrics import DiversityMetrics
from src.quality.bias_detection import BiasDetector
from src.quality.realism_validator import RealismValidator
from src.quality.llm_judge_evaluator import LLMJudgeEvaluator
from src.utils.logger import PipelineLogger
from src.utils.metrics_tracker import MetricsTracker
from src.utils.config_loader import Config
import random


class ReviewGenerationState(TypedDict):
    """State for review generation workflow"""
    review_id: str
    review_text: str
    rating: int
    persona: Dict[str, Any]
    domain: str
    model_used: str
    seed_words: List[str]
    generation_time: float
    cost: float
    
    # Quality metrics
    statistical_metrics: Dict[str, Any]
    llm_judge_results: Dict[str, Any]
    
    # Decision tracking
    regeneration_count: int
    final_decision: str
    rejection_reasons: List[str]
    
    # For context
    existing_reviews: List[str]



class ReviewGenerationOrchestrator:
    """
    Main orchestrator using LangGraph to coordinate all agents
    LLM Judge is now OPTIONAL - can run with statistical checks only
    """
    
    def __init__(
        self,
        config: Config,
        generator_agent: ReviewGeneratorAgent,
        diversity_metrics: DiversityMetrics,
        bias_detector: BiasDetector,
        realism_validator: RealismValidator,
        
        logger: PipelineLogger,
        metrics_tracker: MetricsTracker,
        llm_judge: LLMJudgeEvaluator = None,  # MAKE OPTIONAL
    ):
        self.config = config
        self.generator = generator_agent
        self.diversity_metrics = diversity_metrics
        self.bias_detector = bias_detector
        self.realism_validator = realism_validator
        self.llm_judge = llm_judge  # Can be None
        self.logger = logger
        self.metrics_tracker = metrics_tracker
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        # Storage for generated reviews
        self.accepted_reviews = []
        self.all_reviews_text = []
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow - with optional LLM Judge"""
        
        workflow = StateGraph(ReviewGenerationState)
        
        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("check_statistical", self._check_statistical_node)
        workflow.add_node("decide", self._decide_node)
        
        # Conditionally add LLM Judge node
        if self.llm_judge is not None:
            workflow.add_node("check_llm_judge", self._check_llm_judge_node)
            workflow.add_edge("generate", "check_statistical")
            workflow.add_edge("check_statistical", "check_llm_judge")
            workflow.add_edge("check_llm_judge", "decide")
            self.logger.info("Building workflow WITH LLM Judge")
        else:
            # Simplified workflow without LLM Judge
            workflow.add_edge("generate", "check_statistical")
            workflow.add_edge("check_statistical", "decide")
            self.logger.info("Building workflow WITHOUT LLM Judge (statistical checks only)")
        
        # Conditional edge for regeneration
        workflow.add_conditional_edges(
            "decide",
            self._should_regenerate,
            {
                "regenerate": "generate",
                "accept": END,
                "reject": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        return workflow.compile()
    
    def _generate_node(self, state: ReviewGenerationState) -> ReviewGenerationState:
        """Generation node with feedback integration"""
        
        self.logger.info(f"Generating review {state['review_id']} (attempt {state['regeneration_count'] + 1})")
        
        # Extract feedback from previous results
        previous_feedback = None
        if state['regeneration_count'] > 0:
            # For statistical-only mode, use statistical feedback
            if self.llm_judge is None:
                if state.get('rejection_reasons'):
                    previous_feedback = " | ".join(state['rejection_reasons'])
            else:
                # Use LLM judge feedback if available
                llm_results = state.get('llm_judge_results', {})
                if llm_results and llm_results.get('feedback'):
                    previous_feedback = llm_results['feedback']
        
        # Generate with feedback
        result = self.generator.generate_review(
            persona=state['persona'],
            rating=state['rating'],
            domain=state['domain'],
            existing_reviews=state.get('existing_reviews', []),
            previous_feedback=previous_feedback,
            regeneration_count=state['regeneration_count']
        )
        
        if not result["success"]:
            state['rejection_reasons'].append(f"Generation failed: {result.get('error')}")
            state['final_decision'] = 'reject'
            return state
        
        state['review_text'] = result['review_text']
        state['model_used'] = result['model_used']
        state['generation_time'] = result['generation_time']
        state['cost'] = result['cost']
        state['seed_words'] = result['metadata']['seed_words']
        
        # Log generation with attempt info
        self.logger.log_generation(state['review_id'], {
            "model": state['model_used'],
            "rating": state['rating'],
            "persona": state['persona']['name'],
            "time": state['generation_time'],
            "cost": state['cost'],
            "attempt": state['regeneration_count'] + 1,
            "had_feedback": previous_feedback is not None
        })
        
        return state
    
    
    def _check_statistical_node(self, state: ReviewGenerationState) -> ReviewGenerationState:
        """Statistical quality checks with adaptive thresholds"""
        
        review_text = state['review_text']
        rating = state['rating']
        attempt = state['regeneration_count'] + 1  # Current attempt number
        
        # Length check
        word_count = len(review_text.split())
        unique_words = len(set(review_text.lower().split()))
        
        # CRITICAL: Pass attempt number to bias detector
        bias_result = self.bias_detector.detect_unrealistic_patterns(
            review_text, 
            rating,
            attempt_number=attempt  # Enables adaptive thresholds
        )
        sentiment_result = bias_result['sentiment_mismatch']
        
        # Self-BLEU check
        if len(self.all_reviews_text) > 0:
            test_reviews = self.all_reviews_text + [review_text]
            self_bleu_diversity = self.diversity_metrics.calculate_self_bleu(test_reviews)
        else:
            self_bleu_diversity = 1.0
        
        # Compile metrics
        statistical_metrics = {
            "length": word_count,
            "unique_words": unique_words,
            "self_bleu_diversity": self_bleu_diversity,
            "sentiment_alignment": sentiment_result.get('alignment_score', 0.0),
            "bias_check": bias_result,
            "passed_statistical": True,
            "attempt_number": attempt,
            "adaptive_threshold": bias_result.get('threshold_used', 0.30)
        }
        
        # Check thresholds
        thresholds = self.config.quality_thresholds
        rejection_reasons = []
        
        if word_count < thresholds.min_length:
            rejection_reasons.append(f"Too short: {word_count} words (min: {thresholds.min_length})")
        
        if word_count > thresholds.max_length:
            rejection_reasons.append(f"Too long: {word_count} words (max: {thresholds.max_length})")
        
        if unique_words < thresholds.min_unique_words:
            rejection_reasons.append(f"Low vocabulary: {unique_words} unique words (min: {thresholds.min_unique_words})")
        
        if self_bleu_diversity < (1.0 - thresholds.max_self_bleu):
            rejection_reasons.append(f"Too similar to existing reviews (diversity: {self_bleu_diversity:.2f})")
        
        # CRITICAL: Use adaptive sentiment threshold from bias detector
        adaptive_threshold = sentiment_result.get('threshold_used', thresholds.min_sentiment_alignment)
        
        if sentiment_result.get('alignment_score', 0) < adaptive_threshold:
            # Only reject if circuit breaker isn't active
            if not bias_result.get('circuit_breaker_active', False):
                rejection_reasons.append(
                    f"Sentiment-rating mismatch (alignment: {sentiment_result.get('alignment_score', 0):.2f}, "
                    f"threshold: {adaptive_threshold:.2f})"
                )
            else:
                logger.warning(
                    f"Circuit breaker active for {rating}-star reviews - "
                    f"allowing marginal sentiment alignment"
                )
        
        # Check unrealistic patterns (with adaptive threshold)
        if bias_result.get('is_unrealistic', False):
            if not bias_result.get('circuit_breaker_active', False):
                rejection_reasons.append(
                    f"Unrealistic patterns (score: {bias_result['unrealistic_score']:.2f})"
                )
        
        statistical_metrics['passed_statistical'] = len(rejection_reasons) == 0
        state['statistical_metrics'] = statistical_metrics
        
        if rejection_reasons:
            state['rejection_reasons'] = rejection_reasons
        
        # Log quality check
        self.logger.log_quality_check(state['review_id'], "statistical", statistical_metrics)
        
        return state
    
    def _check_llm_judge_node(self, state: ReviewGenerationState) -> ReviewGenerationState:
        """LLM Judge evaluation node with circuit breaker"""
        
        # Skip if statistical checks already failed
        if not state['statistical_metrics'].get('passed_statistical', False):
            state['llm_judge_results'] = {
                "skipped": True, 
                "reason": "Failed statistical checks",
                "overall_score": None
            }
            return state
        
        self.logger.info(f"Running LLM Judge evaluation for {state['review_id']}")
        
        # Run comprehensive LLM judge evaluation
        llm_results = self.llm_judge.comprehensive_evaluation(
            review_text=state['review_text'],
            rating=state['rating'],
            domain=state['domain'],
            persona=state['persona']['name'],
            existing_reviews=self.all_reviews_text[-10:] if self.all_reviews_text else []
        )
        
        state['llm_judge_results'] = llm_results
        
        # CIRCUIT BREAKER: If too many parsing errors, disable LLM judge temporarily
        error_count = llm_results.get('errors_count', 0)
        if error_count >= 3:
            self.logger.error(
                f"⚠️  LLM Judge having serious issues ({error_count} errors). "
                f"Check your Gemini API or model configuration!"
            )
            # Still continue but flag the issue
            state['rejection_reasons'].append(
                f"LLM Judge evaluation failed ({error_count} errors) - possible API issue"
            )
            return state
        
        # Initialize rejection reasons for LLM judge checks
        llm_rejection_reasons = []
        
        # Check if passed LLM judge
        if llm_results['final_decision'] != 'PASS':
            # IMPROVED: Include actual feedback from dimensions
            detailed_feedback = []
            
            # Get dimension-specific feedback
            for dim, score in llm_results.get('dimension_scores', {}).items():
                if score < 0.6:  # Low score
                    dim_result = llm_results['individual_scores'].get(dim, {})
                    dim_feedback = dim_result.get('feedback', 'Score too low')
                    detailed_feedback.append(f"{dim}({score:.1f}): {dim_feedback}")
            
            if detailed_feedback:
                llm_rejection_reasons.append(
                    f"Failed LLM Judge: {' | '.join(detailed_feedback)}"
                )
            else:
                # Fallback to overall feedback
                llm_rejection_reasons.append(
                    f"Failed LLM Judge: {llm_results.get('feedback', 'Overall score too low')}"
                )
        
        # Check overall score threshold
        if llm_results['overall_score'] < self.config.quality_thresholds.min_llm_judge_score:
            if llm_results['final_decision'] != 'PASS':
                llm_rejection_reasons.append(
                    f"LLM Judge score too low: {llm_results['overall_score']:.2f} "
                    f"(min: {self.config.quality_thresholds.min_llm_judge_score})"
                )
        
        # Only update rejection reasons if LLM judge failed
        if llm_rejection_reasons:
            state['rejection_reasons'] = llm_rejection_reasons

        # Log LLM judge evaluation
        self.logger.log_llm_judge(state['review_id'], llm_results)
        
        # Track judge metrics
        self.metrics_tracker.record_llm_judge(
            passed=llm_results['final_decision'] == 'PASS',
            judges_agreed=llm_results['judge_consensus'].get('agreement_rate', 1.0) > 0.8
        )
        
        return state

    
    
    def _extract_issue_type(self, rejection_reason: str) -> str:
            """
            Extract issue type from rejection reason for loop detection.
            Used to identify if reviews are repeatedly failing for the same reason.
            """
            reason_lower = rejection_reason.lower()
            
            # Check for specific patterns
            if 'sentiment' in reason_lower or 'mismatch' in reason_lower:
                return 'sentiment'
            elif 'unrealistic' in reason_lower:
                return 'unrealistic'
            elif 'short' in reason_lower or 'long' in reason_lower:
                return 'length'
            elif 'vocabulary' in reason_lower or 'unique' in reason_lower or 'unique words' in reason_lower:
                return 'vocabulary'
            elif 'similar' in reason_lower or 'diversity' in reason_lower:
                return 'diversity'
            elif 'llm judge' in reason_lower or 'evaluation' in reason_lower or 'api' in reason_lower:
                return 'llm_judge_error'
            elif 'expertise' in reason_lower or 'technical' in reason_lower:
                return 'expertise'
            elif 'authenticity' in reason_lower or 'authentic' in reason_lower:
                return 'authenticity'
            else:
                return 'other'
    
    def _get_llm_judge_score(self, state: ReviewGenerationState) -> float:
        """
        Safely extract LLM judge score from state.
        Returns None if judge was skipped or had errors.
        """
        llm_results = state.get('llm_judge_results', {})
        
        # Check if skipped
        if llm_results.get('skipped', False):
            return None
        
        # Check if had errors
        if llm_results.get('errors_count', 0) > 0:
            self.logger.debug(
                f"LLM Judge had {llm_results['errors_count']} errors - not recording score"
            )
            return None
        
        # Get the actual score
        overall_score = llm_results.get('overall_score')
        
        # Validate it's a real number
        if overall_score is not None:
            try:
                return float(overall_score)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid LLM judge score: {overall_score} (type: {type(overall_score)})"
                )
                return None
        
        return None
    
    def _decide_node(self, state: ReviewGenerationState) -> ReviewGenerationState:
        """
        Enhanced decision node with:
        - LLM Judge error detection and circuit breaker
        - Loop prevention for repeated issues
        - Adaptive max attempts based on rating difficulty
        - Proper None handling for skipped evaluations
        """
        
        # Ensure llm_judge_results exists even if skipped
        if 'llm_judge_results' not in state or not state['llm_judge_results']:
            state['llm_judge_results'] = {
                "skipped": True,
                "reason": "LLM Judge disabled or not configured",
                "final_decision": "PASS",
                "overall_score": None,  # ✅ Use None instead of 1.0
                "dimension_scores": {},
                "feedback": "LLM Judge not used"
            }
        
        # CIRCUIT BREAKER: Check for repeated LLM Judge parsing errors
        if state['regeneration_count'] >= 2:  # After 2 attempts
            if 'rejection_history' not in state:
                state['rejection_history'] = []
            
            # Track if this is an LLM Judge error
            current_rejections = state.get('rejection_reasons', [])
            is_llm_error = any(
                'evaluation failed' in r.lower() or 
                'api issue' in r.lower() or 
                'parsing failed' in r.lower()
                for r in current_rejections
            )
            
            if is_llm_error:
                # Count consecutive LLM Judge errors
                state['rejection_history'].append('llm_judge_error')
                recent_history = state['rejection_history'][-3:]
                
                # If 2+ of last 3 attempts were LLM Judge errors
                if recent_history.count('llm_judge_error') >= 2:
                    self.logger.error(
                        f"🚨 CIRCUIT BREAKER ACTIVATED for {state['review_id']}: "
                        f"LLM Judge is consistently failing. This indicates a configuration "
                        f"or API issue, not a review quality problem. Forcing acceptance."
                    )
                    
                    # Force accept to prevent infinite loop
                    state['final_decision'] = 'accept'
                    state['rejection_reasons'] = []
                    
                    # Record with special flag
                    self.metrics_tracker.record_generation(
                        model_name=state['model_used'],
                        accepted=True,
                        time_taken=state['generation_time'],
                        cost=state['cost'],
                        quality_score=state['statistical_metrics'].get('self_bleu_diversity', 0),
                        llm_judge_score=None,  # ✅ Mark as unavailable
                        statistical_metrics=state['statistical_metrics']
                    )
                    
                    self.logger.warning(
                        f"⚠️  Review {state['review_id']} accepted despite LLM Judge errors. "
                        f"ACTION REQUIRED: Check your Gemini API configuration and model settings!"
                    )
                    
                    return state
        
        # Check if any rejection reasons exist
        if len(state['rejection_reasons']) > 0:
            
            # LOOP PREVENTION: Check for repeated failures on same issue
            if state['regeneration_count'] >= 2:  # On 3rd attempt
                
                if 'rejection_history' not in state:
                    state['rejection_history'] = []
                
                # Extract issue type from current rejection
                current_issue_type = self._extract_issue_type(state['rejection_reasons'][0])
                state['rejection_history'].append(current_issue_type)
                
                # Check if stuck on same issue for 3+ attempts
                if len(state['rejection_history']) >= 3:
                    last_three = state['rejection_history'][-3:]
                    
                    # If all same type, make final decision
                    if len(set(last_three)) == 1:  # All identical
                        stuck_issue = last_three[0]
                        
                        self.logger.warning(
                            f"⚠️  Review {state['review_id']} stuck on '{stuck_issue}' "
                            f"for 3+ attempts. Forcing decision to prevent infinite loop."
                        )
                        
                        # Decide based on issue severity
                        if stuck_issue in ['sentiment', 'unrealistic']:
                            # For sentiment/unrealistic: accept if borderline
                            alignment = state['statistical_metrics'].get('sentiment_alignment', 0)
                            
                            if alignment > 0.25:  # Borderline acceptable
                                state['final_decision'] = 'accept'
                                state['rejection_reasons'] = []  # Clear rejections
                                
                                self.logger.info(
                                    f"✅ Force accepting {state['review_id']} - "
                                    f"alignment {alignment:.2f} is borderline acceptable after 3 attempts"
                                )
                                
                                # Record with note about borderline acceptance
                                self.metrics_tracker.record_generation(
                                    model_name=state['model_used'],
                                    accepted=True,
                                    time_taken=state['generation_time'],
                                    cost=state['cost'],
                                    quality_score=state['statistical_metrics'].get('self_bleu_diversity', 0),
                                    llm_judge_score=self._get_llm_judge_score(state),
                                    statistical_metrics=state['statistical_metrics']
                                )
                                
                                return state
                            else:
                                # Too far from threshold, reject
                                state['final_decision'] = 'reject'
                                state['rejection_reasons'].append(
                                    f"Forced rejection: Unable to fix {stuck_issue} after 3 attempts "
                                    f"(alignment: {alignment:.2f})"
                                )
                        
                        elif stuck_issue == 'llm_judge_error':
                            # Already handled by circuit breaker above
                            pass
                        
                        else:
                            # For other issues (length, vocabulary, diversity): force reject
                            state['final_decision'] = 'reject'
                            state['rejection_reasons'].append(
                                f"Forced rejection: Unable to fix {stuck_issue} after 3 attempts"
                            )
                        
                        # Log and track forced decision
                        if state['final_decision'] == 'reject':
                            self.logger.log_rejection(
                                state['review_id'],
                                state['rejection_reasons'],
                                state['regeneration_count'] + 1
                            )
                            
                            for reason in state['rejection_reasons']:
                                self.metrics_tracker.metrics.add_rejection_reason(reason)
                        
                        return state
            
            # NORMAL REGENERATION LOGIC
            # Adaptive max attempts based on rating difficulty
            max_attempts = self.config.quality_thresholds.max_regeneration_attempts
            rating = state['rating']
            
            # Reduce attempts for harder ratings (1-3 stars are harder to get right)
            if rating in [1, 2, 3]:
                max_attempts = min(max_attempts, 4)  # Cap at 4 attempts
                self.logger.debug(
                    f"Rating {rating} is difficult - capping attempts at {max_attempts}"
                )
            
            # Decide: regenerate or reject
            state['final_decision'] = (
                'regenerate' if state['regeneration_count'] < max_attempts 
                else 'reject'
            )
            
            # Log rejection
            self.logger.log_rejection(
                state['review_id'],
                state['rejection_reasons'],
                state['regeneration_count'] + 1
            )
            
            # Track rejection reasons in metrics
            for reason in state['rejection_reasons']:
                self.metrics_tracker.metrics.add_rejection_reason(reason)
        
        else:
            # ACCEPT PATH - No rejection reasons
            state['final_decision'] = 'accept'
            
            # Get LLM judge score - only if it ran successfully
            llm_score = self._get_llm_judge_score(state)
            
            # Record successful generation
            self.metrics_tracker.record_generation(
                model_name=state['model_used'],
                accepted=True,
                time_taken=state['generation_time'],
                cost=state['cost'],
                quality_score=state['statistical_metrics'].get('self_bleu_diversity', 0),
                llm_judge_score=llm_score,  # ✅ Will be None if skipped/errored
                statistical_metrics=state['statistical_metrics']
            )
            
            self.logger.info(
                f"✅ Review {state['review_id']} accepted after "
                f"{state['regeneration_count'] + 1} attempt(s)!"
            )
        
        return state
    def _should_regenerate(self, state: ReviewGenerationState) -> str:
        """Determine next step based on decision"""
        
        decision = state['final_decision']
        
        if decision == 'regenerate':
            state['regeneration_count'] += 1
            state['rejection_reasons'] = []
            return 'regenerate'
        elif decision == 'accept':
            return 'accept'
        else:
            return 'reject'
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate complete dataset with production safeguards"""
        
        target_samples = self.config.generation.target_samples
        personas = self.config.personas
        rating_dist = self.config.rating_distribution
        
        self.logger.info(f"Starting dataset generation: {target_samples} samples")
        
        generated_count = 0
        attempts = 0
        max_attempts = target_samples * 10  # INCREASED safety limit
        
        # Track consecutive failures
        consecutive_failures = 0
        max_consecutive_failures = 20  # Stop if stuck
        
        while generated_count < target_samples and attempts < max_attempts:
            attempts += 1
            
            # Safety check: too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                self.logger.error(
                    f"Too many consecutive failures ({consecutive_failures}). "
                    f"Check your configuration or model quality."
                )
                break
            
            # Select persona and rating
            persona = random.choices(personas, weights=[p.weight for p in personas], k=1)[0]
            ratings = [1, 2, 3, 4, 5]
            rating_weights = [rating_dist[f"{r}_star"] for r in ratings]
            rating = random.choices(ratings, weights=rating_weights, k=1)[0]
            
            # Create initial state
            state = ReviewGenerationState(
                review_id=f"rev_{generated_count + 1:04d}",
                review_text="",
                rating=rating,
                persona=persona.dict(),
                domain=self.config.generation.domain,
                model_used="",
                seed_words=[],
                generation_time=0.0,
                cost=0.0,
                statistical_metrics={},
                llm_judge_results={},
                regeneration_count=0,
                final_decision="",
                rejection_reasons=[],
                existing_reviews=self.all_reviews_text.copy()
            )
            
            # INCREASED recursion limit
            max_regen = self.config.quality_thresholds.max_regeneration_attempts
            recursion_limit = max(100, max_regen * 10)  # Much higher
            
            try:
                # Run workflow
                final_state = self.workflow.invoke(
                    state,
                    config={"recursion_limit": recursion_limit}
                )
                
                # Check if accepted
                if final_state['final_decision'] == 'accept':
                    self.accepted_reviews.append(final_state)
                    self.all_reviews_text.append(final_state['review_text'])
                    generated_count += 1
                    consecutive_failures = 0  # Reset counter
                    
                    self.logger.info(f"Progress: {generated_count}/{target_samples} reviews generated")
                
                else:
                    # Record rejection
                    consecutive_failures += 1
                    self.metrics_tracker.record_generation(
                        model_name=final_state.get('model_used', 'unknown'),
                        accepted=False,
                        time_taken=final_state.get('generation_time', 0),
                        cost=final_state.get('cost', 0)
                    )
            
            except Exception as e:
                self.logger.error(f"Workflow error for {state['review_id']}: {e}")
                consecutive_failures += 1
                continue
        
        self.logger.info(
            f"Dataset generation complete: {generated_count} reviews accepted "
            f"out of {attempts} attempts"
        )
        
        return self.accepted_reviews
   
        """Generate complete dataset"""
        
        target_samples = self.config.generation.target_samples
        personas = self.config.personas
        rating_dist = self.config.rating_distribution
        
        self.logger.info(f"Starting dataset generation: {target_samples} samples")
        
        generated_count = 0
        attempts = 0
        max_attempts = target_samples * 5
        
        while generated_count < target_samples and attempts < max_attempts:
            attempts += 1
            
            # Select persona based on weights
            persona = random.choices(
                personas,
                weights=[p.weight for p in personas],
                k=1
            )[0]
            
            # Select rating based on distribution
            ratings = [1, 2, 3, 4, 5]
            rating_weights = [rating_dist[f"{r}_star"] for r in ratings]
            rating = random.choices(ratings, weights=rating_weights, k=1)[0]
            
            # Create initial state
            state = ReviewGenerationState(
                review_id=f"rev_{generated_count + 1:04d}",
                review_text="",
                rating=rating,
                persona=persona.dict(),
                domain=self.config.generation.domain,
                model_used="",
                seed_words=[],
                generation_time=0.0,
                cost=0.0,
                statistical_metrics={},
                llm_judge_results={},
                regeneration_count=0,
                final_decision="",
                rejection_reasons=[],
                existing_reviews=self.all_reviews_text.copy()
            )
            
            # Run workflow
            final_state = self.workflow.invoke(
                state,
                config={"recursion_limit": max(50, self.config.quality_thresholds.max_regeneration_attempts * 5)}
            )
            
            # Check if accepted
            if final_state['final_decision'] == 'accept':
                self.accepted_reviews.append(final_state)
                self.all_reviews_text.append(final_state['review_text'])
                generated_count += 1
                
                self.logger.info(f"Progress: {generated_count}/{target_samples} reviews generated")
            
            else:
                # Record rejection
                self.metrics_tracker.record_generation(
                    model_name=final_state.get('model_used', 'unknown'),
                    accepted=False,
                    time_taken=final_state.get('generation_time', 0),
                    cost=final_state.get('cost', 0)
                )
        
        self.logger.info(f"Dataset generation complete: {generated_count} reviews accepted out of {attempts} attempts")
        
        return self.accepted_reviews
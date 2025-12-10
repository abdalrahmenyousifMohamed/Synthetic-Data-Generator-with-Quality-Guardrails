from typing import Dict, Any, List
from src.utils.faker_utils import DiversityInjector
from src.models.model_router import ModelRouter


class ReviewGeneratorAgent:
    """Agent responsible for generating synthetic reviews"""
    
    def __init__(self, model_router: ModelRouter, diversity_injector: DiversityInjector):
        self.model_router = model_router
        self.diversity_injector = diversity_injector
    
    def generate_review(
        self,
        persona: Dict[str, Any],
        rating: int,
        domain: str,
        existing_reviews: List[str] = None,
        previous_feedback: str = None,  
        regeneration_count: int = 0  
    ) -> Dict[str, Any]:
        """Generate a single review"""
        
        
        seed_words = self.diversity_injector.generate_seed_words(count=10)
        lengths = self.diversity_injector.generate_random_lengths()
        tone = self.diversity_injector.get_random_tone()
        style = self.diversity_injector.get_random_style()
        focus_areas = self.diversity_injector.get_random_focus_areas()
        scenario = self.diversity_injector.get_random_scenario()
        
        
        enhanced_persona = self.diversity_injector.generate_persona_variation(persona)
        
        
        system_prompt = self._build_system_prompt(
            enhanced_persona, rating, domain, seed_words,
            tone, style, focus_areas, scenario,
            previous_feedback, regeneration_count  
        )
        
        
        user_prompt = self._build_user_prompt(rating, lengths, previous_feedback)
        
        
        result = self.model_router.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=min(1.0 + (regeneration_count * 0.1), 1.5)  
        )
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "review_text": None
            }
        
        return {
            "success": True,
            "review_text": result["text"],
            "model_used": result["model_key"],
            "generation_time": result["time"],
            "cost": result["cost"],
            "metadata": {
                "persona": enhanced_persona,
                "rating": rating,
                "domain": domain,
                "seed_words": seed_words,
                "tone": tone,
                "style": style,
                "focus_areas": focus_areas,
                "scenario": scenario
            }
        }
    
    def _build_system_prompt(
        self,
        persona: Dict[str, Any],
        rating: int,
        domain: str,
        seed_words: List[str],
        tone: str,
        style: str,
        focus_areas: List[str],
        scenario: str,
        previous_feedback: str = None,
        regeneration_count: int = 0
    ) -> str:
        """Build system prompt with diversity injection and feedback"""
        
        rating_guidance = {
            1: "extremely negative experience with major issues - EXPRESS STRONG DISSATISFACTION",
            2: "poor experience with significant problems - EXPRESS CLEAR DISSATISFACTION",
            3: "mixed experience with both pros and cons - BALANCED, SLIGHTLY CRITICAL TONE",
            4: "good experience with minor issues - EXPRESS SATISFACTION with some constructive criticism",
            5: "excellent experience, highly satisfied - EXPRESS ENTHUSIASM AND STRONG SATISFACTION"
        }
        
        
        sentiment_instruction = ""
        if rating >= 4:
            sentiment_instruction = """
CRITICAL SENTIMENT REQUIREMENTS:
- Use POSITIVE language: great, excellent, helpful, easy, satisfied, solid, effective, useful
- Express satisfaction and approval throughout
- Even when mentioning weaknesses, frame them constructively
- Tone should be appreciative and recommending"""
        elif rating <= 2:
            sentiment_instruction = """
CRITICAL SENTIMENT REQUIREMENTS:
- Use NEGATIVE/CRITICAL language: disappointing, frustrating, difficult, poor, lacking, limited
- Express dissatisfaction and criticism
- Highlight problems and pain points
- Tone should be critical and cautionary"""
        else:
            sentiment_instruction = """
CRITICAL SENTIMENT REQUIREMENTS:
- BALANCE positive and negative equally
- Mention both strengths AND weaknesses
- Tone should be neutral to slightly critical
- Don't be overly enthusiastic or overly harsh"""
        
        
        feedback_section = ""
        if previous_feedback and regeneration_count > 0:
            feedback_section = f"""
⚠️ PREVIOUS ATTEMPT WAS REJECTED - IMPROVE THIS:
{previous_feedback}

YOU MUST ADDRESS THESE ISSUES:
- If "lacks technical details": Include SPECIFIC technical features, metrics, API details, performance numbers
- If "insufficient expertise": Demonstrate deep knowledge with technical terminology and advanced use cases
- If "not unique": Use different structure, examples, and phrasing than before
- If "not authentic": Add more personal experiences, specific scenarios, concrete examples

CRITICAL: DO NOT repeat the same approach. CHANGE your content significantly.
"""
        
        
        technical_depth = ""
        if domain.lower() in ["developer tools", "devtools", "saas", "software"]:
            technical_depth = f"""
TECHNICAL DEPTH REQUIREMENTS (CRITICAL for {domain}):
- Mention SPECIFIC technical features: APIs, integrations, configurations, performance metrics
- Include technical terminology appropriate for your role: {persona.get('random_job', 'Professional')}
- Provide concrete examples: "reduced build time by 40%", "integrated with GitHub Actions", "supports OAuth 2.0"
- Discuss technical trade-offs: scalability, performance, security, developer experience
- Reference technical workflows: CI/CD pipelines, deployment strategies, monitoring setup
- Be specific about your tech stack: languages, frameworks, tools you integrated with

Example of good technical depth:
"As a DevOps engineer, I integrated CircleCI with our Kubernetes cluster using their API. The pipeline now runs 40% faster with parallelism set to 4x, and caching our Docker layers saved us $500/month in compute costs. The YAML syntax is cleaner than Jenkins Groovy, though debugging failed steps in the UI still requires too many clicks."
"""
        
        base_prompt = f"""You are writing an authentic {domain} product review as a {persona['characteristics']}.

Your Profile:
- Experience Level: {persona['experience']}
- Role: {persona.get('random_job', 'Professional')}
- Company Type: {persona.get('random_company_type', 'Mid-size company')}
- Use Case: {persona.get('random_use_case', 'General usage')}

Review Context:
- Rating: {rating}/5 stars - {rating_guidance.get(rating, 'Balanced assessment')}
- Tone: {tone}
- Writing Style: {style}
- Usage Scenario: {scenario}
- Focus Areas: {', '.join(focus_areas)}

{sentiment_instruction}

{technical_depth}

{feedback_section}

CRITICAL INSTRUCTIONS:
1. Write a genuine, authentic review that sounds like a real human wrote it
2. Include SPECIFIC details and CONCRETE examples from your experience
3. Match the emotional tone to the rating (very important!)
4. Naturally incorporate these seed words for diversity: {', '.join(seed_words[:5])}
5. Focus on the specified areas: {', '.join(focus_areas)}
6. Use the {style} writing style
7. Avoid generic phrases like "game changer", "highly recommend"
8. Be specific about your use case: {scenario}
9. Include both strengths and weaknesses (even for 5-star reviews)
10. Make it sound natural and conversational, not like a template
11. Demonstrate technical expertise with specific examples and metrics
12. {"CHANGE your approach from the previous rejected attempt" if regeneration_count > 0 else "Be unique and creative"}

Write the complete review now. Make it authentic, technically detailed, and unique."""

        return base_prompt
    
    def _build_user_prompt(self, rating: int, lengths: Dict[str, int], previous_feedback: str = None) -> str:
        """Build user prompt"""
        
        emphasis = ""
        if previous_feedback:
            if "technical" in previous_feedback.lower() or "expertise" in previous_feedback.lower():
                emphasis = "\n\nEMPHASIS: Include specific technical details, metrics, and demonstrate deep expertise."
            elif "unique" in previous_feedback.lower():
                emphasis = "\n\nEMPHASIS: Be creative and unique. Use different examples and structure."
            elif "authentic" in previous_feedback.lower():
                emphasis = "\n\nEMPHASIS: Make it sound like a real person wrote it. Add personal experiences."
        
        return f"""Write a complete {rating}-star review.

Length: Between {lengths['min']} and {lengths['max']} words.{emphasis}

Write the review now (only the review text, no additional formatting):"""
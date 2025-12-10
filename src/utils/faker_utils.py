from faker import Faker
from typing import Dict, Any, List
import random


class DiversityInjector:
    """Generate random elements for review diversity"""
    
    def __init__(self, seed: int = None):
        self.fake = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
    
    def generate_seed_words(self, count: int = 10) -> List[str]:
        """Generate random seed words for diversity"""
        return [self.fake.word() for _ in range(count)]
    
    def generate_random_lengths(self) -> Dict[str, int]:
        """Generate random min/max lengths"""
        min_length = self.fake.random_int(min=50, max=100)
        max_length = self.fake.random_int(min=150, max=500)
        return {"min": min_length, "max": max_length}
    
    def generate_persona_variation(self, base_persona: Dict[str, Any]) -> Dict[str, Any]:
        """Add random variations to persona"""
        persona = base_persona.copy()
        
        persona["random_name"] = self.fake.name()
        persona["random_job"] = self.fake.job()
        persona["random_company_type"] = self.fake.random_element([
            "startup", "mid-size", "enterprise", "consulting", "agency"
        ])
        persona["random_use_case"] = self.fake.catch_phrase()
        persona["random_pain_point"] = self.fake.bs()
        
        return persona
    
    def get_random_tone(self) -> str:
        """Get random emotional tone"""
        return self.fake.random_element([
            "enthusiastic", "balanced", "critical", "neutral",
            "frustrated", "satisfied", "impressed", "disappointed"
        ])
    
    def get_random_style(self) -> str:
        """Get random writing style"""
        return self.fake.random_element([
            "technical", "casual", "professional", "storytelling",
            "analytical", "conversational", "formal", "informal"
        ])
    
    def get_random_focus_areas(self, count: int = 3) -> List[str]:
        """Get random product focus areas"""
        all_areas = [
            "performance", "ui/ux", "pricing", "support", "documentation",
            "integrations", "reliability", "scalability", "security",
            "ease of use", "features", "updates", "community"
        ]
        return random.sample(all_areas, min(count, len(all_areas)))
    
    def get_random_scenario(self) -> str:
        """Generate random usage scenario"""
        scenarios = [
            "daily workflow", "team collaboration", "client projects",
            "personal projects", "enterprise deployment", "testing phase",
            "production use", "evaluation period", "migration from another tool"
        ]
        return self.fake.random_element(scenarios)
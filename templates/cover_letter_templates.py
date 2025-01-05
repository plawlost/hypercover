from typing import Dict
import functools

@functools.lru_cache(maxsize=None)
def get_template_info() -> Dict:
    """Get template information with infinite caching"""
    return {
        template_id: {
            "name": template["name"],
            "description": template["description"]
        }
        for template_id, template in COVER_LETTER_TEMPLATES.items()
    }

@functools.lru_cache(maxsize=None)
def get_template_prompt(template_id: str) -> Dict:
    """Get template prompt with infinite caching"""
    if template_id not in COVER_LETTER_TEMPLATES:
        raise ValueError(f"Template {template_id} not found")
    return COVER_LETTER_TEMPLATES[template_id]

# Pre-compiled optimized templates
COVER_LETTER_TEMPLATES = {
    "modern_professional": {
        "name": "Modern Professional",
        "description": "A contemporary and straightforward approach that emphasizes achievements and metrics",
        "prompt_template": """Create a modern and metrics-focused cover letter that:
1. Opens with a strong value proposition
2. Emphasizes quantifiable achievements
3. Demonstrates clear company research
4. Maintains professional tone
5. Focuses on future contribution potential""",
        "structure": {
            "intro": True,
            "achievements": True,
            "company_alignment": True,
            "closing": True
        },
        "max_length": 2000,
        "tone_options": ["formal", "confident", "professional"]
    },
    
    "creative_narrative": {
        "name": "Creative Narrative",
        "description": "A storytelling approach that connects personal journey with company values",
        "prompt_template": """Create a narrative-driven cover letter that:
1. Opens with a compelling personal story
2. Weaves experience into a coherent journey
3. Makes emotional connections with company values
4. Shows personality while maintaining professionalism
5. Emphasizes cultural fit and shared vision""",
        "structure": {
            "story": True,
            "journey": True,
            "values": True,
            "vision": True
        },
        "max_length": 2500,
        "tone_options": ["casual", "conversational", "authentic"]
    },
    
    "technical_expert": {
        "name": "Technical Expert",
        "description": "A detailed approach focusing on technical expertise and specific project experiences",
        "prompt_template": """Create a technically-focused cover letter that:
1. Leads with specific technical expertise
2. Details project implementations and outcomes
3. Uses technical terminology appropriately
4. Demonstrates problem-solving approach
5. Shows understanding of company's technical challenges""",
        "structure": {
            "expertise": True,
            "projects": True,
            "technical_alignment": True,
            "problem_solving": True
        },
        "max_length": 2200,
        "tone_options": ["technical", "precise", "analytical"]
    },
    
    "executive_brief": {
        "name": "Executive Brief",
        "description": "A concise, high-level approach focusing on strategic impact and leadership",
        "prompt_template": """Create an executive-style cover letter that:
1. Emphasizes strategic thinking and leadership
2. Focuses on business impact and results
3. Demonstrates industry insight
4. Projects executive presence
5. Highlights vision alignment""",
        "structure": {
            "leadership": True,
            "impact": True,
            "strategy": True,
            "vision": True
        },
        "max_length": 1800,
        "tone_options": ["authoritative", "strategic", "executive"]
    }
}

# Pre-compile common template elements
COMMON_ELEMENTS = {
    "salutations": [
        "Dear Hiring Manager",
        "Dear [Company] Team",
        "Dear Recruiting Team"
    ],
    "closings": [
        "Best regards",
        "Sincerely",
        "Kind regards"
    ],
    "transition_phrases": [
        "Furthermore",
        "Additionally",
        "Moreover",
        "In addition"
    ]
}

def get_template_structure(template_id: str) -> Dict:
    """Get template structure efficiently"""
    return COVER_LETTER_TEMPLATES[template_id]["structure"]

def get_template_tone_options(template_id: str) -> list:
    """Get template tone options efficiently"""
    return COVER_LETTER_TEMPLATES[template_id]["tone_options"]

def get_template_max_length(template_id: str) -> int:
    """Get template maximum length efficiently"""
    return COVER_LETTER_TEMPLATES[template_id]["max_length"]

def get_common_elements(element_type: str) -> list:
    """Get common template elements efficiently"""
    return COMMON_ELEMENTS.get(element_type, []) 
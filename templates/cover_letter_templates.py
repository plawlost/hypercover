"""Cover letter templates and template management functions."""

TEMPLATES = {
    "standard": {
        "name": "Standard Professional",
        "description": "A classic, professional cover letter format suitable for most industries",
        "prompt_template": """
Dear Hiring Manager,

I am writing to express my keen interest in the {position} position at {company}. With my background in {background} and experience in {experience}, I believe I would be a valuable addition to your team.

{company_specific_content}

{skills_and_achievements}

{closing_paragraph}

Best regards,
{name}
""",
        "structure": {
            "intro": True,
            "experience": True,
            "company_alignment": True,
            "closing": True
        },
        "tone_options": ["formal", "professional", "confident"]
    },
    "creative": {
        "name": "Creative & Modern",
        "description": "A more dynamic and personal approach, ideal for creative industries and startups",
        "prompt_template": """
Hi {company} team,

I'm excited about the opportunity to join your team as a {position}. Your company's focus on {company_values} really resonates with me, and I'd love to contribute to your mission.

{creative_opening}

{skills_and_impact}

{passion_and_culture_fit}

Looking forward to discussing how I can contribute to {company}'s continued success.

Best,
{name}
""",
        "structure": {
            "story": True,
            "impact": True,
            "culture_fit": True,
            "vision": True
        },
        "tone_options": ["casual", "enthusiastic", "authentic"]
    },
    "technical": {
        "name": "Technical Professional",
        "description": "Focused on technical skills and achievements, perfect for tech roles",
        "prompt_template": """
Dear Hiring Team,

I am writing to apply for the {position} position at {company}. With my strong background in {technical_skills} and proven track record in {achievements}, I am confident in my ability to contribute to your technical team.

{technical_experience}

{project_highlights}

{technical_alignment}

Thank you for considering my application.

Best regards,
{name}
""",
        "structure": {
            "expertise": True,
            "projects": True,
            "technical_alignment": True,
            "problem_solving": True
        },
        "tone_options": ["technical", "precise", "analytical"]
    }
}

def get_template_info():
    """Get information about available templates."""
    return {
        template_id: {
            "name": template["name"],
            "description": template["description"]
        }
        for template_id, template in TEMPLATES.items()
    }

def get_template_prompt(template_id: str):
    """Get the prompt template for a specific template ID."""
    if template_id not in TEMPLATES:
        raise ValueError(f"Template ID '{template_id}' not found")
    return TEMPLATES[template_id]

def get_template_structure(template_id: str):
    """Get the structure configuration for a template."""
    if template_id not in TEMPLATES:
        raise ValueError(f"Template ID '{template_id}' not found")
    return TEMPLATES[template_id]["structure"]

def get_template_tone_options(template_id: str):
    """Get the available tone options for a template."""
    if template_id not in TEMPLATES:
        raise ValueError(f"Template ID '{template_id}' not found")
    return TEMPLATES[template_id]["tone_options"] 
from marshmallow import Schema, fields, validate, ValidationError
from typing import Dict, Any, List
import re

class ExperienceSchema(Schema):
    """Schema for validating professional experience entries"""
    company = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    title = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    start_date = fields.Date(required=True)
    end_date = fields.Date(allow_none=True)
    description = fields.Str(required=True, validate=validate.Length(min=10, max=1000))
    achievements = fields.List(fields.Str(), validate=validate.Length(max=10))

class EducationSchema(Schema):
    """Schema for validating education entries"""
    institution = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    degree = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    field = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    graduation_date = fields.Date(required=True)
    gpa = fields.Float(validate=validate.Range(min=0, max=4.0))

class UserProfileSchema(Schema):
    """Schema for validating user profile data"""
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    email = fields.Email(required=True)
    phone = fields.Str(validate=validate.Regexp(r'^\+?[\d\s-]{10,20}$'))
    linkedin = fields.Url(schemes={'http', 'https'})
    summary = fields.Str(validate=validate.Length(min=50, max=500))
    experience = fields.List(
        fields.Nested(ExperienceSchema()),
        required=True,
        validate=validate.Length(min=1, max=10)
    )
    education = fields.List(
        fields.Nested(EducationSchema()),
        required=True,
        validate=validate.Length(min=1, max=5)
    )
    skills = fields.List(
        fields.Str(validate=validate.Length(min=1, max=50)),
        required=True,
        validate=validate.Length(min=1, max=20)
    )
    
    @validate.pre_load
    def sanitize_data(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Pre-process and sanitize input data"""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON data")
                
        # Basic XSS prevention
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potential script tags and other dangerous content
                data[key] = re.sub(r'<[^>]*?>', '', value)
                
        return data

class GenerationRequestSchema(Schema):
    """Schema for validating cover letter generation requests"""
    template_id = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=50)
    )
    user_profile = fields.Nested(
        UserProfileSchema(),
        required=True
    )
    session_id = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=50),
            validate.Regexp(r'^[a-zA-Z0-9_-]+$')
        ]
    )
    
    @validate.pre_load
    def validate_template(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate template ID against available templates"""
        template_id = data.get('template_id')
        if template_id and not self.is_valid_template(template_id):
            raise ValidationError("Invalid template ID")
        return data
    
    @staticmethod
    def is_valid_template(template_id: str) -> bool:
        """Check if template ID exists in available templates"""
        from templates.cover_letter_templates import get_template_info
        available_templates = get_template_info()
        return template_id in available_templates 
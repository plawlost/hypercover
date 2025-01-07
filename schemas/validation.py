from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from typing import Dict, Any, List
import re
import json
from datetime import datetime

def parse_date(date_str):
    """Parse various date formats into ISO format"""
    if not date_str:
        return None
    
    # Handle GMT format
    if isinstance(date_str, str) and 'GMT' in date_str:
        try:
            dt = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S GMT')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Try parsing ISO format
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        pass
    
    # Return as is if it's already in YYYY-MM-DD format
    if isinstance(date_str, str) and len(date_str.split('-')) == 3:
        return date_str
    
    return None

class ExperienceSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    company = fields.Str(required=True)
    title = fields.Str(required=True)
    description = fields.Str(required=True)
    achievements = fields.List(fields.Str(), required=False, missing=[])
    start_date = fields.Function(
        deserialize=parse_date,
        required=True,
        error_messages={"required": "Start date is required"}
    )
    end_date = fields.Function(
        deserialize=parse_date,
        required=True,
        error_messages={"required": "End date is required"}
    )

class EducationSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    institution = fields.Str(required=True)
    degree = fields.Str(required=True)
    field = fields.Str(required=True)
    graduation_date = fields.Function(
        deserialize=parse_date,
        required=True,
        error_messages={"required": "Graduation date is required"}
    )
    gpa = fields.Float(required=False)

class UserProfileSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    phone = fields.Str(required=True)
    linkedin = fields.Url(required=True)
    summary = fields.Str(required=True)
    skills = fields.List(fields.Str(), required=True)
    experience = fields.List(fields.Nested(ExperienceSchema()), required=True)
    education = fields.List(fields.Nested(EducationSchema()), required=True)

class GenerationRequestSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    template_id = fields.Str(required=True)
    user_profile = fields.Nested(UserProfileSchema(), required=True)
    session_id = fields.Str(required=True) 
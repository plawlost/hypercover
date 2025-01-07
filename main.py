from flask import Flask, render_template, request, send_file, jsonify, url_for, send_from_directory, redirect, after_this_request, Response
from werkzeug.utils import secure_filename
from io import BytesIO
from bulk_processor import BulkCoverLetterGenerator, API_ERRORS
import os
from pathlib import Path
import json
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from templates.cover_letter_templates import get_template_info, get_template_prompt, get_template_structure, get_template_tone_options
from flask_socketio import SocketIO, emit
import threading
import uuid
from flask_caching import Cache
from werkzeug.middleware.proxy_fix import ProxyFix
import gzip
import functools
from concurrent.futures import ThreadPoolExecutor
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from limits.storage import MemoryStorage, RedisStorage
import redis
from marshmallow import Schema, fields, validate, ValidationError, validates
import logging
from werkzeug.security import safe_join
import bleach
from prometheus_client import Counter, Histogram
import companyfinder
from linkedin_profile_scraper import get_profile_info
from linkedin_api_client import LinkedInClient
from schemas.validation import GenerationRequestSchema, UserProfileSchema
import atexit
import re
from typing import Dict, Any

def sanitize_input(input_str):
    """Sanitize user input to prevent XSS and injection attacks"""
    if not isinstance(input_str, str):
        return ''
    # Remove any HTML/script tags
    cleaned = bleach.clean(input_str, tags=[], strip=True)
    # Only allow alphanumeric characters, hyphens, underscores, and some special characters
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-_./]', '', cleaned)
    return cleaned.strip()

def sanitize_linkedin_url(url):
    """Specifically sanitize LinkedIn URLs while preserving valid URL characters"""
    if not isinstance(url, str):
        return ''
    # Remove any HTML/script tags
    cleaned = bleach.clean(url, tags=[], strip=True)
    # Allow common URL characters while still preventing injection
    cleaned = re.sub(r'[^a-zA-Z0-9\-_./:%?=&]', '', cleaned)
    return cleaned.strip()

def validate_file_extension(filename):
    """Validate that the file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in production mode
is_production = os.environ.get('FLASK_ENV') != 'development'

# Performance metrics
REQUEST_LATENCY = Histogram('request_processing_seconds', 'Time spent processing request')

# Initialize Flask app with security headers
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.urandom(24)

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self' ws: wss:;"
    )
    return response

# Configure Redis for rate limiting if available
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    storage_url = "redis://localhost:6379"
    logger.info("Redis storage being used for rate limiting")
except:
    storage_url = "memory://"
    logger.warning("Redis not available, falling back to in-memory cache")

# Initialize rate limiter with the appropriate storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=storage_url
)

# Initialize Socket.IO for real-time progress updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize bulk processor with API keys
bulk_processor = BulkCoverLetterGenerator(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    deepseek_api_key=os.getenv('DEEPSEEK_API_KEY')
)

# Initialize cache with simple backend
cache = Cache(app)

# Configure for proper IP handling behind proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Create required directories with proper permissions
for directory in ['uploads', 'generated_letters']:
    directory_path = Path(directory)
    try:
        directory_path.mkdir(mode=0o755, parents=True, exist_ok=True)
        if not directory_path.exists():
            raise Exception(f"Failed to create {directory} directory")
    except Exception as e:
        logger.error(f"Error creating {directory} directory: {str(e)}")
        raise

# Initialize Socket.IO with optimized settings
socketio = SocketIO(
    app,
    async_mode='threading',
    cors_allowed_origins="*",
    ping_timeout=10,
    ping_interval=5
)

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Initialize the application
async def init_app():
    with app.app_context():
        await bulk_processor.initialize()

# Run initialization when starting the app
asyncio.run(init_app())

# Cleanup function to close resources
def cleanup():
    executor.shutdown(wait=True)
    if bulk_processor.session:
        async def close_session():
            await bulk_processor.session.close()
        asyncio.run(close_session())

# Register cleanup function
atexit.register(cleanup)

def validate_template_id(template_id: str) -> bool:
    """Validate that the template ID exists and has required structure."""
    try:
        template_info = get_template_info()
        template_prompt = get_template_prompt(template_id)
        template_structure = get_template_structure(template_id)
        template_tone = get_template_tone_options(template_id)
        return template_id in template_info
    except ValueError:
        return False

@app.route('/api/bulk-generate', methods=['POST'])
async def bulk_generate():
    try:
        # Get form data without awaiting
        form_data = request.form
        user_profile = json.loads(form_data.get('user_profile'))
        template_id = form_data.get('template_id')
        session_id = form_data.get('session_id')
        formats = json.loads(form_data.get('formats', '["docx"]'))  # Default to DOCX if not specified
        
        # Validate formats
        valid_formats = ['docx', 'pdf']
        formats = [fmt for fmt in formats if fmt in valid_formats]
        if not formats:
            formats = ['docx']  # Fallback to DOCX if no valid formats
        
        # Log the request details
        logger.info(f"Received bulk generation request. Form data: {form_data}")
        logger.info(f"Parsed data - template_id: {template_id}, session_id: {session_id}")
        logger.info(f"User profile: {user_profile}")
        logger.info(f"Selected formats: {formats}")
        
        # Validate the data
        if not all([user_profile, template_id]):
            raise ValueError("Missing required fields")
            
        # Save uploaded file
        if 'file' not in request.files:
            raise ValueError("No file uploaded")
            
        uploaded_file = request.files['file']
        if not uploaded_file or not uploaded_file.filename:
            raise ValueError("No file selected")
            
        if not validate_file_extension(uploaded_file.filename):
            raise ValueError("Invalid file type. Only CSV files are allowed.")
            
        # Create upload directory if it doesn't exist
        upload_dir = Path('uploads')
        upload_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        
        # Validate and save the file
        filename = secure_filename(uploaded_file.filename)
        filepath = upload_dir / filename
        uploaded_file.save(str(filepath))
        logger.info(f"File saved successfully at {filepath}")
        
        # Initialize bulk processor if not already done
        if not bulk_processor:
            await initialize_bulk_processor()
        
        # Process the spreadsheet and generate letters
        result = await bulk_processor.process_spreadsheet(
            spreadsheet_path=str(filepath),
            user_profile=user_profile,
            template_id=template_id,
            formats=formats,
            progress_callback=lambda progress: socketio.emit('generation_progress', 
                {'progress': progress}, room=session_id)
        )
        
        # Clean up the uploaded file
        try:
            if filepath.exists():
                filepath.unlink()
            logger.info(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Error cleaning up file {filepath}: {str(e)}")
        
        if result and 'zip_file' in result:
            return jsonify({
                'status': 'success',
                'message': 'Cover letters generated successfully',
                'zip_file': result['zip_file']
            })
        else:
            raise ValueError("Failed to generate cover letters")
            
    except Exception as e:
        logger.error(f"Error in bulk generation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to generate cover letters: {str(e)}"
        }), 500

def gzip_response(f):
    """Decorator to gzip responses"""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        response = f(*args, **kwargs)
        if not isinstance(response, str):
            return response
            
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
            
        gzip_buffer = BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=gzip_buffer) as gz_file:
            gz_file.write(response.encode('utf-8'))
            
        response = gzip_buffer.getvalue()
        
        return Response(
            response,
            content_type='application/json',
            headers={'Content-Encoding': 'gzip'}
        )
    return wrapped

@app.route('/', methods=['GET'])
@cache.cached(timeout=300)
def home():
    """Serve home page with caching"""
    templates = get_template_info()
    return render_template('index.html', templates=templates)

@app.route('/download/template')
def download_template():
    """Download CSV template file"""
    return send_file(
        'templates/job_list_template.csv',
        as_attachment=True,
        download_name='hypercover_template.csv',
        mimetype='text/csv'
    )

@app.route('/download/<filename>')
def download_zip(filename):
    """Download generated zip file with cleanup"""
    try:
        return send_file(
            filename,
            as_attachment=True,
            download_name=filename
        )
    finally:
        # Clean up zip file after download
        try:
            os.remove(filename)
        except:
            pass

@app.route('/api/preview-letter', methods=['POST'])
@gzip_response
def preview_letter():
    """Generate a preview of the cover letter with optimized response"""
    try:
        data = request.get_json()
        template_settings = data.get('template_settings')
        format_type = data.get('format', 'doc')
        
        if not template_settings:
            return jsonify({"error": "Missing template settings"}), 400
            
        # Get template structure and options
        template_id = template_settings.get('baseTemplate')
        structure = get_template_structure(template_id)
        tone_options = get_template_tone_options(template_id)
        
        # Create a new event loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate preview with optimized settings
            preview = loop.run_until_complete(bulk_processor.generate_preview(
                template_settings=template_settings,
                format_type=format_type
            ))
            
            return jsonify({
                "success": True,
                "preview_content": preview,
                "structure": structure,
                "tone_options": tone_options
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Initialize LinkedIn client
linkedin_client = LinkedInClient()

def transform_linkedin_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform LinkedIn API response to match our schema format"""
    if 'error' in profile_data:
        return profile_data

    # Extract contact info
    contact_info = profile_data.get('contact_info', {})
    email = contact_info.get('email_address') or 'user@example.com'
    phone_numbers = contact_info.get('phone_numbers', [])
    phone = phone_numbers[0] if phone_numbers else "+1234567890"

    # Get current experience for summary
    experiences = profile_data.get('experience', [])
    current_role = experiences[0] if experiences else {}
    
    # Build summary from headline and about
    summary_parts = []
    if profile_data.get('headline'):
        summary_parts.append(profile_data['headline'])
    if profile_data.get('summary'):
        summary_parts.append(profile_data['summary'])
    summary = ' '.join(summary_parts)
    
    # Ensure summary meets minimum length requirement
    if len(summary) < 50:
        current_title = current_role.get('title', 'the industry')
        current_company = current_role.get('companyName', 'various organizations')
        summary = f"Experienced professional with a proven track record in {current_title} at {current_company}. Demonstrated expertise in delivering high-quality results and driving business success through innovative solutions and strategic thinking."
    
    # Transform experience items
    transformed_experience = []
    for exp in experiences[:10]:  # Limit to 10 items
        if not exp.get('companyName') or not exp.get('title'):
            continue
            
        # Format dates as YYYY-MM-DD
        time_period = exp.get('timePeriod', {})
        start_date = time_period.get('startDate', {})
        end_date = time_period.get('endDate', {})
        
        start_year = str(start_date.get('year', 2000)).zfill(4)
        start_month = str(start_date.get('month', 1)).zfill(2)
        start_date_str = f"{start_year}-{start_month}-01"
        
        end_year = str(end_date.get('year', start_date.get('year', 2000))).zfill(4)
        end_month = str(end_date.get('month', 12)).zfill(2)
        end_date_str = f"{end_year}-{end_month}-01"
            
        exp_item = {
            'company': exp['companyName'][:100],
            'title': exp['title'][:100],
            'start_date': start_date_str,
            'end_date': end_date_str,
            'description': exp.get('description', 'Responsible for key initiatives and projects.')[:1000],
            'achievements': []
        }
        transformed_experience.append(exp_item)
    
    # Ensure at least one experience entry
    if not transformed_experience:
        transformed_experience = [{
            'company': 'Professional Experience',
            'title': 'Various Roles',
            'start_date': '2000-01-01',
            'end_date': '2000-12-31',
            'description': 'Professional with experience in various roles and organizations.',
            'achievements': []
        }]
    
    # Transform education items
    transformed_education = []
    for edu in profile_data.get('education', [])[:5]:  # Limit to 5 items
        if not edu.get('schoolName'):
            continue
            
        # Format graduation date as YYYY-MM-DD
        time_period = edu.get('timePeriod', {})
        end_date = time_period.get('endDate', {})
        grad_year = str(end_date.get('year', 2000)).zfill(4)
        grad_month = str(end_date.get('month', 12)).zfill(2)
        graduation_date = f"{grad_year}-{grad_month}-01"
            
        edu_item = {
            'institution': edu['schoolName'][:100],
            'degree': edu.get('degreeName', 'Degree Program')[:100],
            'field': edu.get('fieldOfStudy', 'General Studies')[:100],
            'graduation_date': graduation_date,
            'gpa': 3.5
        }
        transformed_education.append(edu_item)
    
    # Ensure at least one education entry
    if not transformed_education:
        transformed_education = [{
            'institution': 'Educational Institution',
            'degree': 'Degree Program',
            'field': 'General Studies',
            'graduation_date': '2000-12-31',
            'gpa': 3.5
        }]
    
    # Transform skills (limit to 20)
    transformed_skills = [
        skill['name'][:50] for skill in profile_data.get('skills', [])[:20]
        if isinstance(skill, dict) and skill.get('name')
    ]
    
    # Ensure at least one skill
    if not transformed_skills:
        transformed_skills = ['Professional Skills']
    
    # Format LinkedIn URL
    linkedin_url = profile_data.get('publicProfileUrl', '')
    if not linkedin_url:
        public_id = profile_data.get('public_id', '')
        if public_id:
            linkedin_url = f"https://www.linkedin.com/in/{public_id}/"
        else:
            linkedin_url = "https://www.linkedin.com/"
    
    # Build the transformed profile
    transformed_profile = {
        'name': (profile_data.get('firstName', '') + ' ' + profile_data.get('lastName', '')).strip()[:100] or 'Professional User',
        'email': email,
        'phone': phone,
        'linkedin': linkedin_url,
        'summary': summary[:500],
        'experience': transformed_experience,
        'education': transformed_education,
        'skills': transformed_skills
    }
    
    return transformed_profile

@app.route('/api/fetch-linkedin-profile', methods=['POST'])
async def fetch_linkedin_profile():
    """Fetch LinkedIn profile information using official API"""
    try:
        # Log the raw request data for debugging
        logger.info(f"Received request data: {request.data}")
        
        # Handle both form data and JSON data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"Processed request data: {data}")
        
        if not data:
            return jsonify(error="Request must include profile data"), 400
            
        # Check for all possible field names
        profile_url = (
            data.get('profile_url') or 
            data.get('linkedinUrl') or 
            data.get('linkedin_url')
        )
        
        if not profile_url:
            return jsonify(error="Profile URL is required (use 'profile_url', 'linkedinUrl', or 'linkedin_url' field)"), 400

        profile_url = sanitize_linkedin_url(profile_url)
        if not profile_url.startswith(('https://www.linkedin.com/in/', 'http://www.linkedin.com/in/', 'www.linkedin.com/in/', 'linkedin.com/in/')):
            return jsonify(error="Invalid LinkedIn profile URL. Must be a valid LinkedIn profile URL (e.g., https://www.linkedin.com/in/username)"), 400

        # Normalize the URL format
        if not profile_url.startswith('https://'):
            profile_url = 'https://' + profile_url.replace('http://', '')
        if not profile_url.startswith('https://www.'):
            profile_url = profile_url.replace('https://', 'https://www.')

        # Extract profile ID from URL and clean it
        profile_id = profile_url.split('/in/')[-1].split('/')[0].split('?')[0].strip()
        logger.info(f"Extracted profile ID: {profile_id}")
        
        # Get profile data using LinkedIn API
        profile_data = linkedin_client.get_profile_data(profile_id)
        logger.info(f"LinkedIn API response type: {type(profile_data)}")
        logger.info(f"LinkedIn API response: {json.dumps(profile_data, indent=2)}")
        
        # Transform the data to match our schema
        transformed_data = transform_linkedin_data(profile_data)
        logger.info(f"Transformed data: {json.dumps(transformed_data, indent=2)}")
        
        if 'error' in transformed_data:
            logger.error(f"Error in transformed data: {transformed_data['error']}")
            return jsonify(transformed_data), 400
            
        # Validate the transformed data
        try:
            schema = UserProfileSchema()
            validated_data = schema.load(transformed_data)
            logger.info("Data validation successful")
            return jsonify(validated_data)
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            logger.error(f"Validation error details: {json.dumps(e.messages, indent=2)}")
            return jsonify(error="Invalid profile data", details=e.messages), 400
        
    except Exception as e:
        logger.error(f"Error fetching LinkedIn profile: {str(e)}", exc_info=True)
        return jsonify(error=str(e)), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({
        "error": "File size limit exceeded (max 16MB)"
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        "error": "Internal server error occurred"
    }), 500

if __name__ == '__main__':
    try:
        socketio.run(
            app,
            debug=not is_production,
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            use_reloader=not is_production
        )
    finally:
        cleanup()

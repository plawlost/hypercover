from flask import Flask, render_template, request, send_file, jsonify, url_for, send_from_directory, redirect, after_this_request
from bulk_processor import BulkCoverLetterGenerator
import os
from pathlib import Path
import json
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from templates.cover_letter_templates import get_template_info, get_template_structure, get_template_tone_options
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
from marshmallow import Schema, fields, validate, ValidationError
import logging
from werkzeug.security import safe_join
import bleach
from prometheus_client import Counter, Histogram
import companyfinder
from linkedin_profile_scraper import get_profile_info
from linkedin_api_client import LinkedInClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_LATENCY = Histogram('request_processing_seconds', 'Time spent processing request')
RATE_LIMIT_HITS = Counter('rate_limit_hits_total', 'Total number of rate limit hits')

# Initialize Flask with security settings
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
app.config['CACHE_TYPE'] = 'simple'  # Use simple cache instead of Redis
app.config['SECRET_KEY'] = os.urandom(24)  # Secure secret key

# Only enable secure cookie settings in production
is_production = os.environ.get('FLASK_ENV') == 'production'
if is_production:
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Try to set up Redis storage for rate limiting, fall back to memory storage if unavailable
storage_url = None
try:
    redis_client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
    redis_client.ping()
    storage_url = "redis://localhost:6379"
    logger.info("Using Redis storage for rate limiting")
except (redis.ConnectionError, redis.TimeoutError) as e:
    logger.warning(f"Redis not available for rate limiting, using memory storage: {str(e)}")
    storage_url = "memory://"

# Initialize rate limiter with the appropriate storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=storage_url
)

# Input validation schemas
class UserProfileSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    experience = fields.List(fields.Dict(), required=True)
    skills = fields.List(fields.Str(), required=True)
    education = fields.List(fields.Dict(), required=True)

class GenerationRequestSchema(Schema):
    template_id = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    user_profile = fields.Nested(UserProfileSchema(), required=True)
    session_id = fields.Str(required=True, validate=validate.Length(min=1, max=50))

def validate_file_extension(filename):
    """Validate file extension for security"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_input(text):
    """Sanitize user input"""
    return bleach.clean(text, strip=True)

@app.before_request
def before_request():
    """Security middleware for all requests"""
    if is_production:
        # Enforce HTTPS only in production
        if not request.is_secure:
            url = request.url.replace('http://', 'https://', 1)
            return redirect(url, code=301)

        # Add security headers only in production
        @after_this_request
        def add_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            return response

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    RATE_LIMIT_HITS.inc()
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address()}")
    return jsonify(error="Rate limit exceeded. Please try again later."), 429

@app.route('/api/bulk-generate', methods=['POST'])
@limiter.limit("10 per minute")
@REQUEST_LATENCY.time()
async def bulk_generate():
    """Handle bulk cover letter generation with security measures"""
    try:
        # Validate request schema
        schema = GenerationRequestSchema()
        try:
            data = schema.load(request.form)
        except ValidationError as err:
            return jsonify(error=err.messages), 400

        # Validate and secure file upload
        if 'file' not in request.files:
            return jsonify(error="No file uploaded"), 400
            
        file = request.files['file']
        if not file or not validate_file_extension(file.filename):
            return jsonify(error="Invalid file type"), 400

        # Secure filename and save path
        filename = secure_filename(file.filename)
        file_path = safe_join(app.config['UPLOAD_FOLDER'], filename)
        
        if not file_path:
            return jsonify(error="Invalid file path"), 400

        # Save file securely
        file.save(file_path)
        
        try:
            # Process with sanitized inputs
            result = await bulk_processor.process_spreadsheet(
                spreadsheet_path=file_path,
                user_profile=data['user_profile'],
                template_id=sanitize_input(data['template_id']),
                session_id=sanitize_input(data['session_id'])
            )
            
            return jsonify(result=result)
            
        finally:
            # Cleanup uploaded file
            os.unlink(file_path)
            
    except Exception as e:
        logger.error(f"Error in bulk generation: {str(e)}")
        return jsonify(error="An error occurred during processing"), 500

# Initialize cache with simple backend
cache = Cache(app)

# Configure for proper IP handling behind proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Initialize Socket.IO with optimized settings
socketio = SocketIO(
    app,
    async_mode='threading',
    cors_allowed_origins="*",
    ping_timeout=10,
    ping_interval=5
)

# Initialize the bulk processor
bulk_processor = BulkCoverLetterGenerator(
    groq_api_key=os.getenv('GROQ_API_KEY')
)

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Initialize the application
async def init_app():
    with app.app_context():
        await bulk_processor.initialize()

# Run initialization when starting the app
asyncio.run(init_app())

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

        profile_url = sanitize_input(profile_url)
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
        
        # Handle different error cases
        if 'error' in profile_data:
            error_type = profile_data['error']
            logger.warning(f"LinkedIn API error: {error_type} - {profile_data['message']}")
            
            if error_type == 'Profile not found':
                return jsonify(error=profile_data['message']), 404
            elif error_type in ['Profile not accessible', 'Access denied']:
                return jsonify(error=profile_data['message']), 403
            elif error_type == 'Rate limit exceeded':
                return jsonify(error=profile_data['message']), 429
            elif error_type == 'Authentication error':
                return jsonify(error=profile_data['message']), 401
            else:
                return jsonify(error=profile_data['message']), 500
        
        # Format the response for the frontend
        formatted_data = {
            'name': f"{profile_data.get('firstName', '')} {profile_data.get('lastName', '')}".strip(),
            'current_role': profile_data.get('headline', ''),
            'skills': [skill.get('name', '') for skill in profile_data.get('skills', []) if isinstance(skill, dict) and 'name' in skill],
            'experience': profile_data.get('experience', []),
            'education': profile_data.get('education', []),
            'summary': profile_data.get('summary', ''),
            'location': profile_data.get('locationName', ''),
            'industry': profile_data.get('industryName', ''),
            'profile_picture': profile_data.get('displayPictureUrl', '') + profile_data.get('img_400_400', ''),
            'public_url': f"https://www.linkedin.com/in/{profile_data.get('public_id', '')}",
            'raw_data': profile_data  # Include raw data for debugging
        }
                
        return jsonify(formatted_data)

    except Exception as e:
        logger.error(f"Error fetching LinkedIn profile: {str(e)}", exc_info=True)
        return jsonify(error="An unexpected error occurred while fetching the profile"), 500

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
    socketio.run(
        app,
        debug=not is_production,  # Enable debug mode in development
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        use_reloader=not is_production  # Enable reloader in development
    )

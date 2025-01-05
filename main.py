from flask import Flask, render_template, request, send_file, jsonify, url_for, send_from_directory
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
    # Enforce HTTPS
    if not request.is_secure and not app.debug:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

    # Add security headers
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
@cache.cached(timeout=3600)  # Cache for 1 hour
def download_template():
    """Download CSV template file with caching"""
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
async def preview_letter():
    """Generate a preview of the cover letter with optimized response"""
    try:
        data = request.get_json()
        template_settings = data.get('template_settings')
        format_type = data.get('format', 'doc')
        
        if not template_settings:
            return jsonify({"error": "Missing template settings"}), 400
            
        # Get template structure and options
        template_id = template_settings.get('template_id')
        structure = get_template_structure(template_id)
        tone_options = get_template_tone_options(template_id)
        
        # Generate preview with optimized settings
        preview = await bulk_processor.generate_preview(
            template_settings=template_settings,
            format_type=format_type
        )
        
        return jsonify({
            "success": True,
            "preview": preview,
            "structure": structure,
            "tone_options": tone_options
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch-linkedin-profile', methods=['POST'])
async def fetch_linkedin_profile():
    """Fetch LinkedIn profile with optimized scraping"""
    try:
        data = request.get_json()
        linkedin_url = data.get('linkedin_url')
        
        if not linkedin_url:
            return jsonify({"error": "Missing LinkedIn URL"}), 400
            
        # Check cache first
        cache_key = f"linkedin_profile:{linkedin_url}"
        cached_profile = cache.get(cache_key)
        if cached_profile:
            return jsonify(cached_profile)
        
        # Fetch and parse profile
        async with aiohttp.ClientSession() as session:
            async with session.get(linkedin_url) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    def parse_profile():
                        soup = BeautifulSoup(html, 'lxml')
                        profile = {
                            "name": soup.find('h1', {'class': 'text-heading-xlarge'}).text.strip(),
                            "current_role": soup.find('div', {'class': 'text-body-medium'}).text.strip(),
                            "skills": [
                                skill.text.strip()
                                for skill in soup.find_all('span', {'class': 'skill'})
                            ],
                            "experience": [
                                {
                                    "title": exp.find('h3').text.strip(),
                                    "company": exp.find('p').text.strip(),
                                    "description": exp.find('div', {'class': 'description'}).text.strip()
                                }
                                for exp in soup.find_all('div', {'class': 'experience-item'})
                            ]
                        }
                        return profile
                    
                    profile = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        parse_profile
                    )
                    
                    # Cache the result
                    cache.set(cache_key, profile)
                    
                    return jsonify(profile)
                    
        return jsonify({"error": "Failed to fetch profile"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        debug=False,  # Disable debug mode in production
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        use_reloader=False  # Disable reloader in production
    )

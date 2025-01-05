from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
import os
import logging
from prometheus_client import Counter, Histogram
from typing import Optional, Dict, Any
from ..schemas.validation import GenerationRequestSchema
from ..services.bulk_processor import BulkCoverLetterGenerator
from ..utils.security import sanitize_input, validate_file_extension
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize logging
logger = logging.getLogger(__name__)

# Metrics
REQUEST_LATENCY = Histogram('api_request_processing_seconds', 'Time spent processing API request')
API_ERRORS = Counter('api_errors_total', 'Total number of API errors')

# Create Blueprint
api = Blueprint('api', __name__)

class CoverLetterAPI:
    def __init__(self, app, bulk_processor: BulkCoverLetterGenerator, upload_folder: str):
        self.bulk_processor = bulk_processor
        self.upload_folder = upload_folder
        
        # Initialize rate limiter
        self.limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register API routes with rate limiting"""
        
        @api.route('/bulk-generate', methods=['POST'])
        @self.limiter.limit("10 per minute")
        @REQUEST_LATENCY.time()
        async def bulk_generate():
            return await self._handle_bulk_generate()
            
        @api.route('/templates', methods=['GET'])
        @self.limiter.limit("100 per hour")
        def get_templates():
            return self._handle_get_templates()
    
    async def _handle_bulk_generate(self) -> Dict[str, Any]:
        """Handle bulk generation request with proper error handling"""
        try:
            # Validate request schema
            schema = GenerationRequestSchema()
            try:
                data = schema.load(request.form)
            except ValidationError as err:
                logger.warning(f"Validation error: {err.messages}")
                return jsonify(error=err.messages), 400

            # Validate and secure file upload
            if 'file' not in request.files:
                return jsonify(error="No file uploaded"), 400
                
            file = request.files['file']
            if not file or not validate_file_extension(file.filename):
                return jsonify(error="Invalid file type"), 400

            # Secure filename and save path
            filename = secure_filename(file.filename)
            file_path = safe_join(self.upload_folder, filename)
            
            if not file_path:
                return jsonify(error="Invalid file path"), 400

            # Save and process file securely
            try:
                file.save(file_path)
                
                # Process with sanitized inputs
                result = await self.bulk_processor.process_spreadsheet(
                    spreadsheet_path=file_path,
                    user_profile=data['user_profile'],
                    template_id=sanitize_input(data['template_id']),
                    session_id=sanitize_input(data['session_id'])
                )
                
                return jsonify(result=result)
                
            finally:
                # Ensure file cleanup
                if os.path.exists(file_path):
                    os.unlink(file_path)
                
        except Exception as e:
            API_ERRORS.inc()
            logger.error(f"Error in bulk generation: {str(e)}")
            return jsonify(error="An error occurred during processing"), 500
    
    def _handle_get_templates(self) -> Dict[str, Any]:
        """Handle template retrieval with caching"""
        try:
            templates = self.bulk_processor.get_available_templates()
            return jsonify(templates=templates)
        except Exception as e:
            API_ERRORS.inc()
            logger.error(f"Error retrieving templates: {str(e)}")
            return jsonify(error="Error retrieving templates"), 500

# Error handlers
@api.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address()}")
    return jsonify(error="Rate limit exceeded. Please try again later."), 429

@api.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    API_ERRORS.inc()
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify(error="An unexpected error occurred"), 500 
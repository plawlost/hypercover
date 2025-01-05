import re
import bleach
from typing import Set, Optional
import hashlib
import secrets
from pathlib import Path
import magic
import logging

logger = logging.getLogger(__name__)

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS: Set[str] = {'csv', 'xlsx', 'xls'}
ALLOWED_MIME_TYPES: Set[str] = {
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
}

# Maximum file size (16MB)
MAX_FILE_SIZE: int = 16 * 1024 * 1024

def validate_file_extension(filename: str) -> bool:
    """
    Validate file extension against allowed extensions
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file_path: str) -> bool:
    """
    Validate file MIME type using libmagic
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if MIME type is allowed, False otherwise
    """
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type in ALLOWED_MIME_TYPES
    except Exception as e:
        logger.error(f"Error validating file type: {str(e)}")
        return False

def validate_file_size(file_path: str) -> bool:
    """
    Validate file size against maximum allowed size
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if file size is within limits, False otherwise
    """
    return Path(file_path).stat().st_size <= MAX_FILE_SIZE

def sanitize_input(text: Optional[str]) -> str:
    """
    Sanitize user input to prevent XSS and other injection attacks
    
    Args:
        text: Input text to sanitize
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
        
    # Remove potentially dangerous HTML/scripts
    clean_text = bleach.clean(
        text,
        tags=[],  # No HTML tags allowed
        strip=True,
        strip_comments=True
    )
    
    # Remove potential SQL injection patterns
    clean_text = re.sub(r'[\'";\-]', '', clean_text)
    
    return clean_text

def generate_secure_filename(original_filename: str) -> str:
    """
    Generate a secure filename with random component
    
    Args:
        original_filename: Original filename
        
    Returns:
        str: Secure filename
    """
    # Get file extension
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    # Generate random component
    random_component = secrets.token_hex(8)
    
    # Create hash of original filename
    filename_hash = hashlib.sha256(original_filename.encode()).hexdigest()[:12]
    
    # Combine components
    secure_name = f"{filename_hash}_{random_component}"
    
    # Add extension if present
    if ext:
        secure_name = f"{secure_name}.{ext}"
        
    return secure_name

def validate_file(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Comprehensive file validation
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check file exists
        if not Path(file_path).exists():
            return False, "File does not exist"
            
        # Validate extension
        if not validate_file_extension(file_path):
            return False, "Invalid file extension"
            
        # Validate MIME type
        if not validate_file_type(file_path):
            return False, "Invalid file type"
            
        # Validate size
        if not validate_file_size(file_path):
            return False, "File size exceeds maximum limit"
            
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        return False, "Error validating file"

def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent path traversal attacks
    
    Args:
        path: Path to sanitize
        
    Returns:
        str: Sanitized path
    """
    # Remove potentially dangerous characters
    clean_path = re.sub(r'[^a-zA-Z0-9_\-./]', '', path)
    
    # Ensure no parent directory traversal
    clean_path = re.sub(r'\.\.|//', '', clean_path)
    
    return clean_path 
from linkedin_api import Linkedin
import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkedInClient:
    def __init__(self):
        """Initialize LinkedIn client with credentials from environment variables"""
        self.email = os.getenv('LINKEDIN_USERNAME')
        self.password = os.getenv('LINKEDIN_PASSWORD')
        self._api = None
        
        # Validate credentials on initialization
        if not self.email or not self.password:
            logger.error("LinkedIn credentials not found in environment variables")
            raise ValueError("LinkedIn credentials not found in environment variables. "
                           "Please set LINKEDIN_USERNAME and LINKEDIN_PASSWORD in .env file")
        
    def _initialize_api(self):
        """Initialize the LinkedIn API client if not already initialized"""
        if not self._api:
            try:
                logger.info(f"Initializing LinkedIn API client with email: {self.email}")
                self._api = Linkedin(self.email, self.password)
                logger.info("LinkedIn API client initialized successfully")
            except Exception as e:
                error_msg = str(e).lower()
                if "challenge" in error_msg:
                    logger.error("LinkedIn security challenge detected")
                    raise ValueError("LinkedIn security challenge detected. Please log in through a browser first.")
                elif "unauthorized" in error_msg:
                    logger.error("Invalid LinkedIn credentials")
                    raise ValueError("Invalid LinkedIn credentials. Please check your username and password.")
                else:
                    logger.error(f"Failed to initialize LinkedIn API client: {error_msg}")
                    raise
    
    def get_profile_data(self, profile_id: str) -> Dict[str, Any]:
        """
        Get comprehensive profile data for a given LinkedIn profile
        
        Args:
            profile_id: LinkedIn profile ID or vanity URL name (e.g., 'john-doe' from linkedin.com/in/john-doe)
            
        Returns:
            Dictionary containing profile data or error information
        """
        try:
            self._initialize_api()
            logger.info(f"Fetching profile data for: {profile_id}")
            
            try:
                # Get basic profile information
                profile = self._api.get_profile(profile_id)
                if not profile:
                    return {
                        'error': 'Profile not found',
                        'message': f'Could not find LinkedIn profile with ID: {profile_id}'
                    }
                
                # Get contact information
                try:
                    contact_info = self._api.get_profile_contact_info(profile_id)
                except Exception as e:
                    logger.warning(f"Could not fetch contact info: {str(e)}")
                    contact_info = {}
                
                # Get profile skills
                try:
                    skills = self._api.get_profile_skills(profile_id)
                except Exception as e:
                    logger.warning(f"Could not fetch skills: {str(e)}")
                    skills = []
                
                # Combine all information
                full_profile = {
                    **profile,
                    'contact_info': contact_info,
                    'skills': skills
                }
                
                logger.info(f"Successfully retrieved profile data for {profile_id}")
                return full_profile
                
            except Exception as e:
                error_msg = str(e).lower()
                if "can't be accessed" in error_msg:
                    return {
                        'error': 'Profile not accessible',
                        'message': 'This profile requires authentication or has restricted access settings'
                    }
                elif "not found" in error_msg:
                    return {
                        'error': 'Profile not found',
                        'message': f'Could not find LinkedIn profile with ID: {profile_id}'
                    }
                elif "permission" in error_msg:
                    return {
                        'error': 'Access denied',
                        'message': 'You do not have permission to view this profile'
                    }
                elif "rate" in error_msg:
                    return {
                        'error': 'Rate limit exceeded',
                        'message': 'Too many requests. Please try again later.'
                    }
                else:
                    return {
                        'error': 'Failed to fetch profile data',
                        'message': str(e)
                    }
                
        except ValueError as e:
            # Re-raise initialization errors
            return {
                'error': 'Authentication error',
                'message': str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching profile data for {profile_id}: {str(e)}")
            return {
                'error': 'Unexpected error',
                'message': str(e)
            }
    
    def get_profile_connections(self, profile_id: str) -> Dict[str, Any]:
        """
        Get 1st-degree connections of a profile (only works for authenticated user's own profile)
        
        Args:
            profile_id: LinkedIn profile ID
            
        Returns:
            Dictionary containing connections or error information
        """
        try:
            self._initialize_api()
            logger.info(f"Fetching connections for: {profile_id}")
            
            try:
                connections = self._api.get_profile_connections(profile_id)
                logger.info(f"Successfully retrieved {len(connections) if connections else 0} connections for {profile_id}")
                return {'connections': connections}
            except Exception as e:
                error_msg = str(e).lower()
                if "can't be accessed" in error_msg:
                    return {
                        'error': 'Connections not accessible',
                        'message': 'This profile\'s connections are private or require authentication'
                    }
                else:
                    return {
                        'error': 'Failed to fetch connections',
                        'message': str(e)
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching connections for {profile_id}: {str(e)}")
            return {
                'error': 'Failed to fetch connections',
                'message': str(e)
            }
    
    def search_people(self, keywords: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for people on LinkedIn
        
        Args:
            keywords: Search keywords
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results or error information
        """
        try:
            self._initialize_api()
            logger.info(f"Searching for people with keywords: {keywords}")
            
            try:
                results = self._api.search_people(
                    keywords=keywords,
                    limit=limit
                )
                logger.info(f"Found {len(results) if results else 0} results for search: {keywords}")
                return {'results': results}
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg:
                    return {
                        'error': 'Rate limit exceeded',
                        'message': 'Too many search requests. Please try again later.'
                    }
                else:
                    return {
                        'error': 'Search failed',
                        'message': str(e)
                    }
                    
        except Exception as e:
            logger.error(f"Error searching people with keywords {keywords}: {str(e)}")
            return {
                'error': 'Search failed',
                'message': str(e)
            } 
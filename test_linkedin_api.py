import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_linkedin_api():
    """Test the LinkedIn API endpoint with different URL formats"""
    url = "http://localhost:5000/api/fetch-linkedin-profile"
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Test cases with different URL formats and field names
    test_cases = [
        {
            'profile_url': 'https://www.linkedin.com/in/ntoz'
        },
        {
            'linkedinUrl': 'linkedin.com/in/ntoz'
        },
        {
            'linkedin_url': 'www.linkedin.com/in/ntoz'
        },
        {
            'profile_url': 'http://www.linkedin.com/in/ntoz'
        },
        # Test with trailing slash
        {
            'linkedin_url': 'https://www.linkedin.com/in/ntoz/'
        },
        # Test with query parameters
        {
            'profile_url': 'https://www.linkedin.com/in/ntoz?originalSubdomain=uk'
        }
    ]
    
    for i, payload in enumerate(test_cases, 1):
        try:
            logger.info(f"\nTest case {i}: {json.dumps(payload, indent=2)}")
            response = requests.post(url, headers=headers, json=payload)
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            try:
                response_data = response.json()
                logger.info(f"Response data: {json.dumps(response_data, indent=2)}")
            except json.JSONDecodeError:
                logger.error(f"Raw response text: {response.text}")
                
        except Exception as e:
            logger.error(f"Error during request: {str(e)}")
            
        logger.info("-" * 80)

if __name__ == "__main__":
    test_linkedin_api() 
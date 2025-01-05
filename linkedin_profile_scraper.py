from bs4 import BeautifulSoup
import aiohttp
import asyncio
import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_profile_info(profile_url):
    """Fetch personal profile information from LinkedIn public page"""
    try:
        # Enhanced headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"'
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(profile_url, headers=headers) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status == 200:
                    html = await response.text()
                    logger.info(f"Retrieved HTML content length: {len(html)}")
                    
                    if 'Join to view full profiles for free' in html:
                        logger.error("LinkedIn requiring authentication")
                        return {
                            'error': 'LinkedIn authentication required',
                            'message': 'Please log in to LinkedIn first or provide an authenticated session'
                        }
                    
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Extract profile information with detailed logging
                    profile_data = {}
                    
                    # Name extraction
                    name = extract_name(soup)
                    if name:
                        profile_data['name'] = name
                        logger.info(f"Extracted name: {name}")
                    
                    # Headline extraction
                    headline = extract_headline(soup)
                    if headline:
                        profile_data['headline'] = headline
                        logger.info(f"Extracted headline: {headline}")
                    
                    # About extraction
                    about = extract_about(soup)
                    if about:
                        profile_data['about'] = about
                        logger.info("Successfully extracted about section")
                    
                    # Experience extraction
                    experience = extract_experience(soup)
                    if experience:
                        profile_data['experience'] = experience
                        logger.info(f"Extracted {len(experience)} experience items")
                    
                    # Education extraction
                    education = extract_education(soup)
                    if education:
                        profile_data['education'] = education
                        logger.info(f"Extracted {len(education)} education items")
                    
                    # Skills extraction
                    skills = extract_skills(soup)
                    if skills:
                        profile_data['skills'] = skills
                        logger.info(f"Extracted {len(skills)} skills")
                    
                    if not any(profile_data.values()):
                        logger.error("No profile data could be extracted")
                        return None
                    
                    return profile_data
                
                elif response.status == 999:
                    logger.error("LinkedIn's anti-bot mechanism triggered")
                    return {
                        'error': 'LinkedIn security check triggered',
                        'message': 'Please try again later or use an authenticated session'
                    }
                elif response.status == 401:
                    logger.error("Authentication required")
                    return {
                        'error': 'Authentication required',
                        'message': 'Please provide valid LinkedIn credentials'
                    }
                else:
                    logger.error(f"Unexpected status code: {response.status}")
                    return None
                    
    except aiohttp.ClientError as e:
        logger.error(f"Network error: {str(e)}")
        return {
            'error': 'Network error',
            'message': str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'error': 'Unexpected error',
            'message': str(e)
        }

def extract_name(soup):
    """Extract name from profile"""
    name_tag = soup.find('h1', {'class': re.compile(r'text-heading-xlarge.*')})
    return name_tag.get_text().strip() if name_tag else None

def extract_headline(soup):
    """Extract headline/title from profile"""
    headline_tag = soup.find('div', {'class': re.compile(r'text-body-medium.*')})
    return headline_tag.get_text().strip() if headline_tag else None

def extract_about(soup):
    """Extract about section"""
    about_section = soup.find('div', {'id': re.compile(r'about.*')})
    if about_section:
        about_text = about_section.find('div', {'class': re.compile(r'inline-show-more-text.*')})
        return about_text.get_text().strip() if about_text else None
    return None

def extract_experience(soup):
    """Extract work experience"""
    experience_section = soup.find('div', {'id': re.compile(r'experience.*')})
    experiences = []
    if experience_section:
        experience_items = experience_section.find_all('li', {'class': re.compile(r'artdeco-list__item.*')})
        for item in experience_items:
            title = item.find('span', {'class': re.compile(r'mr1.*')})
            company = item.find('span', {'class': re.compile(r't-14.*company-name')})
            date_range = item.find('span', {'class': re.compile(r't-14.*date-range')})
            
            if title and company:
                experiences.append({
                    'title': title.get_text().strip(),
                    'company': company.get_text().strip(),
                    'date_range': date_range.get_text().strip() if date_range else None
                })
    return experiences

def extract_education(soup):
    """Extract education information"""
    education_section = soup.find('div', {'id': re.compile(r'education.*')})
    education = []
    if education_section:
        education_items = education_section.find_all('li', {'class': re.compile(r'artdeco-list__item.*')})
        for item in education_items:
            school = item.find('h3', {'class': re.compile(r't-16.*')})
            degree = item.find('span', {'class': re.compile(r't-14.*degree-name')})
            
            if school:
                education.append({
                    'school': school.get_text().strip(),
                    'degree': degree.get_text().strip() if degree else None
                })
    return education

def extract_skills(soup):
    """Extract skills"""
    skills_section = soup.find('div', {'id': re.compile(r'skills.*')})
    skills = []
    if skills_section:
        skill_items = skills_section.find_all('span', {'class': re.compile(r'skill-name')})
        skills = [skill.get_text().strip() for skill in skill_items]
    return skills 
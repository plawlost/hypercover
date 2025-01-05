from bs4 import BeautifulSoup
import aiohttp
import asyncio
import json
import re

async def get_company_info_from_linkedin(company_name):
    """Fetch company information from LinkedIn public page"""
    try:
        # Format company name for URL
        formatted_name = company_name.lower().replace(' ', '-')
        url = f"https://www.linkedin.com/company/{formatted_name}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Extract company info from public page
                    description = soup.find('section', {'class': 'description'})
                    website = soup.find('a', {'class': 'website'})
                    
                    return (
                        description.get_text().strip() if description else "No description available",
                        website.get('href') if website else None
                    )
                return None, None
    except Exception as e:
        print(f"Error fetching LinkedIn data: {str(e)}")
        return None, None

async def scrape_website_for_paragraphs(website_url):
    """Scrape text content from company website"""
    if not website_url:
        return 'No website content available'
        
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(website_url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Extract meaningful text content
                    paragraphs = soup.find_all(['p', 'article', 'section'])
                    text_data = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                    
                    return '\n'.join(text_data) if text_data else 'No content found'
                return 'Failed to retrieve website content'
    except Exception as e:
        print(f"Error scraping website: {str(e)}")
        return 'Error retrieving website content'

def save_data_to_file(company_description, company_website, website_text):
    # Save the data to a JSON file
    data = {
        'company_description': company_description,
        'company_website': company_website,
        'website_text': website_text
    }
    
    with open('company_data.json', 'w') as file:
        json.dump(data, file, indent=4)
    
    print("Data saved to company_data.json")

async def getCompany(company_name):
    """Get company information from LinkedIn and website"""
    # Get company info from LinkedIn
    company_description, company_website = await get_company_info_from_linkedin(company_name)
    
    if company_description and company_website:
        print(f"Company Description: {company_description}")
        print(f"Company Website: {company_website}")
        
        # Scrape the company website for paragraphs
        website_text = await scrape_website_for_paragraphs(company_website)
        
        return {
            "name": company_name,
            "description": company_description,
            "website": company_website,
            "website_data": website_text
        }
    else:
        print("Company not found or invalid LinkedIn data")
        return None



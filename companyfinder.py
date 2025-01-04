
import requests
from bs4 import BeautifulSoup
import json

def scrape_website_for_info(company_name):
    search_url = f"https://www.google.com/search?q={company_name}+company+about"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract snippet from search results
        snippets = soup.find_all('div', {'class': ['VwiC3b', 'yXK7lf', 'MUxGbd', 'yDYNvb', 'lyLwlc']})
        description = ' '.join([s.get_text() for s in snippets[:2]])
        
        return {
            "name": company_name,
            "description": description or f"Information about {company_name}",
            "website_data": description
        }
    except Exception as e:
        return {
            "name": company_name,
            "description": f"Information about {company_name}",
            "website_data": ""
        }

def getCompany(company_name):
    return scrape_website_for_info(company_name)

from linkedin_api import Linkedin
import requests
from bs4 import BeautifulSoup
import json

def get_company_info_from_linkedin(company_name):
    # Authenticate with LinkedIn API
    api = Linkedin("email", "password")  # Use your LinkedIn credentials

    # Search for the company by name
    search_results = api.search_companies(company_name)

    if search_results:
        
        company_id = search_results[0]['urn_id']
        company_info = api.get_company(company_id)
        company_description = company_info.get('description', 'No description available')
        company_website = company_info.get('companyPageUrl', 'No website URL available')

        return company_description, company_website
    else:
        return None, None

def scrape_website_for_paragraphs(website_url):
    # Send a request to the company website
    response = requests.get(website_url)
    print(response)
    if response.status_code == 200:
        # Parse the website content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraphs from the webpage
        paragraphs = soup.find_all('p')
        text_data = [p.get_text() for p in paragraphs]
        
        return '\n'.join(text_data)
    else:
        return 'Failed to retrieve the website content'

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

def getCompany(company_name):# Take user input for the company name

    # Get company info from LinkedIn
    company_description, company_website = get_company_info_from_linkedin(company_name)
    
    if company_description and company_website:
        print(f"Company Description: {company_description}")
        print(f"Company Website: {company_website}")
        
        # Scrape the company website for paragraphs
        website_text = scrape_website_for_paragraphs(company_website)
        
        return {"name": company_name, "description": company_description, "website": company_website, "website_data": website_text}
    else:
        print("Company not found or invalid LinkedIn data")



import asyncio
import pandas as pd
from typing import List, Dict, Optional
from groq import AsyncGroq
from pathlib import Path
import aiofiles
from docx import Document
import pypandoc
from io import BytesIO
import zipfile
import json
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import os
from templates.cover_letter_templates import get_template_prompt, get_template_info
from redis import asyncio as aioredis
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import aiodns
from charset_normalizer import detect
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Histogram

# Performance metrics
GENERATION_TIME = Histogram('cover_letter_generation_seconds', 'Time spent generating cover letters')
CACHE_HITS = Counter('cache_hits_total', 'Total number of cache hits')
API_ERRORS = Counter('api_errors_total', 'Total number of API errors')

class BulkCoverLetterGenerator:
    def __init__(self, groq_api_key: str):
        self.groq_client = AsyncGroq(api_key=groq_api_key)
        self.output_dir = Path("generated_letters")
        self.output_dir.mkdir(exist_ok=True)
        self.company_cache = {}
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)  # Optimize for CPU cores
        self.dns_resolver = None
        self.redis_client = None
        self.template_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Configure optimal chunk sizes based on testing
        self.COMPANY_BATCH_SIZE = 25  # Optimal batch size for company info fetching
        self.LETTER_BATCH_SIZE = 20   # Optimal batch size for letter generation
        self.MAX_CONCURRENT_REQUESTS = 50  # Maximum concurrent API requests
        
    async def initialize(self):
        """Initialize async resources with optimized settings"""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Optimize HTTP session
        timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_connect=5, sock_read=10)
        connector = aiohttp.TCPConnector(
            limit=self.MAX_CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False,
            keepalive_timeout=60
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        # Initialize DNS resolver
        self.dns_resolver = aiodns.DNSResolver()
        
        # Try to initialize Redis, fall back to None if not available
        self.redis_client = None
        try:
            redis = await aioredis.Redis.from_url(
                'redis://localhost',
                max_connections=20,
                socket_timeout=1
            )
            # Validate connection with a ping
            if await redis.ping():
                self.redis_client = redis
                self.logger.info("Redis connection established and validated")
            else:
                self.logger.warning("Redis connection failed ping check")
        except Exception as e:
            self.logger.warning(f"Redis not available, falling back to in-memory cache: {str(e)}")
        
        # Warm up template cache
        await self._preload_templates()
        
    async def _is_redis_available(self) -> bool:
        """Check if Redis is available and connected"""
        if not self.redis_client:
            return False
        try:
            return await self.redis_client.ping()
        except Exception:
            return False

    async def _safe_redis_operation(self, operation):
        """Safely execute a Redis operation with connection check"""
        if not await self._is_redis_available():
            return None
        try:
            return await operation()
        except Exception as e:
            self.logger.warning(f"Redis operation failed: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch_with_retry(self, url: str) -> str:
        """Fetch URL with exponential backoff retry"""
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            API_ERRORS.inc()
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise

    async def _preload_templates(self):
        """Preload and cache templates for faster access"""
        templates = get_template_info()
        for template_id in templates.keys():
            template = get_template_prompt(template_id)
            self.template_cache[template_id] = template
            
            if self.redis_client:
                await self._safe_redis_operation(
                    lambda: self.redis_client.set(
                        f"template:{template_id}",
                        json.dumps(template)
                    )
                )

    @GENERATION_TIME.time()
    async def process_spreadsheet(self, spreadsheet_path: str, user_linkedin_profile: dict, template_id: str, progress_callback=None) -> str:
        """Process entire spreadsheet with optimized parallel processing"""
        try:
            df = pd.read_csv(spreadsheet_path)
            total_tasks = len(df)
            completed_tasks = 0
            
            # Pre-fetch and cache company data in optimal batches
            company_names = df['company_name'].unique()
            company_batches = [
                company_names[i:i + self.COMPANY_BATCH_SIZE] 
                for i in range(0, len(company_names), self.COMPANY_BATCH_SIZE)
            ]
            
            # Use semaphore to control concurrent API requests
            sem = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
            
            async def fetch_with_semaphore(name):
                async with sem:
                    return await self.gather_company_info(name)
            
            # Fetch company data with controlled concurrency
            for batch in company_batches:
                await asyncio.gather(*[
                    fetch_with_semaphore(name) for name in batch
                ], return_exceptions=True)
            
            # Process cover letters in optimized batches
            results = []
            for i in range(0, total_tasks, self.LETTER_BATCH_SIZE):
                batch_df = df.iloc[i:i + self.LETTER_BATCH_SIZE]
                batch_tasks = []
                
                for _, row in batch_df.iterrows():
                    task = self.generate_single_letter(
                        company_name=row['company_name'],
                        position=row['position'],
                        user_profile=user_linkedin_profile,
                        template_id=template_id,
                        notes=row.get('notes', '')
                    )
                    batch_tasks.append(task)
                
                # Process batch with error handling
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results and update progress
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Error in letter generation: {str(result)}")
                        API_ERRORS.inc()
                    else:
                        results.append(result)
                
                completed_tasks += len(batch_results)
                if progress_callback:
                    progress = (completed_tasks / total_tasks) * 100
                    await progress_callback(progress)
            
            return await self._package_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in bulk processing: {str(e)}")
            raise

    async def _async_remove(self, path: str):
        """Asynchronously remove a file"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, os.remove, path
            )
        except:
            pass

    @lru_cache(maxsize=1000)
    async def gather_company_info(self, company_name: str) -> Dict:
        """Gather and cache company information with optimized parallel fetching"""
        # Check Redis cache first
        if self.redis_client:
            cached_data = await self.redis_client.get(f"company:{company_name}")
            if cached_data:
                return json.loads(cached_data)
        
        if company_name in self.company_cache:
            return self.company_cache[company_name]
            
        try:
            # Fetch all data in parallel with timeouts
            tasks = [
                asyncio.create_task(self._fetch_linkedin_data(company_name)),
                asyncio.create_task(self._fetch_website_data(company_name)),
                asyncio.create_task(self._fetch_glassdoor_data(company_name)),
                asyncio.create_task(self._fetch_news_data(company_name))
            ]
            
            # Wait for all tasks with a timeout
            done, pending = await asyncio.wait(
                tasks,
                timeout=5,  # Aggressive timeout
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Get results, defaulting to empty data for failed tasks
            results = []
            for task in done:
                try:
                    results.append(task.result())
                except:
                    results.append({})
            
            linkedin_data, website_data, glassdoor_data, news_data = (
                results + [{}] * (4 - len(results))
            )[:4]
            
            # Process data in parallel
            keywords_task = asyncio.create_task(self._extract_keywords(website_data))
            culture_task = asyncio.create_task(self._analyze_company_culture(
                linkedin_data.get("description", ""),
                glassdoor_data.get("culture", ""),
                website_data
            ))
            tech_stack_task = asyncio.create_task(self._extract_tech_stack(
                website_data,
                linkedin_data.get("description", "")
            ))
            
            # Wait for processing with timeout
            website_keywords, culture_analysis, tech_stack = await asyncio.gather(
                keywords_task, culture_task, tech_stack_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(website_keywords, Exception): website_keywords = []
            if isinstance(culture_analysis, Exception): culture_analysis = {}
            if isinstance(tech_stack, Exception): tech_stack = []
            
            company_data = {
                "name": company_name,
                "description": linkedin_data.get("description", ""),
                "industry": linkedin_data.get("industry", ""),
                "website_data": website_data,
                "culture": {
                    "values": culture_analysis.get("values", []),
                    "work_environment": culture_analysis.get("environment", ""),
                    "benefits": glassdoor_data.get("benefits", [])
                },
                "recent_news": news_data.get("articles", [])[:3],
                "keywords": website_keywords,
                "technologies": tech_stack
            }
            
            # Cache the results
            self.company_cache[company_name] = company_data
            if self.redis_client:
                await self.redis_client.set(
                    f"company:{company_name}",
                    json.dumps(company_data),
                    expire=3600  # Cache for 1 hour
                )
            
            return company_data
            
        except Exception as e:
            return {"name": company_name, "description": f"Company information for {company_name}"}

    async def _fetch_linkedin_data(self, company_name: str) -> Dict:
        """Fetch LinkedIn data with optimized retry logic and parsing"""
        retry_count = 2  # Reduced retries for speed
        for attempt in range(retry_count):
            try:
                url = f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '-')}"
                
                # Resolve DNS first
                try:
                    result = await self.dns_resolver.query(url.split('/')[2], 'A')
                    ip = result[0].host
                except:
                    ip = None
                
                async with self.session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=3),  # Aggressive timeout
                    ssl=False,  # Skip SSL verification
                    headers={'Host': url.split('/')[2]} if ip else {}
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        return await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self._parse_linkedin_data,
                            BeautifulSoup(html, 'lxml')  # Using lxml for faster parsing
                        )
            except asyncio.TimeoutError:
                if attempt == retry_count - 1:
                    return {}
                await asyncio.sleep(0.1)
        return {}

    def _parse_linkedin_data(self, soup: BeautifulSoup) -> Dict:
        """Parse LinkedIn HTML with optimized selectors"""
        try:
            description = soup.find('section', {'class': 'description'})
            industry = soup.find('div', {'class': 'industry'})
            
            return {
                "description": description.get_text() if description else "",
                "industry": industry.get_text() if industry else ""
            }
        except:
            return {}

    async def _fetch_website_data(self, company_name: str) -> str:
        """Fetch website data with optimized timeout and error handling"""
        try:
            website = f"https://www.{company_name.lower().replace(' ', '')}.com"
            
            # Try to resolve DNS first
            try:
                result = await self.dns_resolver.query(website.split('/')[2], 'A')
                ip = result[0].host
            except:
                ip = None
            
            async with self.session.get(
                website,
                timeout=aiohttp.ClientTimeout(total=3),
                ssl=False,
                headers={'Host': website.split('/')[2]} if ip else {}
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    return await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._parse_website_data,
                        html
                    )
        except:
            return ""
        return ""

    def _parse_website_data(self, html: str) -> str:
        """Parse website HTML efficiently"""
        try:
            soup = BeautifulSoup(html, 'lxml')
            paragraphs = soup.find_all('p')
            return "\n".join(p.get_text() for p in paragraphs)
        except:
            return ""
    
    async def _fetch_glassdoor_data(self, company_name: str) -> Dict:
        """Fetch company data from Glassdoor"""
        try:
            # Note: This is a mock implementation. In production, you'd use Glassdoor's API
            async with self.session.get(
                f"https://api.glassdoor.com/api/v1/companies",
                params={"name": company_name},
                headers={"Authorization": os.getenv("GLASSDOOR_API_KEY")},
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "culture": data.get("culture", ""),
                        "benefits": data.get("benefits", []),
                        "ratings": data.get("ratings", {})
                    }
        except:
            return {}
        return {}
        
    async def _fetch_news_data(self, company_name: str) -> Dict:
        """Fetch recent news about the company"""
        try:
            # Note: This is a mock implementation. In production, you'd use a news API
            async with self.session.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": company_name,
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": os.getenv("NEWS_API_KEY")
                },
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "articles": [
                            {
                                "title": article["title"],
                                "summary": article["description"],
                                "date": article["publishedAt"]
                            }
                            for article in data.get("articles", [])
                        ]
                    }
        except:
            return {"articles": []}
        return {"articles": []}
        
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text using AI"""
        try:
            response = await self.groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"""Extract 5-10 relevant keywords from this text that would be useful for a cover letter:
                    {text[:1000]}
                    
                    Return only the keywords, separated by commas."""
                }],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=100
            )
            
            keywords = response.choices[0].message.content.split(",")
            return [k.strip() for k in keywords if k.strip()]
        except:
            return []
            
    async def _analyze_company_culture(self, linkedin_desc: str, glassdoor_culture: str, website_content: str) -> Dict:
        """Analyze company culture from various sources"""
        try:
            combined_text = f"""
            LinkedIn: {linkedin_desc}
            Glassdoor: {glassdoor_culture}
            Website: {website_content[:500]}
            """
            
            response = await self.groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this company's culture and values based on the following sources:
                    {combined_text}
                    
                    Return a JSON object with:
                    1. A list of 3-5 core company values
                    2. A brief description of the work environment
                    Format: {{"values": [], "environment": ""}}"""
                }],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"values": [], "environment": ""}
            
    async def _extract_tech_stack(self, website_content: str, company_desc: str) -> List[str]:
        """Extract technology stack mentioned in company materials"""
        try:
            combined_text = f"""
            Description: {company_desc}
            Website: {website_content[:500]}
            """
            
            response = await self.groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"""Extract the technology stack and tools mentioned in this text:
                    {combined_text}
                    
                    Return only the technology names, separated by commas."""
                }],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=100
            )
            
            tech_stack = response.choices[0].message.content.split(",")
            return [tech.strip() for tech in tech_stack if tech.strip()]
        except:
            return []
    
    async def generate_single_letter(self, company_name: str, position: str, user_profile: dict, template_id: str, notes: str = "") -> Dict:
        """Generate a single cover letter with optimized processing"""
        try:
            # Generate cache key
            cache_key = f"letter:{company_name}:{position}:{template_id}:{hash(json.dumps(user_profile))}"
            
            # Check Redis cache
            if self.redis_client:
                cached_letter = await self.redis_client.get(cache_key)
                if cached_letter:
                    cached_data = json.loads(cached_letter)
                    return {
                        'docx_path': cached_data['docx_path'],
                        'pdf_path': cached_data['pdf_path']
                    }
            
            # Gather company info and generate content in parallel
            company_info_task = asyncio.create_task(self.gather_company_info(company_name))
            template_task = asyncio.create_task(self._get_template(template_id))
            
            company_info, template = await asyncio.gather(company_info_task, template_task)
            
            # Generate content with optimized prompt
            content = await self._generate_content(
                company_info=company_info,
                position=position,
                user_profile=user_profile,
                template=template,
                notes=notes
            )
            
            # Generate documents in parallel
            docx_task = asyncio.create_task(self._create_docx(content, company_name, position))
            pdf_task = asyncio.create_task(self._create_pdf(content, company_name, position))
            
            docx_path, pdf_path = await asyncio.gather(docx_task, pdf_task)
            
            result = {
                'docx_path': docx_path,
                'pdf_path': pdf_path
            }
            
            # Cache the result
            if self.redis_client:
                await self.redis_client.set(
                    cache_key,
                    json.dumps(result),
                    expire=3600  # Cache for 1 hour
                )
            
            return result
            
        except Exception as e:
            raise Exception(f"Error generating letter for {company_name}: {str(e)}")

    async def _get_template(self, template_id: str) -> Dict:
        """Get template with caching"""
        # First try memory cache
        if template_id in self.template_cache:
            return self.template_cache[template_id]
            
        # Then try Redis if available
        if self.redis_client:
            cached_template = await self._safe_redis_operation(
                lambda: self.redis_client.get(f"template:{template_id}")
            )
            if cached_template:
                template = json.loads(cached_template)
                self.template_cache[template_id] = template
                return template
        
        # Fall back to fetching fresh
        template = get_template_prompt(template_id)
        self.template_cache[template_id] = template
        return template

    async def _generate_content(self, company_info: Dict, position: str, user_profile: Dict, template: Dict, notes: str) -> str:
        """Generate cover letter content with optimized prompt"""
        # Prepare optimized prompt with only essential information
        prompt = template.get("prompt_template", "")
        
        # Extract only needed company info to reduce token usage
        company_context = {
            "name": company_info["name"],
            "description": company_info["description"][:500],  # Limit length
            "culture": {
                "values": company_info["culture"]["values"][:3],  # Limit to top values
                "work_environment": company_info["culture"]["work_environment"][:200]
            }
        }
        
        # Extract only relevant user profile info
        user_context = {
            "name": user_profile["name"],
            "current_role": user_profile["current_role"],
            "key_skills": user_profile["skills"][:5],  # Limit to top skills
            "experience": user_profile.get("experience", [])[:2]  # Limit to recent experience
        }
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Using Mixtral for faster inference
                messages=[{
                    "role": "system",
                    "content": "You are an expert cover letter writer. Generate a concise, impactful cover letter."
                }, {
                    "role": "user",
                    "content": f"Generate a cover letter for {position} at {company_info['name']} using this template style: {prompt}\n\nCompany Info: {json.dumps(company_context)}\n\nCandidate Profile: {json.dumps(user_context)}\n\nAdditional Notes: {notes}"
                }],
                temperature=0.7,
                max_tokens=1000,  # Limit response length
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")

    async def _create_docx(self, content: str, company_name: str, position: str) -> str:
        """Create DOCX file efficiently"""
        doc = Document()
        
        # Optimize document creation
        def create_doc():
            # Add content with minimal formatting
            doc.add_paragraph(content)
            
            # Save with a unique filename
            filename = f"{self.output_dir}/CL_{company_name}_{position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            doc.save(filename)
            return filename
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            create_doc
        )

    async def _create_pdf(self, content: str, company_name: str, position: str) -> str:
        """Create PDF file efficiently using pandoc"""
        try:
            # Generate unique filename
            filename = f"{self.output_dir}/CL_{company_name}_{position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            def convert_to_pdf():
                pypandoc.convert_text(
                    content,
                    'pdf',
                    format='markdown',
                    outputfile=filename,
                    extra_args=[
                        '--pdf-engine=xelatex',
                        '-V', 'geometry:margin=1in'
                    ]
                )
                return filename
            
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                convert_to_pdf
            )
            
        except Exception as e:
            raise Exception(f"Error creating PDF: {str(e)}")
    
    async def generate_preview(self, template_settings: dict, format_type: str = 'doc') -> str:
        """Generate a preview of the cover letter with the given template settings"""
        # Use a sample company and position for the preview
        sample_data = {
            "company_name": "Example Corp",
            "position": "Software Engineer",
            "notes": "Sample preview for customization"
        }
        
        # Gather company info for the sample
        company_data = await self.gather_company_info(sample_data["company_name"])
        
        # Generate content using the template settings
        content = await self.generate_ai_content(
            company_data=company_data,
            position=sample_data["position"],
            user_profile={
                "name": "John Doe",
                "current_role": "Senior Developer",
                "experience": [
                    {"title": "Lead Developer", "company": "Tech Co", "duration": "3 years"},
                    {"title": "Software Engineer", "company": "Startup Inc", "duration": "2 years"}
                ],
                "skills": ["Python", "JavaScript", "React", "Node.js", "AWS"]
            },
            template_id=template_settings["baseTemplate"],
            template_customization=template_settings
        )
        
        if format_type == 'pdf':
            return await self.format_content_for_pdf(content)
        else:
            return self.format_content_for_doc(content)
            
    def format_content_for_doc(self, content: str) -> str:
        """Format the content for DOC preview"""
        # Convert newlines to <br> tags and wrap paragraphs
        paragraphs = content.split('\n\n')
        formatted_paragraphs = []
        for p in paragraphs:
            if p.strip():
                # Replace newlines with <br> before using in f-string
                p_with_br = p.replace('\n', '<br>')
                formatted_paragraphs.append(f'<p class="mb-4">{p_with_br}</p>')
        return '\n'.join(formatted_paragraphs)
        
    async def format_content_for_pdf(self, content: str) -> str:
        """Format the content for PDF preview"""
        # Add any PDF-specific formatting
        html_template = '''
        <html>
        <head>
            <style>
                body { font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.5; }
                p { margin-bottom: 1em; }
            </style>
        </head>
        <body>
            {}
        </body>
        </html>
        '''
        formatted_content = content.replace('\n\n', '</p><p>').replace('\n', '<br>')
        return html_template.format(formatted_content)
        
    async def save_preview_pdf(self, content: str, output_path: str):
        """Save the preview content as a PDF file"""
        def create_pdf():
            import pypandoc
            pypandoc.convert_text(
                content,
                'pdf',
                format='html',
                outputfile=output_path,
                extra_args=['--pdf-engine=weasyprint']
            )
            
        await asyncio.get_event_loop().run_in_executor(self.executor, create_pdf) 
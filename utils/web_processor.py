
from urllib.parse import urlparse
import datetime
import requests
import xml.etree.ElementTree as ET
from utils.prompts import url_filter_prompt_template, key_word_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from dotenv import load_dotenv
load_dotenv()
import os

from playwright.async_api import async_playwright
import asyncio
import pathlib
from utils.mongo_manager import MongoDBmanager
from utils.logger import get_debug_logger

from concurrent.futures import ThreadPoolExecutor, as_completed

web_collection = MongoDBmanager("site_data")

url_logger = get_debug_logger(
    "web_agent", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs/web_bot.log")
)

# store the errored sites, so we dont need process them
ERRORED_SITES = []


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
from zoneinfo import ZoneInfo
TIME_ZONE = os.environ["TIME_ZONE"]

llm = ChatOpenAI(
                model='gpt-4.1-mini',
                api_key=OPENAI_API_KEY,
                max_tokens=2048,
                temperature=0,
            )

 # Use a browser-like User-Agent to reduce the chance of being blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/109.0.0.0 Safari/537.36'
}

# Create the URL filtering chain
url_filter_chain = url_filter_prompt_template | llm | JsonOutputParser()

keyword_chain = key_word_prompt | llm | StrOutputParser()


# Extract main site name from URL
def extract_site_name(sitemap_url):
    """
    Extract only the main domain name 
    """
    parsed_url = urlparse(sitemap_url)
    domain = parsed_url.netloc
    
    # Remove 'www.' prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Extract main domain name (remove TLD)
    domain_parts = domain.split('.')
    if len(domain_parts) >= 2:
        # Handle cases like example.co.uk, example.com, etc.
        main_domain = domain_parts[0]
    else:
        main_domain = domain
    
    return main_domain


# Function for individual sitemaps 
def fetch_individual_sitemap(url):
    """Function to fetch URLs from an individual sitemap"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        urls = []
        for url_element in root.findall('ns:url', namespace):
            loc = url_element.find('ns:loc', namespace)
            if loc is not None:
                urls.append(loc.text)
        
        return urls
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching individual sitemap {url}: {e}")
        return []
        
def process_sitemap(sitemap_element, namespace):
    loc = sitemap_element.find('ns:loc', namespace)
    if loc is not None:
        individual_sitemap_url = loc.text
        print(f"Processing individual sitemap: {individual_sitemap_url}")
        
        # Fetch URLs from this individual sitemap
        individual_urls = fetch_individual_sitemap(individual_sitemap_url)
        print(f"Extracted {len(individual_urls)} URLs from {individual_sitemap_url}")
        return individual_urls
    return []

def extract_sitemap_urls(sitemap_url):
    """
    Fetches the XML sitemap from the given URL, parses it, and returns a list of extracted URLs.
    Handles both sitemap index files and individual sitemap files.
    """

    try:
        # Fetch the main sitemap
        response = requests.get(sitemap_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse the XML
        root = ET.fromstring(response.text)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        all_urls = []
        
        #  Logic to handle sitemap index files 
        # Check if this is a sitemap index file (contains <sitemap> elements)
        sitemap_elements = root.findall('ns:sitemap', namespace)
        
        if sitemap_elements:
            # This is a sitemap index file
            print(f"Found sitemap index with {len(sitemap_elements)} individual sitemaps")
            
            with ThreadPoolExecutor(max_workers=16) as executor:   # adjust workers as needed
                futures = [executor.submit(process_sitemap, sitemap_element, namespace) 
                        for sitemap_element in sitemap_elements]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_urls.extend(result)
                    except Exception as e:
                        print(f"Error processing sitemap: {e}")
        
        else:
            # Regular sitemap file with direct URLs
            for url_element in root.findall('ns:url', namespace):
                loc = url_element.find('ns:loc', namespace)
                if loc is not None:
                    all_urls.append(loc.text)
           
        
        # Log extracted URLs
        
        return all_urls
   
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []

def save_urls_to_mongodb(org_id, extracted_urls, sitemap_url):
    """
    Save extracted URLs to MongoDB 
    """
    try:
                
        # Extract site name from sitemap URL
        site_name = extract_site_name(sitemap_url)
        
        # Prepare documents for insertion, for each web site we will have a separate document
        # for the following updates use the site_name as index
        documents = []
        for url in extracted_urls:
                
            ret = extract_text_from_url(url)
            
            result = keyword_chain.invoke({"text": ret['text_content']})
            keywords = [kw.strip() for kw in result.split(",")]
            
            document = {
                'org_id' : org_id,
                'site_name': site_name,
                'url': url,
                'title' : ret['title'],
                'key_words' : keywords,
                'text_content' : ret['text_content'],
                'extracted_date': datetime.datetime.now(ZoneInfo(TIME_ZONE)),
                'sitemap_source': sitemap_url
            }
            documents.append(document)
        
        # Insert urls in the site map xml
        if documents:
            result = web_collection.insert_documents(documents)
            print(f"Successfully inserted {result} URLs for {site_name} into MongoDB")
        
        else:
            print("No URLs to insert")
        
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

def filter_urls_with_llm(user_query, urls_data):
    """
    Filter URLs using LLM based on user query
    """
    try:
        # Use the chain to filter URLs
        filter_result = url_filter_chain.invoke({
            "user_input": user_query,
            "site_urls": urls_data
        })
        
        return filter_result
        
    except Exception as e:
        print(f"Error filtering URLs with LLM: {e}")
        return []

# from playwright.async_api import async_playwright
import datetime

async def _extract_text_async(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        page = await browser.new_page()
        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        try:
            await page.goto(url, timeout=30000, wait_until='domcontentloaded')
        except:
            try:
                await page.goto(url, timeout=20000, wait_until='load')
            except:
                await page.goto(url, timeout=15000)

        try:
            await page.wait_for_load_state('networkidle', timeout=5000)
        except:
            await page.wait_for_timeout(2000)

        await page.evaluate("""
            () => {
                document.querySelectorAll('script, style, button, select, option, nav, footer, ul.sub-menu, ul.dropdown-menu, div.menu-wrapper, div.gtranslate_wrapper').forEach(el => el.remove());
            }
        """)

        title = await page.title()
        text_content = await page.inner_text('body')
        await browser.close()

        lines = (line.strip() for line in text_content.splitlines())
        text = ' '.join(line for line in lines if line)

        return {
            'url': url,
            'title': title.strip() if title else '',
            'text_content': text[:5000],
        }


def extract_text_from_url(url):
    """Extract text content from URL using Playwright"""
    try:
        # Use ProactorEventLoop explicitly to support subprocess creation on Windows
        # regardless of the global event loop policy set by the main app.
        loop = asyncio.ProactorEventLoop()
        try:
            extracted_content = loop.run_until_complete(_extract_text_async(url))
        finally:
            loop.close()

        url_logger.info(f"Extracted content from: {url}")
        url_logger.info(f"Title: {extracted_content['title']}")
        url_logger.info(f"Content: {extracted_content['text_content']}")

        return extracted_content

    except Exception as e:
        url_logger.error(f"Failed to extract content from {url}: {str(e)}")
        return {
            'url': url,
            'error': str(e),
            'extracted_date': datetime.datetime.now(ZoneInfo(TIME_ZONE)),
            'status': 'failed'
        }

from concurrent.futures import ProcessPoolExecutor, as_completed

# def init_worker():
#     global web_collection
#     from utils.mongo_manager import MongoDBmanager
#     web_collection = MongoDBmanager("site_data")

def process_url(url):
    try:
        # if isinstance(url_info, dict) and 'url' in url_info:
        site_doc = web_collection.get_one_document({'url': url})
        if site_doc and 'text_content' in site_doc:
            return {
                'title': site_doc['title'],
                'text_content': site_doc['text_content']
            }
        else:
            text_data = extract_text_from_url(url)
            # text_data = await extract_text_from_url(url)
            
            site_doc['title'] = text_data['title'],
            site_doc['text_content'] =  text_data['text_content']
            ret = site_doc.pop('_id')
            # update/insert into DB
            web_collection.update_one({'url': url}, site_doc)
            return {
                'title': text_data['title'],
                'text_content': text_data['text_content']
            }
    except Exception as e:
        print(f"Error: {e}")
        return {
                'error': url,
                'text_content': e
            }
    # return None

# def chunk_list(lst, n):
#     """Split list into n roughly equal chunks."""
#     k, m = divmod(len(lst), n)
#     return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

# def process_batch(user_query, batch):
#     # Call your existing filter function on a sublist
#     return filter_urls_with_llm(user_query, batch)
            
def url_filter_agent(org_id, user_query):
    """Main agent function to filter URLs and extract text"""
    
    url_logger.debug({
        "user_query": user_query
    })        

    # Get all URLs from database belongs to users organization
    all_urls = []
    sites = web_collection.get_documents({'org_id': org_id})
    # url_logger.debug(sites)
    for site in sites:
        if site['url'] in ERRORED_SITES:
            continue
        all_urls.append({"url" : site['url'],
                        "key_words" : site['key_words']})

    # url_logger.debug(f"All : {all_urls}")
    
    # Filter URLs using LLM, use batching to make faster
    # url_batches = chunk_list(all_urls, 4)
    filtered_urls = []
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(process_batch, user_query, batch) for batch in url_batches]
    #     for f in as_completed(futures):
    #         result = f.result()
    #         if result:
    #             filtered_urls.extend(result)
    
    filtered_urls = filter_urls_with_llm(user_query, all_urls)
    extracted_texts = []
    if filtered_urls:
    
        # Extract text from filtered URLs
        # with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:  # tune workers
        #     futures = [executor.submit(process_url, url_info) for url_info in filtered_urls]
        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result is None:
        #             continue
        #         if 'error' in result:
        #             if result['error'] not in ERRORED_SITES:
        #                 ERRORED_SITES.append(result['error'])
        #         else:
        #             extracted_texts.append(result)
        for f_url in filtered_urls:
            res = process_url(f_url)
            extracted_texts.append(res)

    url_logger.debug({
        "status": "success",
        "filtered_urls": filtered_urls,
        "extracted_content": extracted_texts
    })        
    # Return results in JSON format
    return {
        "status": "success",
        "filtered_urls": filtered_urls,
        "extracted_content": extracted_texts
    }
    
        
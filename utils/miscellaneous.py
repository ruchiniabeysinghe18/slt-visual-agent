import re
import requests
from xml.etree import ElementTree
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(
                model='gpt-4.1-mini',
                api_key=OPENAI_API_KEY,
                max_tokens=2048,
                temperature=0,
            )


prompt_template = PromptTemplate(
    template="""
Change the provided input into a human-readable form by removing list contents.
Replace underscored names (e.g., service_a) with proper names (e.g., Service A).

Input:
{input}

Output only the transformed text. Do not provide explanations or reasons.
""",
    input_variables=["input"]
)

chain = prompt_template | llm | StrOutputParser()
    
    
# utility functions
def remove_key_recursive(d, key_to_remove):
    if isinstance(d, dict):
        d.pop(key_to_remove, None)
        for value in d.values():
            remove_key_recursive(value, key_to_remove)
    elif isinstance(d, list):
        for item in d:
            remove_key_recursive(item, key_to_remove)

def is_phone_number(user_id: str) -> bool:
    """
    Check if the given userID is a phone number in the format:
    - Digits only
    - Starts with a country code (no +, no spaces, no dashes)
    - Length between 9 and 15 digits (E.164 standard without '+')
    """
    return user_id.isdigit() and 9 <= len(user_id) <= 15

def parse_services(s: str):
    s = s.strip()

    if s.startswith("[") and s.endswith("]"):
        # split by commas, remove spaces and stray quotes
        items = [re.sub(r"['\"]", "", x.strip()) for x in s.strip("[]").split(",")]
        return items if items != [''] else []
    else:
        return [re.sub(r"['\"]", "", s)]

"""
cases :

1. no previously started services, one detected service
2. no previously started services, two detected service
3. one previously started services incomplete, one new detected service
4. one previously started services incomplete, same detected service
5. one previously started services complete, one new detected service

"""
# def get_uncompleted(booking_session, detected_services):
    
#     # check uncompleted services
#     if booking_session and (len(detected_services) > 1):
#         completed_booking_ = {}
#         for service in detected_services:
#             completed_booking_[service] = False
            
#         for service in detected_services:
#             if (service in booking_session) and (booking_session[service]['status'] == 'complete'):
#                 completed_booking_[service] = True
        
#         return [ k for k, v in completed_booking_.items() if v == False]
#     else:
#         return detected_services
    
def get_uncompleted(booking_session, newly_detected_services):
    if (len(newly_detected_services) > 1):
        # multiple detected services
        if booking_session:
            # get existing uncompleted services
            existing_uncompleted_services = []
            for k,v in booking_session.items():
                if booking_session[k]['status'] == 'incomplete':
                    existing_uncompleted_services.append(k)
            
            # there are exititing booking sessions
            completed_booking = {}
            for service in newly_detected_services:
                completed_booking[service] = False
                
            for service in newly_detected_services:
                if (service in booking_session) and (booking_session[service]['status'] == 'complete'):
                    completed_booking[service] = True
            
            current_uncompleated_services =  [ k for k, v in completed_booking.items() if v == False]
            
            if len(current_uncompleated_services) > 1:
                existing_uncompleted = []
                for uncomp_service in current_uncompleated_services:
                    if uncomp_service in existing_uncompleted_services:
                        existing_uncompleted.append(uncomp_service)

                newly_detected_uncompleted_servoces = [x for x in current_uncompleated_services if x not in existing_uncompleted]

                reults = chain.invoke({
                    "input": f"there is an uncompleted {existing_uncompleted} in the conversation, do you wish to continue with it or start with {newly_detected_uncompleted_servoces}",
                })

                return reults
            elif len(current_uncompleated_services) == 0:
                return "You have already completed bookings for those services. To handle multiple new requests, would you like to connect with a consultant?"
            else:
                # single uncompleted service
                return current_uncompleated_services
            
        else:
            # no existing booking sessions
            return "I can only handle one service at a time. Would you like to go through them one by one, or connect with a consultant?"
        
    else:
        # single service
        return newly_detected_services
    
def is_sitemap(url: str) -> bool:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        if "xml" not in r.headers.get("Content-Type", ""):
            return False
        
        root = ElementTree.fromstring(r.content)
        return root.tag.endswith("urlset") or root.tag.endswith("sitemapindex")
    except Exception:
        return False


def get_consultant_conversation(chat_history):
    formatted_msgs = []

    for msg in chat_history:
        flag = getattr(msg, 'additional_kwargs', {}).get('flag')

        if flag == 'consultant':
            if isinstance(msg, HumanMessage):
                formatted_msgs.append(f"User: {msg.content}")
            else:  # AIMessage
                formatted_msgs.append(f"Consultant: {msg.content}")

    return formatted_msgs
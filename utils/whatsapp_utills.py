
import json
import os
import subprocess
from dotenv import load_dotenv
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Load the .env file
load_dotenv(dotenv_path)

WHAT_TOKEN = os.getenv("ACCESS_TOKEN")   
VERSION = os.getenv("VERSION")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID") # test number

import requests
import json

def send_message(response, received_phone_num, type='text'):
    url = f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages"
    print(f"[send_message] url={url} | to={received_phone_num} | type={type}")
    headers = {
        "Authorization": f"Bearer {WHAT_TOKEN}",
        "Content-Type": "application/json"
    }

    if type == 'text':
        payload = {
            "text": {
                "body": response
            }
        }

    elif type == 'document':
        payload = {
            "document": {
                "id": response['media_id'],
                "filename": response['file_name'],
                "caption": response['caption']
            }
        }

    elif type == 'template':
        # response can be a dict with 'name' and optional 'language_code',
        # or a plain string treated as the template name with default language.
        if isinstance(response, dict):
            template_name = response.get('name', 'hello_world')
            language_code = response.get('language_code', 'en_US')
        else:
            template_name = response
            language_code = 'en_US'
        payload = {
            "template": {
                "name": template_name,
                "language": {
                    "code": language_code
                }
            }
        }

    data = {
        "messaging_product": "whatsapp",
        "to": received_phone_num,
        "type": type,
        **payload
    }

    try:
        result = requests.post(url, headers=headers, json=data)
        result.raise_for_status()
        print(result.json())
        return result.json()

    except requests.exceptions.RequestException as e:
        print("Error occurred:", e)
        return -1

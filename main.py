
"""
Main app
"""

import asyncio
import os
import os
import uuid
import base64
import tempfile
import uvicorn

import sys
import os

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ["PYTHONIOENCODING"] = "utf-8"

USE_TTS = False
# USE_AVATAR = os.getenv("USE_AVATAR", "false").lower() == "true"

# Fix for Windows ProactorEventLoop spurious ConnectionResetError on client disconnect
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import pathlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from utils.mongo_manager import MongoDBmanager
from utils.logger import get_debug_logger
from utils.prompts import SERVICE_INFO
import json
from typing import List, Optional
import requests
from common_bot import common_chatbot, analyse_consultant_conversation

from fastapi import FastAPI,HTTPException,Request,UploadFile,Form,Depends,Response, BackgroundTasks, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils.whatsapp_utills import send_message
from utils.pdf_processor import DocumentKB, PDFProcessor
from utils.web_processor import save_urls_to_mongodb
from utils.miscellaneous import remove_key_recursive, is_sitemap, is_phone_number
from bson import ObjectId

import traceback
from zoneinfo import ZoneInfo
from utils.quotation import generate_quotation
from utils.speech_processor import transcribe_audio_bytes
from utils.tts_processor import synthesize_text

logger = get_debug_logger(
    "main", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "./logs/main.log")
)


google_api_key = os.getenv("GOOGLE_API_KEY")
KB = os.getenv("KB")
WHAT_TOKEN = os.getenv("ACCESS_TOKEN")
VERSION = os.getenv("VERSION")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID") #
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
VERIFY_TOKEN=os.getenv("VERIFY_TOKEN")
DOCUMENT_LIMIT=int(os.getenv("DOCUMENT_LIMIT"))
WEBSITE_LIMIT=int(os.getenv("WEBSITE_LIMIT"))
TIME_ZONE = os.environ["TIME_ZONE"]
   
user_state_collection = MongoDBmanager("user_states")
booking_collection = MongoDBmanager("user_bookings")
pdf_doc_collection = MongoDBmanager("pdf_documents")
web_collection = MongoDBmanager("site_data")
org_collection = MongoDBmanager("organization")
unread_counts_collection = MongoDBmanager("unread_counts")

document_kb = DocumentKB(pdf_doc_collection, google_api_key)

received_message_ids = set()

app = FastAPI()

os.makedirs("chat_uploads", exist_ok=True)
app.mount("/chat_uploads", StaticFiles(directory="chat_uploads"), name="chat_uploads")

os.makedirs("audio_tts", exist_ok=True)
app.mount("/audio_tts", StaticFiles(directory="audio_tts"), name="audio_tts")

# In-memory store: { user_id: {"status": "pending"|"completed"|"failed", "video_url": str|None} }
avatar_status_store: dict = {}

os.makedirs("avatar_videos", exist_ok=True)
app.mount("/avatar_videos", StaticFiles(directory="avatar_videos"), name="avatar_videos")

# Serve local avatar GLB models (from wawa-lipsync examples)
import pathlib as _pl
_MODELS_DIR = _pl.Path(__file__).parent.parent / "wawa-lipsync" / "examples" / "lipsync-demo" / "public" / "models"
if _MODELS_DIR.exists():
    app.mount("/models", StaticFiles(directory=str(_MODELS_DIR)), name="models")

@app.get("/admin")
def serve_admin():
    return FileResponse(str(_pl.Path(__file__).parent / "admin.html"))

@app.get("/app")
def serve_app():
    return FileResponse(str(_pl.Path(__file__).parent / "app.html"))

app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_credentials = True,
  allow_methods = ["*"],
  allow_headers = ["*"]
)


# compile the graph
chatbot_app = common_chatbot()


#To handle Unread message counts
def track_new_message(org_id: str, user_id: str):
    """Function to track new message count"""
    try:
        # Get current count
        existing = unread_counts_collection.get_one_document({"org_id": org_id, "user_id": user_id})
        
        if existing:
            new_count = existing["unread_count"] + 1
            unread_counts_collection.update_one(
                {"org_id": org_id, "user_id": user_id},
                {"unread_count": new_count, "last_updated": datetime.now().isoformat()}
            )
        else:
            new_count = 1
            unread_counts_collection.insert_one({
                "org_id": org_id,
                "user_id": user_id,
                "unread_count": new_count,
                "last_updated": datetime.now().isoformat()
            })
        
        logger.debug(f"New message tracked: {org_id}/{user_id} - count: {new_count}")
        return new_count
        
    except Exception as e:
        logger.error(f"Error tracking message: {str(e)}")
        return 0


@app.post("/select_knowledge_base/")
async def select_knowledge_base(request: Request):
    """
    User will select which knowledge base to use
    """
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = '001'
        kbs = payload.get("knowledge_bases")  # list with following ["doc", "web"]
        
        org_collection.update_one({'org_id': org_id}, {"selected_knowledge_bases" : kbs})
        return Response(content="knowledge base selected", status_code=200)
    
    except Exception as e:
        logger.debug(f"Error select_knowledge_base/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error selecting knowledge bases: {str(e)}")

def process_website(org_id, url):
    try:
        save_urls_to_mongodb(org_id, [url], url)
    except Exception as e:
        logger.debug(f"Error upload_website (background task)/ : \n {traceback.format_exc()}")

@app.post("/upload_website/")
async def upload_website(request: Request, background_tasks: BackgroundTasks):
    """
    Accept a single web URL and add it to the knowledge base
    """
    # try:
    payload = await request.json()
    org_id = payload.get("org_id")
    url = payload.get("url")

    if not url:
        return Response(content="No URL provided", status_code=422)

    existing_urls = web_collection.get_distinct("url", {"org_id": org_id})

    if len(existing_urls) >= WEBSITE_LIMIT:
        return Response(content=f"Maximum number of websites allowed is {WEBSITE_LIMIT}", status_code=422)

    if url in existing_urls:
        return Response(content=f"Website {url} is already in the knowledge base", status_code=422)

    background_tasks.add_task(process_website, org_id, url)

    return Response(content="Website is being processed and added to the knowledge base!", status_code=200)

    # except Exception as e:
    #     logger.debug(f"Error upload_website/ : \n {traceback.format_exc()}")
    #     raise HTTPException(status_code=500, detail={"status": "error", "message": f"Error uploading to DB: {str(e)}"})
    

@app.post("/upload_pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    org_id: str = Form(...)
):
    try:
    
        # org_id = '001'
        
        # Check and limit the number of documents to 5
        pdfs = pdf_doc_collection.get_documents({"org_id" : org_id})
        
        if len(pdfs) >= DOCUMENT_LIMIT:
            return Response(content=f"Maximum number of documents allowed is {DOCUMENT_LIMIT}", status_code=422)            
        
        existing_docs = []
        for pdf in pdfs:
            existing_docs.append(pdf['document_name'])
            
        if file.filename in existing_docs:
            return Response(content=f"file {file.filename} is already uploaded", status_code=422)
        
        # Validate file type - only PDFs allowed
        if not file.filename.lower().endswith('.pdf'):
            return Response(content=f"Maximum number of documents allowed is {DOCUMENT_LIMIT}", status_code=400)

        pdf_processor = PDFProcessor(upload_dir=org_id)
        
        logger.info(f"Processing PDF upload: {file.filename}")

        # Save uploaded PDF file
        file_path = pdf_processor.save_fastapi_pdf(file.file, file.filename)
        
        # Extract text from PDF
        pdf_text = pdf_processor.extract_text_from_pdf(file_path)
        
        if not pdf_text.strip():
            return Response(content="No text content found in the PDF document", status_code=400)

        metadata = {
            "filename": file.filename,
            "file_size": file.size,
            "file_type": "pdf",
        }

        result, collection = document_kb.store_pdf_document(
            org_id=org_id,
            pdf_text=pdf_text,
            pdf_name=file.filename,
            metadata=metadata
        )

        try:
            os.remove(file_path)
        except Exception:
            pass

        logger.info(f"Successfully processed PDF {file.filename}")

        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded and processed successfully",
            "document_id": str(result.inserted_id),
            "chunks_count": collection.count(),
            "text_length": len(pdf_text),
            "document_type": "pdf"
        }

    except Exception as e:
        logger.debug(f"Error upload_pdf/ {file.filename}: \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Error processing PDF document {file.filename}: {str(e)}"})


@app.post("/list_pdf", response_model=List[str])
async def list_pdf(request: Request):
    payload = await request.json()
    org_id = payload.get("org_id")
    # org_id = '001'
    ret = pdf_doc_collection.get_documents({"org_id" : org_id})
    pdf_list = []
    for doc in ret:
        pdf_list.append(doc['document_name'])
    return pdf_list

@app.post("/delete_pdf")
async def delete_pdf(request: Request):
    try:
        payload = await request.json()
        doc_name = payload.get("name")
        org_id = payload.get("org_id")
        # org_id = '001'
        ret = pdf_doc_collection.delete_one_document({'org_id' : org_id,'document_name' : doc_name})
        return Response(content="file deleted successfully!", status_code=200)
    except Exception as e:
        logger.debug(f"Error delete_pdf/: \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Error deleting PDF document: {str(e)}"})

from pydantic import BaseModel

class UserIDResponse(BaseModel):
    phone_number: str
    name: str
    consultant_mode: bool
    time_stamp: str 
    unread_count: int
    studio_style: str  # "bold" or "unbold"
    
@app.post("/list_user_ids", response_model=List[UserIDResponse])
async def list_user_ids(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = '001'
        
        docs = user_state_collection.get_documents({"org_id" : org_id})
        
        # Find the most recent test conversation user_id (same logic as get_test_conversation)
        recent_test_user_id = None
        sorted_docs = user_state_collection.get_documents({"org_id": org_id}, sort=[("state.time_stamp", -1)])
        if sorted_docs:
            for doc in sorted_docs:
                if not is_phone_number(str(doc["user_id"])):
                    recent_test_user_id = doc["user_id"]
                    break
        
        phone_numbers = []
        # TODO: limited temporarly to phon numbers later we need to use web chats also
        for doc in docs:
            if is_phone_number(str(doc['user_id'])):
                # Phone number users always have 0 unread count (WhatsApp handles its own notifications)
                try:
                    phone_numbers.append(
                        UserIDResponse(
                            phone_number=doc['user_id'],
                            name=doc['user_name'],
                            consultant_mode=doc['state']['consultant_mode'],
                            time_stamp=doc['state']['time_stamp'],
                            unread_count=0,  # Always 0 for phone number users
                            studio_style="unbold",  # set as default, It is not needed for whatsapp users
                        )
                    )
                except KeyError:
                    continue
            else:
                # Non-phone number users (web/test portal) - get actual unread count
                unread_doc = unread_counts_collection.get_one_document({"org_id": org_id, "user_id": doc['user_id']})
                unread_count = unread_doc["unread_count"] if unread_doc else 0
                
                # Determine studio_style: bold if this is the most recent test user AND has unread messages
                studio_style = "bold" if (doc['user_id'] == recent_test_user_id and unread_count > 0) else "unbold"
                
                try:
                    phone_numbers.append(
                            UserIDResponse(
                                phone_number= str(doc['user_id']) , #"9471123456", # ,
                                name="Test User" , # f"User_{doc['user_id']}",
                                consultant_mode=doc['state']['consultant_mode'],
                                time_stamp=doc['state']['time_stamp'],
                                unread_count=unread_count,  # Actual count for web portal users
                                studio_style=studio_style,  # Bold if recent test user with unread messages
                            )
                        )
                    k = 1
                except KeyError:
                    continue
        

        
        return phone_numbers
    except Exception as e:
        logger.debug(f"Error list_user_ids/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message":f"Error listing users: {str(e)}"})

## APIs for handling consultant connections
@app.post('/consultant_send_message')
async def send_consultant_message(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = "001"
        user_id = payload.get("userId")
        
        # update the user chat_history   
        ret = user_state_collection.get_one_document({'org_id' : org_id, 'user_id': user_id})
        if "response_text" in payload:
            consultant_response = payload.get("response_text")
            msg = AIMessage(content=consultant_response, additional_kwargs={"flag" : "consultant", 
                                                                            "time_stamp" : datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")})
        
        elif "media_id" in payload:
            consultant_response = {}
            consultant_response['media_id'] = payload.get("media_id")
            consultant_response['caption'] = payload.get("caption", "")
            consultant_response['file_name'] = payload.get("file_name", "")
            file_data = {
                "media_id" : payload.get("media_id"),
                "file_name" : payload.get("file_name", ""),
                "caption" : payload.get("caption", "")
            }
            msg = AIMessage(content=consultant_response['caption'], additional_kwargs={
                "flag" : "consultant", 
                "time_stamp" : datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")})
            
            msg = AIMessage(content=json.dumps(file_data), additional_kwargs={
                "file" : "true",
                "flag" : "consultant", 
                "time_stamp" : datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")})
        
        if ret:
            chat_history = ret['chat_history']
            chat_history.append(msg.model_dump())
            user_state_collection.update_one({'org_id' : org_id, 'user_id': user_id}, {'chat_history' : chat_history}) 
        
        # Check if the user_id is a phone number
        if is_phone_number(user_id) and not isinstance(consultant_response, dict):
            # send the message to whatsapp
            send_message(consultant_response, user_id)
            return Response(content="whatsapp response send successfully", status_code=200)
        
        elif is_phone_number(user_id) and isinstance(consultant_response, dict):
            # send the message to whatsapp
            send_message(consultant_response, user_id, type='document')
            return Response(content="whatsapp response send successfully", status_code=200)
        
        elif not is_phone_number(user_id) and not isinstance(consultant_response, dict):
            # TODO : support document uploads for the web chat
            # send the message through web UI   
            api_result = {
                "bot_text" : consultant_response
            }
            return {"userId": user_id, "result": api_result}
        
    except Exception as e:
        logger.debug(f"Error consultant_send_message/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message":f"Error sending consultant messsage: {str(e)}"})


@app.post("/number_of_test_conversations")
async def get_number_of_test_conversations(request: Request):
    payload = await request.json()
    org_id = payload.get("org_id")
    # org_id = '001'
        
    docs = user_state_collection.get_documents({"org_id" : org_id})
    
    test_conv_count = 0
    
    for doc in docs:
        if not is_phone_number(str(doc['user_id'])):
            test_conv_count += 1
            
    return test_conv_count


@app.post('/get_conversation')
async def get_conversation(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = "001"
        user_id = payload.get("userId")
        
        # load data from db
        ret = user_state_collection.get_one_document({'org_id' : org_id, 'user_id': user_id})
        
        if ret:
            chat_history = []
            for msg_dict in ret['chat_history']:
                if msg_dict['type'] == 'ai':
                    # Identify consultant messages
                    if 'flag' in msg_dict['additional_kwargs']:
                        if 'file' in msg_dict['additional_kwargs']:
                            chat_history.append({"file": True,
                                                "consultant": msg_dict['content'],
                                                "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                        else:  
                            chat_history.append({"file": False,
                                                "consultant": msg_dict['content'],
                                                "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                            
                    elif 'file' in msg_dict['additional_kwargs']:
                        # file_data = json.loads(msg_dict['content'])
                        chat_history.append({"file": True,
                                            msg_dict['type'] : msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                    else:
                        chat_history.append({"file": False,
                                            msg_dict['type']: msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                
                else:
                    if 'file' in msg_dict['additional_kwargs']:
                        # file_data = json.loads(msg_dict['content'])
                        chat_history.append({"file": True,
                                            msg_dict['type'] : msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                    else:
                        chat_history.append({"file": False,
                                            msg_dict['type']: msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                        
            return {"conversation" : chat_history}
        else:
            return {"conversation" : []}
    
    except Exception as e:
        logger.debug(f"Error get_conversation/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message":f"Error retriving conversations: {str(e)}"})
    

    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = "001"
        
        docs = user_state_collection.get_documents({"org_id": org_id}, sort=[("state.time_stamp", -1)])
        
        if docs:
            latest_doc = None
            for doc in docs:
                if not is_phone_number(str(doc["user_id"])):
                    latest_doc = doc
                    break
                
            chat_history = []
            for msg_dict in latest_doc['chat_history']:
                if msg_dict['type'] == 'ai':
                    # Identify consultant messages
                    if 'flag' in msg_dict['additional_kwargs']:
                        if 'file' in msg_dict['additional_kwargs']:
                            chat_history.append({"file": True,
                                                "consultant": msg_dict['content'],
                                                "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] }),
                        else:  
                            chat_history.append({"file": False,
                                                    "consultant": msg_dict['content'],
                                                    "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                    elif 'file' in msg_dict['additional_kwargs']:
                        # file_data = json.loads(msg_dict['content'])
                        chat_history.append({"file": True,
                                            msg_dict['type'] : msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                    else:
                        chat_history.append({"file": False,
                                            msg_dict['type']: msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                
                else:
                    if 'file' in msg_dict['additional_kwargs']:
                        # file_data = json.loads(msg_dict['content'])
                        chat_history.append({"file": True,
                                            msg_dict['type'] : msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                    else:
                        chat_history.append({"file": False,
                                            msg_dict['type']: msg_dict['content'],
                                            "time_stamp" : msg_dict['additional_kwargs']['time_stamp'] })
                        
            return {"user_id": latest_doc['state']['user_id'], "conversation" : chat_history, 
                    "consultant_mode" : latest_doc['state']['consultant_mode']}

        else:
            return {"user_id" : "" , "conversation" : [], "consultant_mode" : False}
    except Exception as e:
        logger.debug(f"Error get_conversation/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message":f"Error retriving test conversation: {str(e)}"})
    
@app.post('/consultant_mode_switch')
async def set_consultant_mode(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        # org_id = "001"
        user_id = payload.get("userId")
        consultation_mode = payload.get("mode") ## bool
        
        if consultation_mode == "on":
            mode_ = True
        elif consultation_mode == "off":
            mode_ = False
        else:
            mode_ = False
        doc = user_state_collection.get_one_document({"org_id" : org_id, "user_id": user_id})
        
        if doc and ('state' in doc):
            # update the consultant_mode in the DB
            user_state_collection.update_one({"org_id" : org_id, "user_id": user_id} , {'state.consultant_mode' : mode_})
            
            if not mode_:
                # get the conversation between user and consultant, and do necessary updates in DB
                
                # get the latest consultant session start time
                start_session = datetime.strptime(doc["consultant_session"], "%Y-%m-%d %H:%M:%S")
                user_state_collection.update_one(
                        {"org_id": org_id, "user_id": user_id}, 
                        {'state.escalation_status.escalation_mode': 'off'}
                    )
                # fetch updated doc
                doc = user_state_collection.get_one_document({"org_id": org_id, "user_id": user_id})
                logger.debug(f"user_state_collection updated: {doc['state']['escalation_status']['escalation_mode']}")

                consultant_conversation = []
                for msg in doc['chat_history']:
                    if 'flag' in msg['additional_kwargs']:
                        if msg['additional_kwargs']['flag'] == 'consultant':
                            msg_time = datetime.strptime(msg['additional_kwargs']['time_stamp'], "%Y-%m-%d %H:%M:%S") 
                            if start_session <= msg_time:
                                consultant_conversation.append(msg)

                # analyse_consultant_conversation(consultant_conversation, SERVICE_INFO)
            elif mode_:
                # update the db for new consultant session (which will be identified through the time stamp)
                current_time = datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")
                user_state_collection.update_one({"org_id" : org_id, "user_id": user_id} , 
                                         {"consultant_session" : current_time})
             
            return Response(content=f"set the consultation mode to : {consultation_mode}", status_code=200) 
        
    except Exception as e:
        logger.debug(f"Error consultant_mode_switch/ : \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"status": "error", "message":f"Error setting the consultant mode: {str(e)}"})


## Conversation APIs
@app.post("/common_chat/")
async def common_chat(
    background_tasks: BackgroundTasks,
    org_id: str = Form(...),
    userId: str = Form(...),
    question: str = Form(default=""),
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    language_code: str = Form(default="en-US"),
):
    user_id = userId
    user_input = question
    file_url = None
    file_type = None
    file_name = None

    # Handle audio input: transcribe and use as question when provided
    if audio and audio.filename:
        try:
            audio_bytes = await audio.read()
            mime_type = audio.content_type or "audio/webm"
            transcribed = transcribe_audio_bytes(audio_bytes, mime_type=mime_type, language_code=language_code)
            if transcribed:
                user_input = f"{transcribed} {question}".strip() if question else transcribed
            logger.debug(f"user: {user_id} : speech transcription : {transcribed}")
        except Exception:
            logger.debug(f"Error transcribing audio for user {user_id}: \n {traceback.format_exc()}")

    # Handle file upload: image (vision) or PDF (text extraction)
    if file and file.filename:
        file_content = await file.read()
        mime_type = file.content_type or ""
        filename = file.filename

        file_ext = os.path.splitext(filename)[1] or ""
        file_id = f"{uuid.uuid4().hex}{file_ext}"
        save_path = os.path.join("chat_uploads", file_id)
        with open(save_path, "wb") as fp:
            fp.write(file_content)
        file_url = f"/chat_uploads/{file_id}"
        file_type = mime_type
        file_name = filename

        if mime_type.startswith("image/"):
            from langchain_openai import ChatOpenAI as _ChatOpenAI
            from langchain_core.messages import HumanMessage as _HumanMessage
            _vision_llm = _ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, max_tokens=1024, temperature=0)
            image_b64 = base64.b64encode(file_content).decode()
            _vision_msg = _HumanMessage(content=[
                {"type": "text", "text": "Describe what you see in this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
            ])
            enhanced_input = _vision_llm.invoke([_vision_msg]).content
        elif "pdf" in mime_type or filename.lower().endswith(".pdf"):
            _pdf_proc = PDFProcessor()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                pdf_text = _pdf_proc.extract_text_from_pdf(tmp_path)
            finally:
                os.remove(tmp_path)
            enhanced_input = f"User uploaded document '{filename}':\n{pdf_text[:4000]}"
        else:
            enhanced_input = f"User uploaded a file: {filename}"

        user_input = f"Uploaded content: {enhanced_input}\n\nUser question: {question}" if question else enhanced_input

    logger.debug(f"user: {user_id} : input : {user_input}")

    # Track new message for unread count (only for web portal/test users, not WhatsApp)
    track_new_message(org_id, user_id)

    # get the available knowledge bases
    org_doc = org_collection.get_one_document({'org_id': org_id})
    if org_doc:
        selected_knowledge_bases = org_doc['selected_knowledge_bases']
    else:
        selected_knowledge_bases = KB

    ret = user_state_collection.get_one_document({'user_id': user_id})

    if ret:
        chat_history = []
        for msg_dict in ret['chat_history']:
            msg_type = msg_dict['type']
            if msg_type == 'human':
                chat_history.append(HumanMessage(**msg_dict))
            elif msg_type == 'ai':
                chat_history.append(AIMessage(**msg_dict))

        state = {}
        state["chat_history"] = chat_history
        state.update(ret['state'])
        state['selected_knowledge_bases'] = selected_knowledge_bases
        current_time = datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")
        state["time_stamp"] = current_time

    else:
        current_time = datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")
        state = {
            "org_id": org_id,
            "user_id": user_id,
            "time_stamp": current_time,
            "consultant_mode": False,
            "selected_knowledge_bases": selected_knowledge_bases,
            "chat_history": [],
            "service_info": SERVICE_INFO
        }

    state["user_input"] = user_input

    state = chatbot_app.invoke(state)
    return_quotation = True

    if not state['consultant_mode']:
        if 'CONSULTANT_CONF:' in state['bot_responce']:
            return_quotation = False
            api_result = {"bot_text": state['bot_responce'].replace('CONSULTANT_CONF:', '')}
        else:
            # return_quotation = False
            api_result = {"bot_text": state['bot_responce']}
    else:
        return_quotation = False
        if 'CONSULTANT_CONF:' in state['bot_responce']:
            api_result = {"bot_text": state['bot_responce'].replace('CONSULTANT_CONF:', '')}
        elif 'CONSULTANT_MODE_ON' in state['bot_responce']:
            api_result = {"bot_text": state['bot_responce']}
            logger.debug(f"user: {user_id} is connected with a consultant now")

    if file_url:
        api_result["file_url"] = file_url
        api_result["file_type"] = file_type
        api_result["file_name"] = file_name

    # save the state in db
    db_data = {}
    db_data["org_id"] = org_id
    db_data["user_id"] = user_id
    db_data["chat_history"] = [msg.model_dump() for msg in state['chat_history']]

    try:
        completed_bookings = {"org_id": org_id, "user_id": user_id}
        for service, service_data in state['booking_session'].items():
            if service_data['status'] == 'complete' and ('search_object' in service_data):
                completed_bookings[service] = service_data['search_object']
        if len(completed_bookings) > 2:
            # completed_bookings["intiate_date"] = datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")
            booking_collection.update_one({'user_id': user_id}, completed_bookings)
            api_result["completed_bookings"] = completed_bookings
            remove_key_recursive(state, 'search_object')
    except KeyError:
        pass

    logger.debug(f"user: {user_id} : AI response : {state['bot_responce']}")

    if ("completed_bookings" in api_result) and (len(api_result["completed_bookings"]) > 2) and return_quotation:
        quotation = generate_quotation(completed_bookings)
        api_result["completed_bookings"] = quotation
        state['chat_history'] = state['chat_history'] + [
            AIMessage(content=quotation, additional_kwargs={"time_stamp": datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")})
        ]
        api_result['bot_text'] = f"{api_result['bot_text']} \n\n {api_result['completed_bookings']}"

    db_data["chat_history"] = [msg.model_dump() for msg in state['chat_history']]
    db_data['state'] = {k: v for k, v in state.items() if k != 'chat_history'}
    user_state_collection.update_one({'user_id': user_id}, db_data)

    # Text-to-speech: synthesize bot reply and include audio URL in response
    if USE_TTS:
        bot_text_for_tts = api_result.get('bot_text', '')
        if bot_text_for_tts:
            try:
                audio_path = os.path.join("audio_tts", f"{user_id}.mp3")
                actual_path = await synthesize_text(bot_text_for_tts, audio_path, language_code=language_code)
                audio_filename = os.path.basename(actual_path)
                api_result['audio_url'] = f"/audio_tts/{audio_filename}"
            except Exception:
                logger.error(f"TTS error for user {user_id}: \n {traceback.format_exc()}")

    return {"userId": user_id, "result": api_result}


@app.post("/set_tts")
async def set_tts(request: Request):
    global USE_TTS
    payload = await request.json()
    USE_TTS = bool(payload.get("enabled", False))
    return {"tts_enabled": USE_TTS}


#applicable only for portal
@app.post("/mark_as_read_testchat/")
async def mark_as_read(request: Request):
    """Mark messages as read for specific user"""
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        user_id = payload.get("user_id")
        
        # Reset unread count to 0
        unread_counts_collection.update_one(
            {"org_id": org_id, "user_id": user_id},
            {
                "unread_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        )
        
        # Verify the update
        updated_doc = unread_counts_collection.get_one_document({'org_id': org_id, 'user_id': user_id})
        logger.debug(f"Mark as read - org_id: {org_id}, user_id: {user_id}, updated unread_count: {updated_doc['unread_count'] if updated_doc else 'Document not found'}")
        
        return {
            "success": True,
            "org_id": org_id,
            "user_id": user_id,
            "unread_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error marking as read: {str(e)}")
        return {"success": False, "error": str(e)}



# Service display names and which field to use as description
_SERVICE_DISPLAY = {
    "telco_support":  ("Telco Support",  ["issue type", "issue description"]),
    "new_connection": ("New Connection", ["connection type", "installation address"]),
}

def _format_booking(doc, idx: int) -> dict:
    """Convert a raw booking_collection document into an admin-UI booking dict."""
    services = [k for k in doc if k not in ("_id", "org_id", "user_id", "status", "booking_status")]
    service_key = services[0] if services else None
    service_data = doc.get(service_key, {}) if service_key else {}

    if isinstance(service_data, dict) and service_data:

        display_name, desc_fields = _SERVICE_DISPLAY.get(service_key, (service_key or "Booking", []))

        # Build a short destination/description from the service data
        destination_parts = [str(service_data.get(f, "")) for f in desc_fields if service_data.get(f)]
        destination = " – ".join(destination_parts) if destination_parts else display_name

        # Extract customer name and phone from service data
        customer_name = (
            service_data.get("full name")
            or service_data.get("name")
            or doc.get("user_id", "Unknown")
        )
        phone = (
            service_data.get("contact number")
            or service_data.get("account number or registered phone number")
            or doc.get("user_id", "")
        )

        booking_id = f"BK-{str(doc.get('_id', idx))[-5:].upper()}"
        status = doc.get("booking_status", "pending")

        return {
            "id": booking_id,
            "customer_name": customer_name,
            "phone": str(phone),
            "service_type": display_name,
            "destination": destination,
            "date": service_data.get("preferred installation date") or service_data.get("date") or "—",
            "guests": 1,
            "amount": 0,
            "status": status,
            "_raw_id": str(doc.get("_id", "")),
        }

    # Fallback for documents with no recognisable service data
    return {
        "id": f"BK-{str(doc.get('_id', idx))[-5:].upper()}",
        "customer_name": doc.get("user_id", "Unknown"),
        "phone": str(doc.get("user_id", "")),
        "service_type": "Unknown",
        "destination": "—",
        "date": "—",
        "guests": 1,
        "amount": 0,
        "status": doc.get("booking_status", "pending"),
        "_raw_id": str(doc.get("_id", "")),
    }


@app.post("/get_bookings")
async def get_bookings(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        docs = booking_collection.get_documents({"org_id": org_id})
        bookings = [_format_booking(d, i) for i, d in enumerate(docs)]
        return {"bookings": bookings}
    except Exception as e:
        logger.debug(f"Error get_bookings/: \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_booking_status")
async def update_booking_status(request: Request):
    try:
        payload = await request.json()
        org_id = payload.get("org_id")
        raw_id = payload.get("booking_id", "")
        new_status = payload.get("status", "pending")

        # booking_id from the UI is the formatted "BK-XXXXX"; raw_id may also be the full _id string
        docs = booking_collection.get_documents({"org_id": org_id})
        for doc in docs:
            doc_raw_id = str(doc.get("_id", ""))
            formatted_id = f"BK-{doc_raw_id[-5:].upper()}"
            if formatted_id == raw_id or doc_raw_id == raw_id:
                booking_collection.update_one({"_id": ObjectId(doc_raw_id)}, {"booking_status": new_status})
                return {"success": True}

        return Response(content="Booking not found", status_code=404)
    except Exception as e:
        logger.debug(f"Error update_booking_status/: \n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5001)
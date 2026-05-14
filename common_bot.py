
"""
Chat model
"""

from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils.mongo_manager import MongoDBmanager
from dotenv import load_dotenv
load_dotenv()
import os
import json
import pathlib
from datetime import datetime
from utils.pdf_processor import DocumentKB
import re
from utils.logger import get_debug_logger
from utils.prompts import *
from utils.web_processor import url_filter_agent
from utils.whatsapp_utills import send_message
from utils.miscellaneous import parse_services, get_uncompleted, get_consultant_conversation

from zoneinfo import ZoneInfo

logger = get_debug_logger(
    "common", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "./logs/common_bot.log")
)

google_api_key = os.getenv("GOOGLE_API_KEY")
KB = os.getenv("KB")

user_state_collection = MongoDBmanager("user_states")
booking_collection = MongoDBmanager("user_bookings")
booking_search_collection = MongoDBmanager("user_search_objects")
pdf_doc_collection = MongoDBmanager("pdf_documents")

document_kb = DocumentKB(pdf_doc_collection, google_api_key)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TIME_ZONE = os.environ["TIME_ZONE"]
AGENTS = json.loads(os.environ["AGENTS"])

llm = ChatOpenAI(
    model='gpt-4.1-mini',
    api_key=OPENAI_API_KEY,
    max_tokens=2048,
    temperature=0,
)

qa_chain = qa_prompt | llm | StrOutputParser()
chain = question_type_prompt_template | llm | StrOutputParser()
booking_info_req_chain = booking_info_request_prompt_template | llm | StrOutputParser()
escalation_chain = escalation_prompt_template | llm | JsonOutputParser()
consultation_mode_chain = consultant_switch_prompt_template | llm | StrOutputParser()
consutalnt_session_extraction_chain = consultant_conv_data_extractor_prompt_template | llm | JsonOutputParser()
validation_chain = question_validator_prompt_template | llm | JsonOutputParser()
knowledge_classifier = kb_classifier_prompt | llm | StrOutputParser()


def analyse_consultant_conversation(chat_history, service_info):
    extracted_data = consutalnt_session_extraction_chain.invoke({
        "chat_history": chat_history,
        "service_info": service_info
    })
    return extracted_data


def detect_service_type(chat_history, user_input, service_info):
    """
    Uses LLM to detect service type(s) based on chat history + current user input.
    """
    services_list = "\n".join([f"- {key}: {val['description']}" for key, val in service_info.items()])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are a service detection bot.

Your job is to identify which of the following services the user is requesting based on the conversation history and current input.

1. Prioritize the current user input if it clearly indicates a new service.
2. If the current input does not clearly indicate a new service, check the last AI message and conversation history to determine which service the user is continuing.
3. If the current user input indicates multiple services, output them as a list.

Available services:
{services_list}

Return ONLY the service key (e.g. service_a). If user input is unrelated to these services, return "none".

Examples for numerical/date inputs where context matters:
chat_history:
    Human: I want to book service_a
    AI: How many people will be attending?
    Human: 3
    AI: What date would you like?
user_input: 2025-12-02
Output: service_a

Examples for multiple services:
user_input: I want both service_a and service_b
Output: [service_a, service_b]
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    service_type_chain = prompt_template | llm | StrOutputParser()

    result = service_type_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    detected_service = parse_services(result)

    detected_service_ = []
    for ser in detected_service:
        if ser in service_info.keys():
            detected_service_.append(ser)
        else:
            logger.debug(f"Unidentified service: {ser}")

    return detected_service_


def detect_general_question(chat_history, user_input, service_info):
    """
    Returns True if the user input contains a general question (not a booking request).
    """
    services_list = {}
    for key, val in service_info.items():
        services_list[key] = {
            "description": val['description'],
            "required_info": val['required_info']
        }

    result = chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "services_list": services_list
    })

    return result == "yes"


def call_qa_agent(state):
    """
    QA agent — answers general questions using RAG over PDFs and website content.
    """
    chat_history = state['chat_history']
    user_input = state['user_input']
    service_info = state['service_info']
    user_id = state['user_id']
    org_id = state['org_id']

    try:
        booking_status = {}
        booking_session = state['booking_session']
        for key, val in booking_session.items():
            booking_status[key] = val['status']
    except KeyError:
        booking_status = "No booking/service requested yet"

    services_list = {}
    for key, val in service_info.items():
        services_list[key] = {
            "description": val['description'],
            "required_info": val['required_info']
        }

    services_filter_list = "\n".join([
        f"- {key}: {val['filtering_info']}" for key, val in service_info.items()
    ])

    rag_invoke = knowledge_classifier.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })
    context = {}

    if rag_invoke == 'yes':
        if 'doc' in state['selected_knowledge_bases']:
            try:
                context["document_context"] = document_kb.get_context_for_question(org_id, user_input)
            except Exception as e:
                logger.debug(f"Issue extracting context from document: {e}")

        if 'web' in state['selected_knowledge_bases']:
            try:
                context["web_context"] = url_filter_agent(org_id, user_input)
            except Exception as e:
                logger.debug(f"Issue extracting context from web: {e}")

    qa_result = qa_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "services_list": services_list,
        "services_filter_list": services_filter_list,
        "context": context,
        "booking_stat": booking_status
    })

    return qa_result


def validation_node(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    try:
        validation_results = validation_chain.invoke({
            "user_input": user_input,
            "date_time": datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d"),
            "chat_history": chat_history
        })
    except OutputParserException as e:
        validation_results = {}
        validation_results['validation_state'] = 'no'
        validation_results['response'] = e.llm_output

    state['valid_question'] = (validation_results['validation_state'] == 'yes')
    state['validation_response'] = validation_results['response']
    return state


def router_node(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    service_info = state['service_info']

    if not state['consultant_mode']:
        if state['valid_question']:
            if not detect_general_question(chat_history, user_input, service_info):
                next_node = "booking_agent"
            else:
                next_node = "qa_agent"
        else:
            next_node = "end"
            state['chat_history'] = chat_history + [
                HumanMessage(content=user_input, additional_kwargs={"time_stamp": state['time_stamp']}),
            ]
            state['bot_responce'] = state['validation_response']
    else:
        next_node = "consultant"

    state['next_node'] = next_node
    logger.debug(f"next node: {next_node}")
    return state


def required_information_extraction(user_input, chat_history, service_type, require_info):
    """
    Extracts required booking fields from the user input and chat history.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
Search for the following required information for the {service_type} in the user input and conversation history.
The user might refer to information they provided earlier in the conversation.

Required fields:
{require_info}

Note: Consider the current date {datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")}
when calculating dates if the user uses relative phrases like 'next month', 'December', etc.

If found, output a JSON object where each key is a required field and the value is the extracted value.
If not found, set that key's value to null.

Return only the JSON object. No extra explanation.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    required_info_extraction_chain = prompt_template | llm | JsonOutputParser()

    result = required_info_extraction_chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
    })

    return result


def booking_infomation_request(pending_information, collected_information, chat_history):
    result = booking_info_req_chain.invoke({
        "pending_info": pending_information,
        "collected_info": collected_information,
        "chat_history": chat_history
    })
    return result


def booking_agent_node(state):
    chat_history = state['chat_history']
    user_input = state['user_input']
    service_info = state['service_info']

    booking_session = state.get('booking_session', {})

    service_type_detected = detect_service_type(chat_history, user_input, service_info)
    if service_type_detected:
        service_type = get_uncompleted(booking_session, service_type_detected)

        if isinstance(service_type, str):
            state['bot_responce'] = service_type
            return state
        else:
            state['current_service'] = service_type[0]
            if service_type[0] not in booking_session:
                booking_session[service_type[0]] = {}
    else:
        service_type = []

    if not service_type:
        qa_response = call_qa_agent(state)
        state['bot_responce'] = qa_response
        state['chat_history'] = chat_history + [
            HumanMessage(content=user_input, additional_kwargs={"time_stamp": state['time_stamp']}),
        ]
    else:
        required_info = service_info[service_type[0]]["required_info"]

        default_template = {key: None for key in required_info}
        
        collected_info = booking_session[service_type[0]].get('collected_info', default_template)

        try:
            extracted_info = required_information_extraction(user_input, chat_history, service_type[0], required_info)
        except OutputParserException as e:
            state['bot_responce'] = e.llm_output
            return state

        for key, value in extracted_info.items():
            if value is not None:
                collected_info[key] = value

        collected_info_list = "\n".join([
            f"- {key}: {val}" for key, val in collected_info.items() if val is not None
        ])

        pending_info = [key for key in required_info if collected_info[key] is None]

        has_general_question = detect_general_question(chat_history, user_input, service_info)
        response_text = ""

        if pending_info:
            request_info = booking_infomation_request(pending_info, collected_info_list, chat_history)
            response_text += request_info
        else:
            # All required fields collected — mark booking complete
            if not booking_session[service_type[0]]:
                state['booking_session'] = state.get('booking_session', {})
                state['booking_session'][service_type[0]] = {
                    'collected_info': collected_info,
                    'pending_info': pending_info,
                    'status': 'complete',
                    'search_object': collected_info,
                }
            else:
                if state['booking_session'][service_type[0]].get('status') == 'incomplete':
                    state['booking_session'][service_type[0]]['collected_info'] = collected_info
                    state['booking_session'][service_type[0]]['pending_info'] = pending_info
                    state['booking_session'][service_type[0]]['status'] = 'complete'
                    state['booking_session'][service_type[0]]['search_object'] = collected_info

            service_label = service_type[0].replace('_', ' ').title()
            response_text += f"Your {service_label} information is complete. Please refer to the following summary.\n"

            booking_session[service_type[0]]['search_object'] = collected_info
            logger.debug(f"Booking complete for user {state['user_id']}, service {service_type[0]}: {collected_info}")

            booking_search_collection.update_one(
                {"org_id": state['org_id'], "user_id": state['user_id']},
                {service_type[0]: collected_info}
            )

        if has_general_question:
            qa_answer = call_qa_agent(state)
            response_text += f"\nRegarding your other question: {qa_answer}"

        booking_session[service_type[0]]["collected_info"] = collected_info
        booking_session[service_type[0]]["pending_info"] = pending_info
        booking_session[service_type[0]]["status"] = "complete" if not pending_info else "incomplete"

        state['booking_session'] = booking_session

        state['chat_history'] = chat_history + [
            HumanMessage(content=user_input, additional_kwargs={"time_stamp": state['time_stamp']}),
        ]
        state['bot_responce'] = response_text

    return state


def qa_agent_node(state):
    chat_history = state['chat_history']
    user_input = state['user_input']

    qa_answer = call_qa_agent(state)

    state['chat_history'] = chat_history + [
        HumanMessage(content=user_input, additional_kwargs={"time_stamp": state['time_stamp']}),
    ]
    state['bot_responce'] = qa_answer

    return state


def escalation_agent(state):
    user_input = state['user_input']
    chat_history = state['chat_history']
    previous_agent = state['next_node']

    cons_chat = get_consultant_conversation(chat_history)

    escalation_result = escalation_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "previous_agent": previous_agent,
        "consultant_chat_history": cons_chat
    })

    return escalation_result


def escalation_agent_node(state):
    if not state['consultant_mode']:
        try:
            ret = escalation_agent(state)
        except OutputParserException as e:
            state['bot_responce'] = e.llm_output
            return state

        state['escalation_status'] = ret
        logger.debug(f"escalation_status: {ret}")

        if ret['escalation_mode'] == 'on':
            state['escalation_node'] = 'consultant'
            try:
                if state['next_node'] == 'booking_agent':
                    if state['booking_session'][state['current_service']]['status'] == 'complete':
                        state['booking_session'][state['current_service']]['status'] = 'incomplete'
            except Exception:
                pass

        elif ret['escalation_mode'] == 'off':
            state['escalation_node'] = 'end'
    else:
        state['escalation_node'] = 'end'

    return state


def consultant_mode_switch(state):
    user_input = state['user_input']
    chat_history = state['chat_history']

    consultation_mode = consultation_mode_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    if consultation_mode == 'yes':
        state['consultant_mode'] = True
        current_time = datetime.now(ZoneInfo(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S")
        user_state_collection.update_one(
            {"org_id": state['org_id'], "user_id": state['user_id']},
            {"consultant_session": current_time}
        )
        send_message(f"User {state['user_id']} request for support", AGENTS[0])
    else:
        state['consultant_mode'] = False

    return state


def consultant_agent_node(state):
    user_input = state['user_input']
    chat_history = state['chat_history']

    if not state['consultant_mode']:
        escalation_reason = state['escalation_status']['reason']

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
You are SAM, a friendly customer support specialist at a telecommunications company. You are a real person — never refer to yourself as a bot or AI.

Your task is to handle the handoff to a senior support agent when needed:

1. Consider the reason this situation needs a senior agent: {escalation_reason}.
2. If the user has not yet been asked, let them know warmly and ask for their confirmation to connect them with a senior agent, mentioning the reason naturally.
   Example: "For this I'll need to bring in one of our senior agents who can help you directly. Would that be okay?"
3. If the user confirms, respond exactly with: "One of our consultants will connect with you shortly."
4. Do not ask again if confirmation has already been received.

- If user not confirmed yet → Ask warmly and naturally as SAM would.
- If user confirmed or agreed → Output exactly: "One of our consultants will connect with you shortly."
- Never provide explanations, reasoning, or additional text beyond this.

Example (Implicit Acceptance e.g., "okay", "alright", "fine"):
SAM: "For this I'll need to connect you with one of our senior agents. Would that be okay?"
User: "ok"
Output: "One of our consultants will connect with you shortly."
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])

        consultation_chain = prompt_template | llm | StrOutputParser()

        consultation_response = consultation_chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })

        state['chat_history'] = chat_history + [
            AIMessage(content=consultation_response, additional_kwargs={"time_stamp": state['time_stamp']})
        ]
        state['bot_responce'] = f"CONSULTANT_CONF:{consultation_response}"

        state = consultant_mode_switch(state)

    elif state['consultant_mode']:
        state['chat_history'] = chat_history + [
            HumanMessage(content=user_input, additional_kwargs={"flag": "consultant", "time_stamp": state['time_stamp']})
        ]
        state['bot_responce'] = "CONSULTANT_MODE_ON"

    logger.debug(f"state: {state}")
    return state


def end_node(state):
    state['chat_history'] = state['chat_history'] + [
        AIMessage(content=state['bot_responce'], additional_kwargs={"time_stamp": state['time_stamp']})
    ]
    return state


class ServiceState(dict):
    user_id: str
    org_id: str
    selected_knowledge_bases: list
    chat_history: list
    user_input: str
    time_stamp: str
    service_info: dict
    next_node: str
    booking_session: dict
    current_service: str
    bot_responce: str
    escalation_node: str
    escalation_status: dict
    consultant_mode: bool
    valid_question: bool
    validation_response: str


def common_chatbot():
    graph = StateGraph(ServiceState)
    graph.add_node("validation_node", validation_node)
    graph.add_node("router", router_node)
    graph.add_node("qa_agent", qa_agent_node)
    graph.add_node("booking_agent", booking_agent_node)
    graph.add_node("escalation_check", escalation_agent_node)
    graph.add_node("consultant_agent", consultant_agent_node)
    graph.add_node("end_node", end_node)

    graph.set_entry_point("validation_node")

    graph.add_edge("validation_node", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "qa_agent": "qa_agent",
            "booking_agent": "booking_agent",
            "consultant": "consultant_agent",
            "end": "end_node"
        },
    )

    graph.add_edge("qa_agent", "escalation_check")
    graph.add_edge("booking_agent", "escalation_check")

    graph.add_conditional_edges(
        "escalation_check",
        lambda state: state["escalation_node"],
        {
            "end": "end_node",
            "consultant": "consultant_agent",
        },
    )

    graph.add_edge("end_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = common_chatbot()
    image_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_bytes)
    print("Saved to graph.png")

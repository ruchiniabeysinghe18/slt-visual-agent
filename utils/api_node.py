"""
API Node utilities — loads the OpenAPI spec, converts it to LLM tool schemas,
handles multi-turn parameter collection, simulates API calls, and formats responses.
"""

import json
import pathlib
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from utils.logger import get_debug_logger

logger = get_debug_logger(
    "api_node",
    pathlib.Path.joinpath(pathlib.Path(__file__).parent.parent.resolve(), "./logs/api_node.log")
)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model='gpt-4.1-mini',
    api_key=OPENAI_API_KEY,
    max_tokens=2048,
    temperature=0,
)

# ---------------------------------------------------------------------------
# Spec loading — parsed once at import time
# ---------------------------------------------------------------------------

_SPEC_PATH = pathlib.Path(__file__).parent.parent / "openapi_spec.json"

try:
    with open(_SPEC_PATH, "r") as _f:
        openapi_spec: dict = json.load(_f)
except FileNotFoundError:
    logger.warning("openapi_spec.json not found — API node running without a spec")
    openapi_spec: dict = {}


def openapi_to_tools(spec: dict) -> list:
    """Converts an OpenAPI 3.0 spec to OpenAI function-calling tool schemas."""
    tools = []
    for path_item in spec.get("paths", {}).values():
        for method, operation in path_item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            operation_id = operation.get("operationId", f"{method}_unknown")
            description = operation.get("description") or operation.get("summary", "")
            properties: dict = {}
            required: list = []

            for param in operation.get("parameters", []):
                name = param["name"]
                schema = param.get("schema", {"type": "string"})
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": param.get("description", ""),
                }
                if param.get("required", False):
                    required.append(name)

            body_schema = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            for prop_name, prop_schema in body_schema.get("properties", {}).items():
                properties[prop_name] = {
                    "type": prop_schema.get("type", "string"),
                    "description": prop_schema.get("description", ""),
                }
                if prop_name in body_schema.get("required", []):
                    required.append(prop_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": operation_id,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
    return tools


openai_tools: list = openapi_to_tools(openapi_spec)


def reload_spec() -> None:
    """Re-reads openapi_spec.json from disk and rebuilds the tools list in-place."""
    global openapi_spec, openai_tools
    try:
        with open(_SPEC_PATH, "r") as _f:
            openapi_spec = json.load(_f)
        openai_tools = openapi_to_tools(openapi_spec)
        logger.info("OpenAPI spec reloaded from disk")
    except FileNotFoundError:
        logger.warning("openapi_spec.json not found — spec not reloaded")



def build_api_summary(spec: dict = openapi_spec) -> str:
    """Returns a plain-text list of available operations for use in prompts."""
    lines = []
    for path_item in spec.get("paths", {}).values():
        for method, operation in path_item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            op_id = operation.get("operationId", "")
            desc = operation.get("description") or operation.get("summary", "")
            lines.append(f"- {op_id}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool metadata helpers
# ---------------------------------------------------------------------------

def get_tool_info(tool_name: str, tools: list = openai_tools) -> dict:
    """Returns description, required_params, all_params, and param_descriptions for a tool."""
    for tool in tools:
        fn = tool["function"]
        if fn["name"] == tool_name:
            params = fn["parameters"]
            return {
                "description": fn.get("description", ""),
                "required_params": params.get("required", []),
                "all_params": list(params.get("properties", {}).keys()),
                "param_descriptions": {
                    k: v.get("description", "")
                    for k, v in params.get("properties", {}).items()
                },
            }
    return {}


# ---------------------------------------------------------------------------
# LLM-based API identification via function calling
# ---------------------------------------------------------------------------

def identify_api_and_params(chat_history: list, user_input: str) -> dict:
    """
    Uses LLM function calling to identify which API operation the user wants
    and extract whatever parameters are already available in the conversation.
    Returns {"tool_name": str | None, "extracted_params": dict}.
    """
    from langchain_core.messages import HumanMessage

    llm_with_tools = llm.bind_tools(openai_tools)
    messages = chat_history + [HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tc = response.tool_calls[0]
        return {"tool_name": tc["name"], "extracted_params": tc.get("args", {})}

    return {"tool_name": None, "extracted_params": {}, "data_request" : response.content}


# ---------------------------------------------------------------------------
# Parameter extraction for pending calls (multi-turn collection)
# ---------------------------------------------------------------------------

def extract_pending_params(user_input: str, chat_history: list,
                            pending_params: list, collected_params: dict,
                            operation_desc: str) -> dict:
    """
    Extracts values for the still-missing parameters from the user's latest message.
    Returns a dict keyed by param name; values are null when not found.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are extracting information for the following operation: {operation_desc}

Extract values for these parameters from the conversation:
{pending_params}

Currently collected: {collected_params}

Return ONLY a JSON object with the parameter names as keys.
Set the value to null if it was not provided.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ])

    chain = prompt_template | llm | JsonOutputParser()
    try:
        return chain.invoke({"user_input": user_input, "chat_history": chat_history})
    except OutputParserException:
        return {p: None for p in pending_params}


# ---------------------------------------------------------------------------
# User-facing messages
# ---------------------------------------------------------------------------

def ask_for_missing_params(missing_params: list, collected_params: dict,
                            chat_history: list, operation_desc: str,
                            param_descriptions: dict) -> str:
    """Generates a natural-language request for the still-missing parameters."""
    param_info = "\n".join(
        f"- {p}: {param_descriptions.get(p, '')}" for p in missing_params
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are SAM, a friendly customer support specialist at a telecommunications company.

The user wants to: {operation_desc}

You still need the following information to proceed:
{param_info}

Already collected: {collected_params}

Ask the user for the missing details naturally and conversationally.
Do not use technical terms like "parameters", "API", "customer_id", or "deviceId" —
describe what you need in plain, friendly language (e.g. "your account/customer ID").
Keep the request brief.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"chat_history": chat_history})


def format_api_response(tool_name: str, api_result: dict,
                         chat_history: list, user_input: str) -> str:
    """Presents the API result as a friendly, conversational message."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are SAM, a friendly customer support specialist at a telecommunications company.

Present the following result to the user in a clear, warm, and natural way.
Do not mention "API", "function", JSON, or any technical details.
Format any data neatly using plain text. Use line breaks for readability.
Keep your response concise and helpful.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
        ("system", "Result data:\n{api_result}"),
    ])

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "api_result": json.dumps(api_result, indent=2),
    })


# ---------------------------------------------------------------------------
# Simulated API calls (dummy responses for development)
# ---------------------------------------------------------------------------

def simulate_api_call(tool_name: str, params: dict) -> dict:
    """Returns a simulated API response. Replace with real HTTP calls in production."""

    dummy: dict = {
        "getDataBalance": {
            "status": "success",
            "customer_id": params.get("customer_id", "N/A"),
            "device_id": params.get("deviceId", "N/A"),
            "balances": [
                {
                    "package": "Main Data Balance",
                    "value": 15.5,
                    "unit": "GB",
                    "expiry": "2026-05-30"
                },
                {
                    "package": "Bonus Data",
                    "value": 2.0,
                    "unit": "GB",
                    "expiry": "2026-05-20"
                },
                {
                    "package": "Night Data",
                    "value": 50.0,
                    "unit": "GB",
                    "expiry": "2026-05-31"
                },
            ],
            "total_remaining_gb": 67.5,
        },

        "activatePackage": {
            "status": "success",
            "customer_id": params.get("customer_id", "N/A"),
            "package_id": params.get("package_id", "N/A"),
            "message": "Package activated successfully.",
            "activation_time": "2026-05-16 10:30:00",
            "validity": "30 days",
            "reference_no": "ACT-2026-0516-7842",
        },

        "cancelPackage": {
            "status": "success",
            "customer_id": params.get("customer_id", "N/A"),
            "package_id": params.get("package_id", "N/A"),
            "message": "Package cancellation request submitted.",
            "reference_no": "CAN-2026-0516-3301",
            "effective_date": "End of current billing cycle (2026-05-31)",
        },

        "getCurrentPlan": {
            "status": "success",
            "customer_id": params.get("customer_id", "N/A"),
            "plan_name": "Fiber 100 Mbps Unlimited",
            "monthly_fee": "LKR 3,999",
            "data_limit": "Unlimited",
            "speed": "100 Mbps download / 50 Mbps upload",
            "contract_end_date": "2026-12-31",
            "account_status": "Active",
            "next_billing_date": "2026-06-01",
        },

        "payBill": {
            "status": "success",
            "account_number": params.get("account_number", "N/A"),
            "total_payable_amount": params.get("total_payable_amount", "N/A"),
            "transaction_id": "PAY-2026-0523-9981",
            "payment_status": "Completed",
            "payment_method": "Credit Card",
            "payment_date": "2026-05-23 14:45:00",
            "message": "Bill payment completed successfully."
        },
    }

    return dummy.get(
        tool_name,
        {
            "status": "success",
            "message": "Operation completed successfully."
        }
    )
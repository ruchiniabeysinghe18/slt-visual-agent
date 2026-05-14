"""
Generic booking summary / quotation formatter.
Formats completed booking sessions into a readable text summary.
"""
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import os
from zoneinfo import ZoneInfo

TIME_ZONE = os.environ["TIME_ZONE"]


def generate_quotation(data: dict) -> str:
    """
    Generate a text summary for all completed bookings.

    Args:
        data: dict with keys org_id, user_id, and one entry per completed service.
              Each service value is a dict of collected field → value.

    Returns:
        str: Formatted quotation text
    """
    data = dict(data)
    data.pop("org_id", None)
    user_id = data.pop("user_id", "N/A")

    lines = []
    lines.append("Service Summary")
    lines.append(f"Date      : {datetime.now(ZoneInfo(TIME_ZONE)).strftime('%Y-%m-%d')}")
    lines.append(f"Reference : REF-{user_id}")
    lines.append("-" * 50)

    for service_key, service_data in data.items():
        service_label = service_key.replace("_", " ").title()
        lines.append(f"\n{service_label}")

        if isinstance(service_data, dict):
            for field, value in service_data.items():
                if value is not None:
                    field_label = field.replace("_", " ").title()
                    lines.append(f"  {field_label:<25}: {value}")
        elif isinstance(service_data, list):
            for idx, item in enumerate(service_data, start=1):
                lines.append(f"\n  Entry {idx}:")
                if isinstance(item, dict):
                    for field, value in item.items():
                        if value is not None:
                            field_label = field.replace("_", " ").title()
                            lines.append(f"    {field_label:<23}: {value}")
                else:
                    lines.append(f"    {item}")

        lines.append("-" * 50)

    # lines.append("\nNote: Availability is subject to change. Please confirm at the earliest.")

    return "\n".join(lines)

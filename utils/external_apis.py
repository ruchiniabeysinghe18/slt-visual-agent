import requests

API_KEY = "AIzaSyDCV0pEE4ETKswgrsXq2Lbp1S72hze2Kkc"
COUNTRY_CODE = "LK"

def get_distance_matrix(origin, destination):
    """
    Calls Google Distance Matrix API and returns distance and duration data.

    Args:
        origin (str): Starting location
        destination (str): Destination location

    Returns:
        dict: Dictionary containing distance, duration text
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": f"{origin},{COUNTRY_CODE}",
        "destinations": f"{destination},{COUNTRY_CODE}",
        "key": API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["status"] == "OK":
        element = data["rows"][0]["elements"][0]
        if element["status"] == "OK":
            
            distance_text = element["distance"]["text"]
            parts = distance_text.strip().split()
            if len(parts) != 2:
                raise ValueError("Invalid distance format")
            
            value, unit = parts
            value = value.replace(",", "")
            
            distance_info = {
                "distance": int(float(value)),
                # "distance_value_meters": element["distance"]["value"],
                "duration": element["duration"]["text"],
                # "duration_value_seconds": element["duration"]["value"]
            }
            return distance_info
        else:
            return {"error": element["status"]}
    else:
        return {"error": data["status"]}


# result = get_distance_matrix("Katunayake", "Galle")
# print(result)

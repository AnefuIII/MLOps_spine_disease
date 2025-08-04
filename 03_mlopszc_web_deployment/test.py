import requests
import json # Good practice to import for JSON, though 'requests.post(json=...)' handles serialization

# Make sure this line is commented out. test.py is a client, not part of the Flask app.
# import predict

# The input data for the prediction
diag = {
    "pelvic_incidence": 54.92085752,
    "_pelvic_tilt": 18.96842952,
    "_lumbar_lordosis_angle": 51.60145541,
    "_sacral_slope": 35.952428,
    "_pelvic_radius": 125.8466462,
    "_grade_of_spondylolisthesis": 2.001642472,
}

# The URL of your Flask API endpoint
# It must include the specific route '/predict' as defined in your Flask app.
# Use 0.0.0.0 if Flask says it's running on 0.0.0.0, otherwise localhost is fine.
url = 'http://localhost:9696/predict'

print(f"Attempting to send request to: {url}")
print(f"Data to send: {diag}")

try:
    # Send the POST request with the JSON data
    # The 'json=diag' argument automatically sets the Content-Type header to application/json
    response = requests.post(url, json=diag)

    # Raise an HTTPError for bad responses (4xx or 5xx status codes)
    response.raise_for_status()

    # Parse the JSON response from the Flask API
    result = response.json()

    # Print the prediction result
    print("\nPrediction Result:")
    print(result)

except requests.exceptions.ConnectionError as e:
    # Catches errors if test.py cannot establish a connection with the Flask server
    print(f"\nError: Could not connect to the Flask server.")
    print(f"Please ensure predict.py is running and accessible at {url}.")
    print(f"Details: {e}")
except requests.exceptions.HTTPError as e:
    # Catches HTTP errors returned by the Flask server (e.g., 404 Not Found, 500 Internal Server Error)
    print(f"\nHTTP Error: The server returned an error response.")
    print(f"Status Code: {e.response.status_code}")
    print(f"Response Text: {e.response.text}")
    print(f"Details: {e}")
except requests.exceptions.RequestException as e:
    # Catches any other unexpected errors that occur during the request
    print(f"\nAn unexpected error occurred during the HTTP request: {e}")
except json.JSONDecodeError as e:
    # Catches errors if the server's response is not valid JSON
    print(f"\nError: Could not decode JSON response from the server.")
    print(f"Response content: {response.text if 'response' in locals() else 'No response received'}")
    print(f"Details: {e}")
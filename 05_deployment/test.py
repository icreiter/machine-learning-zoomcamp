import time

import requests

# --- Client Data from Question 4 ---
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
}
url = "http://127.0.0.1:9695/predict"


def test_prediction_endpoint(url, data):
    """Sends a POST request and prints the result."""
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes

        result = response.json()
        print("\n--- Prediction Result ---")
        print(result)

        # Extract and format the final answer
        probability = result.get("conversion_probability")
        if probability is not None:
            print(
                f"\nWhat's the probability that this client will get a subscription? -> {probability:.3f}"
            )

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server.")
        print("Please ensure 'predict_server.py' is running in a separate terminal.")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    # We add a small delay in case the server is just starting
    time.sleep(1)
    test_prediction_endpoint(url, client)

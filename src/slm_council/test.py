import requests
import time
import threading


def test_council_workflow():
    url = "http://127.0.0.1:8080/council/run"
    
    payload = {
        "query": "Create a java program to find if a string is palindrome or not",
        "language": "java",
        "max_iterations": 2,
    }

    print(f"Sending request to SLM Council...")
    print(f"Query: {payload['query']}\n")

    try:
        start_time = time.time()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(15):
                elapsed = time.time() - start_time
                print(f"Still processing... {elapsed:.1f}s elapsed")

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

        response = requests.post(url, json=payload, timeout=(10, 1800))
        stop_event.set()
        duration = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            
            print(f"Success! (Time: {duration:.2f}s)")
            print("-" * 50)
            print(f"Session ID: {result.get('session_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Refinement Passes: {result.get('refinement_passes')}")
            print(f"Summary: {result.get('summary')}")
            
            print("\n--- GENERATED CODE ---")
            code = result.get("code")
            files = code.get("files", []) if isinstance(code, dict) else []
            if files:
                for file in files:
                    print(f"File: {file.get('filename')}")
                    content = file.get("content", "")
                    print(content[:200] + "...")
            else:
                print("No code returned.")
            
            print("\n--- TEST REPORT ---")
            test_report = result.get("test_report")
            if isinstance(test_report, dict):
                print(f"Verdict: {test_report.get('verdict')}")
                print(f"Pass Count: {test_report.get('pass_count')}")
                print(f"Fail Count: {test_report.get('fail_count')}")
            else:
                print("No test report returned.")
            
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure uvicorn is running on port 8080.")
    except requests.exceptions.Timeout:
        print("Error: Request timed out on client side while the server may still be processing.")
        print("Tip: increase read timeout or reduce max_iterations for faster turnaround.")

if __name__ == "__main__":
    test_council_workflow()
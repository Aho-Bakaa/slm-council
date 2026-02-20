import json
import time

import requests


def test_council_workflow() -> None:
    url = "http://127.0.0.1:8080/council/run"

    payload = {
        "query": "Create a Python script that scrapes a website and saves it to a CSV",
        "language": "python",
    }

    print("Sending request to SLM Council...")
    print(f"Query: {payload['query']}\n")

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=180)
        duration = time.time() - start_time

        print(f"HTTP: {response.status_code} (Time: {duration:.2f}s)")

        if response.status_code != 200:
            print("Non-200 response")
            print(response.text)
            return

        result = response.json()
        print("-" * 60)
        print(f"Session ID: {result.get('session_id')}")
        print(f"Status: {result.get('status')}")
        print(f"Refinement Passes: {result.get('refinement_passes')}")
        print(f"Summary: {result.get('summary')}\n")

        code = result.get("code")
        if code and code.get("files"):
            print("--- GENERATED CODE ---")
            for file in code["files"]:
                content_preview = (file.get("content") or "")[:200]
                print(f"File: {file.get('filename')}")
                print(content_preview + ("..." if len(content_preview) == 200 else ""))
                print()
        else:
            print("No code produced in this run.")

        test_report = result.get("test_report")
        if test_report:
            print("--- TEST REPORT ---")
            print(f"Verdict: {test_report.get('verdict')}")
            print(f"Pass Count: {test_report.get('pass_count')}")
            print(f"Fail Count: {test_report.get('fail_count')}")
        else:
            print("No test report produced in this run.")

    except requests.exceptions.ConnectionError:
        print("Could not connect to the server. Is uvicorn running on port 8080?")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    test_council_workflow()

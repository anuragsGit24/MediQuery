"""Live integration tests for the Experiment 7 Healthcare RAG API.

Start the API first with: uvicorn app:app --reload
Then run: python test_live_api.py

This script tests the running service (unlike pytest tests that mock the RAG backend).
It saves a human-readable TXT report and a machine-readable JSON report.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 180
REPORT_DIR = Path(__file__).resolve().parent / "test_reports"
FALLBACK_TEXT = "the answer is not available in the provided documents."

results = []


def print_section(title):
    print("\n" + "=" * 88)
    print(f" {title.upper()} ")
    print("=" * 88)


def record_result(category, name, passed, details, duration_ms=None):
    result = {
        "category": category,
        "test": name,
        "result": "PASS" if passed else "FAIL",
        "details": details,
        "duration_ms": duration_ms,
    }
    results.append(result)
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}")
    if details:
        print(f"       {details}")
    if duration_ms is not None:
        print(f"       Test duration: {duration_ms:.2f} ms")


def timed_request(method, url, **kwargs):
    started = time.perf_counter()
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT_SECONDS)
    response = requests.request(method, url, timeout=timeout, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return response, elapsed_ms


def run_health_check():
    print_section("1. Service health and API availability")
    try:
        response, elapsed = timed_request("GET", f"{BASE_URL}/health", timeout=8)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        passed = response.status_code == 200 and body.get("status") == "ok"
        record_result(
            "HEALTH",
            "GET /health returns HTTP 200 and status=ok",
            passed,
            f"HTTP {response.status_code}; response={json.dumps(body, ensure_ascii=False)}",
            elapsed,
        )
        provider = body.get("generation_provider", "not exposed by this API version")
        print(f"       Generation provider: {provider}")

        docs_response, docs_elapsed = timed_request("GET", f"{BASE_URL}/openapi.json", timeout=8)
        docs_passed = docs_response.status_code == 200 and "paths" in docs_response.json()
        record_result(
            "HEALTH",
            "OpenAPI schema is available",
            docs_passed,
            f"HTTP {docs_response.status_code}; API docs should be at {BASE_URL}/docs",
            docs_elapsed,
        )
    except requests.RequestException as exc:
        record_result("HEALTH", "API server is reachable", False, str(exc))
        print("Start the server in another terminal: uvicorn app:app --reload")
        return False
    except (ValueError, json.JSONDecodeError) as exc:
        record_result("HEALTH", "Health response is valid JSON", False, str(exc))
        return False
    return True


def run_validation_tests():
    print_section("2. Input validation and malformed requests")
    cases = [
        ("Empty question", {"question": ""}, 422),
        ("Whitespace-only question", {"question": "     \t  "}, 422),
        ("Question exceeds 1000 characters", {"question": "x" * 1001}, 422),
        ("Missing required question field", {"query": "What is hydration?"}, 422),
        ("Question supplied as number", {"question": 12345}, 422),
        ("Null question", {"question": None}, 422),
    ]
    for name, payload, expected_status in cases:
        try:
            response, elapsed = timed_request("POST", f"{BASE_URL}/ask", json=payload)
            record_result(
                "VALIDATION",
                name,
                response.status_code == expected_status,
                f"HTTP {response.status_code}; expected HTTP {expected_status}; body={response.text[:240]}",
                elapsed,
            )
        except requests.RequestException as exc:
            record_result("VALIDATION", name, False, str(exc))

    try:
        response, elapsed = timed_request(
            "POST",
            f"{BASE_URL}/ask",
            data='{"question": ',
            headers={"Content-Type": "application/json"},
        )
        record_result(
            "VALIDATION",
            "Malformed JSON body is rejected",
            response.status_code == 422,
            f"HTTP {response.status_code}; expected HTTP 422; body={response.text[:240]}",
            elapsed,
        )
    except requests.RequestException as exc:
        record_result("VALIDATION", "Malformed JSON body is rejected", False, str(exc))


def run_relevant_questions():
    print_section("3. Experiment 5 knowledge-base questions")
    questions = [
        "What does the wellness note say about hydration?",
        "What are the signs of under-nutrition?",
        "What are some healthy-living tips regarding proteins?",
        "What is the turnaround time for cashless file submission to the TPA?",
        "What is the Provider Module?",
        "Where should the hospital submit claim documents physically?",
        "What is the National Health Claim Exchange (NHCX)?",
        "How is a medication error defined in the medication-safety document?",
        "What is polypharmacy?",
    ]

    for index, question in enumerate(questions, 1):
        name = f"Relevant KB question {index:02d}: {question}"
        try:
            response, elapsed = timed_request(
                "POST", f"{BASE_URL}/ask", json={"question": question}
            )
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if response.status_code != 200:
                record_result("RELEVANT_QA", name, False, f"HTTP {response.status_code}; body={response.text[:300]}", elapsed)
                continue

            answer = str(body.get("answer", "")).strip()
            sources = body.get("sources", [])
            grounded = body.get("grounded")
            source_names = sorted({s.get("file_name", "unknown") for s in sources if isinstance(s, dict)})
            valid_schema = bool(answer) and isinstance(sources, list) and isinstance(body.get("latency_ms"), (int, float))
            not_fallback = FALLBACK_TEXT not in answer.lower()
            has_source = len(source_names) > 0
            passed = valid_schema and grounded is True and not_fallback and has_source
            details = (
                f"HTTP 200; grounded={grounded}; source_files={source_names}; "
                f"server_latency_ms={body.get('latency_ms')}; "
                f"answer={answer[:180]}"
            )
            record_result("RELEVANT_QA", name, passed, details, elapsed)
        except (requests.RequestException, ValueError) as exc:
            record_result("RELEVANT_QA", name, False, str(exc))


def run_out_of_domain_tests():
    print_section("4. Out-of-domain questions and safe fallback")
    questions = [
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2022?",
        "How do you build a quantum computer?",
        "What is the recipe for chocolate chip cookies?",
        "Explain the plot of a popular science-fiction movie.",
    ]
    for index, question in enumerate(questions, 1):
        name = f"Out-of-domain question {index:02d}: {question}"
        try:
            response, elapsed = timed_request(
                "POST", f"{BASE_URL}/ask", json={"question": question}
            )
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            answer = str(body.get("answer", "")).strip()
            fallback = FALLBACK_TEXT in answer.lower()
            passed = (
                response.status_code == 200
                and body.get("grounded") is False
                and fallback
            )
            record_result(
                "OUT_OF_DOMAIN",
                name,
                passed,
                f"HTTP {response.status_code}; grounded={body.get('grounded')}; "
                f"fallback_detected={fallback}; answer={answer[:180]}",
                elapsed,
            )
        except (requests.RequestException, ValueError) as exc:
            record_result("OUT_OF_DOMAIN", name, False, str(exc))


def run_special_input_test():
    print_section("5. Special characters and injection-like input")
    question = "<script>alert('xss')</script> OR '1'='1' --"
    try:
        response, elapsed = timed_request(
            "POST", f"{BASE_URL}/ask", json={"question": question}
        )
        if response.status_code == 200:
            body = response.json()
            passed = isinstance(body.get("answer"), str) and bool(body.get("answer").strip())
            details = f"HTTP 200; API returned structured response; grounded={body.get('grounded')}; answer={body.get('answer', '')[:160]}"
        else:
            passed = response.status_code in (400, 422)
            details = f"HTTP {response.status_code}; body={response.text[:240]}"
        record_result("SECURITY_INPUT", "Script/SQL-like text does not crash the API", passed, details, elapsed)
    except (requests.RequestException, ValueError) as exc:
        record_result("SECURITY_INPUT", "Script/SQL-like text does not crash the API", False, str(exc))


def run_response_contract_test():
    print_section("6. Response schema and latency contract")
    question = "What is cashless hospitalization?"
    try:
        response, elapsed = timed_request("POST", f"{BASE_URL}/ask", json={"question": question})
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        required = {"answer", "sources", "best_distance", "latency_ms", "grounded"}
        missing = sorted(required - set(body.keys())) if isinstance(body, dict) else sorted(required)
        source_shape_ok = isinstance(body.get("sources"), list) and all(
            isinstance(source, dict)
            and {"file_name", "page", "excerpt"}.issubset(source.keys())
            for source in body.get("sources", [])
        )
        passed = (
            response.status_code == 200
            and not missing
            and isinstance(body.get("answer"), str)
            and isinstance(body.get("latency_ms"), (int, float))
            and body.get("latency_ms") >= 0
            and isinstance(body.get("grounded"), bool)
            and source_shape_ok
        )
        record_result(
            "CONTRACT",
            "Answer response contains required fields and valid source metadata",
            passed,
            f"HTTP {response.status_code}; missing_fields={missing}; source_shape_valid={source_shape_ok}; "
            f"server_latency_ms={body.get('latency_ms')}; total_client_latency_ms={elapsed:.2f}",
            elapsed,
        )
    except (requests.RequestException, ValueError) as exc:
        record_result("CONTRACT", "Answer response contains required fields and valid source metadata", False, str(exc))


def save_reports():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    passed = sum(item["result"] == "PASS" for item in results)
    failed = sum(item["result"] == "FAIL" for item in results)
    summary = {
        "project": "Experiment 7 - Healthcare Knowledge RAG API",
        "base_url": BASE_URL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "overall_result": "PASS" if failed == 0 and results else "FAIL",
        "note": "Live integration test results. Passing HTTP checks do not independently prove medical or semantic correctness; review answers and cited source documents manually.",
        "tests": results,
    }

    json_path = REPORT_DIR / "experiment7_live_test_report.json"
    txt_path = REPORT_DIR / "experiment7_live_test_report.txt"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "EXPERIMENT 7 - HEALTHCARE KNOWLEDGE RAG API",
        "LIVE INTEGRATION TEST REPORT",
        "=" * 72,
        f"Base URL: {BASE_URL}",
        f"Generated (UTC): {summary['generated_at_utc']}",
        f"Total tests: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        f"Overall result: {summary['overall_result']}",
        "",
    ]
    for item in results:
        lines.append(f"[{item['result']}] {item['category']} - {item['test']}")
        lines.append(f"    Details: {item['details']}")
        if item["duration_ms"] is not None:
            lines.append(f"    Client duration: {item['duration_ms']:.2f} ms")
        lines.append("")
    lines.extend([
        "INTERPRETATION",
        "- Relevant questions should return grounded answers and source metadata.",
        "- Out-of-domain questions should return the configured insufficient-information fallback.",
        "- Validation cases should be rejected with HTTP 422.",
        "- Read the returned answers and source excerpts manually before presenting the result as semantically correct.",
        "- This report records actual responses from the live server; it does not fabricate results.",
    ])
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print_section("7. Final summary")
    print(f"Total tests : {len(results)}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Result      : {summary['overall_result']}")
    print(f"Text report : {txt_path}")
    print(f"JSON report : {json_path}")
    return failed


def main():
    print("EXPERIMENT 7 - LIVE HEALTHCARE RAG API TEST SUITE")
    print(f"Target API: {BASE_URL}")
    print("Note: first real RAG question may take longer while models warm up.")

    if not run_health_check():
        save_reports()
        return 2

    run_validation_tests()
    run_relevant_questions()
    run_out_of_domain_tests()
    run_special_input_test()
    run_response_contract_test()
    failures = save_reports()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
EpiLongAI — Load test script.

Uses only Python stdlib (no locust or external deps required).

Usage:
    python scripts/load_test.py --url http://localhost:8000 --workers 10 --requests 100
    python scripts/load_test.py --url http://localhost:8000 --workers 20 --requests 500 --endpoint /health
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    status: int
    latency_ms: float
    error: str | None = None


@dataclass
class LoadTestReport:
    endpoint: str
    total_requests: int
    workers: int
    results: list[RequestResult] = field(default_factory=list)

    def print_report(self) -> None:
        successes = [r for r in self.results if r.error is None and 200 <= r.status < 300]
        errors = [r for r in self.results if r.error is not None or r.status >= 400]
        latencies = sorted(r.latency_ms for r in successes)

        total_time_s = max(r.latency_ms for r in self.results) / 1000 if self.results else 0

        print("\n" + "=" * 60)
        print(f"  EpiLongAI Load Test Report — {self.endpoint}")
        print("=" * 60)
        print(f"  Workers:          {self.workers}")
        print(f"  Total requests:   {self.total_requests}")
        print(f"  Successful:       {len(successes)}")
        print(f"  Failed:           {len(errors)}")
        print(f"  Error rate:       {len(errors) / self.total_requests * 100:.1f}%")
        print()

        if latencies:
            print(f"  Min latency:      {latencies[0]:.1f} ms")
            print(f"  Mean latency:     {statistics.mean(latencies):.1f} ms")
            print(f"  p50 latency:      {_percentile(latencies, 50):.1f} ms")
            print(f"  p95 latency:      {_percentile(latencies, 95):.1f} ms")
            print(f"  p99 latency:      {_percentile(latencies, 99):.1f} ms")
            print(f"  Max latency:      {latencies[-1]:.1f} ms")
        else:
            print("  No successful requests — cannot compute latency stats.")

        if total_time_s > 0:
            # Approximate throughput (wall-clock based on concurrent execution)
            print(f"\n  ~Requests/sec:    {len(successes) / total_time_s * 1000:.1f}")

        if errors:
            print(f"\n  Sample errors (first 5):")
            for e in errors[:5]:
                print(f"    status={e.status}  {e.error}")

        print("=" * 60 + "\n")


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile from pre-sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _make_request(url: str, method: str = "GET", body: bytes | None = None) -> RequestResult:
    """Execute a single HTTP request and return timing info."""
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()  # consume body
            latency = (time.perf_counter() - start) * 1000
            return RequestResult(status=resp.status, latency_ms=latency)
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(status=e.code, latency_ms=latency, error=str(e))
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(status=0, latency_ms=latency, error=str(e))


def _build_sample_payload() -> bytes:
    """Build a minimal synthetic payload for /predict_sample."""
    # 50 methylation features — synthetic values typical for ONT data
    features = [0.1 + (i % 10) * 0.08 for i in range(50)]
    payload = {"methylation_features": features}
    return json.dumps(payload).encode()


def run_load_test(
    base_url: str,
    endpoint: str,
    workers: int,
    num_requests: int,
) -> LoadTestReport:
    url = base_url.rstrip("/") + endpoint
    is_predict = "predict" in endpoint.lower()
    method = "POST" if is_predict else "GET"
    body = _build_sample_payload() if is_predict else None

    report = LoadTestReport(endpoint=endpoint, total_requests=num_requests, workers=workers)

    print(f"Running load test: {method} {url}")
    print(f"  {num_requests} requests with {workers} concurrent workers")

    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_make_request, url, method, body) for _ in range(num_requests)]
        for future in as_completed(futures):
            report.results.append(future.result())

    wall_time = time.perf_counter() - start_wall
    print(f"  Completed in {wall_time:.2f}s")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="EpiLongAI load test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Single endpoint to test (default: test both /health and /predict_sample)",
    )
    args = parser.parse_args()

    if args.endpoint:
        report = run_load_test(args.url, args.endpoint, args.workers, args.requests)
        report.print_report()
    else:
        # Test health endpoint
        health_report = run_load_test(args.url, "/health", args.workers, args.requests)
        health_report.print_report()

        # Test predict endpoint
        predict_report = run_load_test(args.url, "/predict_sample", args.workers, args.requests)
        predict_report.print_report()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Query Unpaywall for open-access status, links, and license metadata."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from threading import local

import requests

try:
    from .common import ProgressStore, load_tasks, run_batch, write_results
except ImportError:
    from common import ProgressStore, load_tasks, run_batch, write_results


FIELDS = ["is_oa", "oa_status", "oa_url", "license_raw", "license_category", "error"]
thread_state = local()


def classify_license(raw_license):
    value = str(raw_license or "").strip().lower()
    if not value:
        return "UNKNOWN"
    if re.search(r"(?<![a-z])(?:cc-?0|cc-?zero)(?![a-z0-9])", value):
        return "CC0"
    has_by = bool(re.search(r"(?<![a-z])cc[\s-]*by(?![a-z])", value))
    has_by = has_by or bool(re.search(r"creativecommons\.org/licenses/by(?:/|$)", value))
    has_by = has_by or "creative commons attribution" in value
    if not has_by:
        return "OTHER"

    parts = ["CC", "BY"]
    if re.search(r"(?<![a-z])(?:nc|non[\s-]*commercial)(?![a-z])", value):
        parts.append("NC")
    if re.search(r"(?<![a-z])(?:sa|share[\s-]*alike)(?![a-z])", value):
        parts.append("SA")
    if re.search(r"(?<![a-z])(?:nd|no[\s-]*deriv(?:ative)?s?)(?![a-z])", value):
        parts.append("ND")
    return "-".join(parts)


def extract_license(data):
    locations = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    locations.extend(item for item in data.get("oa_locations", []) if isinstance(item, dict))
    raw = next((str(item.get("license")).strip() for item in locations if item.get("license")), "")
    return raw, classify_license(raw)


def get_session():
    if not hasattr(thread_state, "session"):
        thread_state.session = requests.Session()
    return thread_state.session


def make_fetcher(email, timeout, retries, backoff, min_interval):
    def fetch(doi):
        session = get_session()
        headers = {"User-Agent": f"FTPD-CL/1.0 (mailto:{email})"}
        url = (
            "https://api.unpaywall.org/v2/"
            + requests.utils.quote(doi, safe="/:")
            + "?email="
            + requests.utils.quote(email)
        )
        last_error = ""
        if min_interval:
            time.sleep(min_interval)
        for attempt in range(retries):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                if response.status_code == 404:
                    return {
                        "is_oa": "0", "oa_status": "not_found", "oa_url": "",
                        "license_raw": "", "license_category": "UNKNOWN",
                        "error": "DOI not found",
                    }
                if response.status_code == 429:
                    last_error = "HTTP 429"
                elif response.ok:
                    data = response.json()
                    best = data.get("best_oa_location") or {}
                    raw, category = extract_license(data)
                    return {
                        "is_oa": str(int(bool(data.get("is_oa")))),
                        "oa_status": data.get("oa_status", "closed") or "closed",
                        "oa_url": best.get("url", "") or "",
                        "license_raw": raw,
                        "license_category": category,
                        "error": "",
                    }
                else:
                    last_error = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(backoff * (2 ** attempt))
        return {
            "is_oa": "0", "oa_status": "error", "oa_url": "",
            "license_raw": "", "license_category": "UNKNOWN",
            "error": last_error[:500],
        }
    return fetch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with a required doi column")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--progress-db", type=Path, help="SQLite checkpoint path")
    parser.add_argument("--email", default=os.getenv("UNPAYWALL_EMAIL", ""),
                        help="Email required by Unpaywall")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--backoff", type=float, default=5)
    parser.add_argument("--min-interval", type=float, default=0,
                        help="Delay before each request in each worker, in seconds")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.email:
        raise SystemExit("Unpaywall requires --email or UNPAYWALL_EMAIL")
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("--workers and --retries must be positive")
    unique_dois, tasks = load_tasks(args.input)
    db_path = args.progress_db or args.output.with_suffix(".sqlite")
    store = ProgressStore(db_path, FIELDS)
    run_batch(
        unique_dois,
        store,
        make_fetcher(
            args.email, args.timeout, args.retries, args.backoff, args.min_interval
        ),
        args.workers,
        args.retry_errors,
    )
    write_results(args.output, tasks, store, FIELDS)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

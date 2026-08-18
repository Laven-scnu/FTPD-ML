#!/usr/bin/env python3
"""Check DOI records for retraction-related metadata in Crossref."""

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


FIELDS = ["is_retracted", "retraction_reason", "error"]
RETRACTION_PREFIXES = (
    "retracted:", "retraction:", "withdrawn:", "removed:",
    "withdrawal:", "temporarily removed:",
)
thread_state = local()


def check_retraction(data):
    message = data.get("message", {}) if isinstance(data, dict) else {}
    if not isinstance(message, dict):
        return 0, ""
    reasons = []

    for entry in message.get("update-by", []) or []:
        if not isinstance(entry, dict):
            continue
        values = " ".join(
            str(entry.get(key, "")) for key in ("label", "type", "updated-type", "DOI")
        )
        if re.search(r"retract", values, re.I):
            reasons.append("update-by(retraction)")
            break
        if re.search(r"withdraw", values, re.I):
            reasons.append("update-by(withdrawal)")
            break

    titles = message.get("title", [])
    title = str(titles[0]).strip().lower() if titles else ""
    if any(title.startswith(prefix) for prefix in RETRACTION_PREFIXES):
        reasons.append("title_prefix")

    for assertion in message.get("assertion", []) or []:
        if not isinstance(assertion, dict):
            continue
        name = str(assertion.get("name", "")).lower()
        value = str(assertion.get("value", "")).lower()
        label = str(assertion.get("label", "")).lower()
        if name == "crossmark_status" and value in {"retracted", "withdrawn"}:
            reasons.append(f"crossmark({value})")
            break
        if "retract" in label or "withdraw" in label:
            reasons.append(f"crossmark_label({label[:50]})")
            break

    article_type = str(message.get("type", "")).lower()
    if article_type in {"retraction", "retracted-article"}:
        reasons.append("type(retraction)")

    relation = message.get("relation", {})
    if isinstance(relation, dict):
        for key, values in relation.items():
            if re.search(r"retract|withdraw", key, re.I) and values:
                reasons.append(f"relation({key})")
                break

    return (1, "; ".join(dict.fromkeys(reasons))) if reasons else (0, "")


def get_session():
    if not hasattr(thread_state, "session"):
        thread_state.session = requests.Session()
    return thread_state.session


def make_fetcher(email, timeout, retries, backoff):
    def fetch(doi):
        session = get_session()
        headers = {"User-Agent": f"FTPD-CL/1.0 (mailto:{email})"} if email else {}
        url = "https://api.crossref.org/works/" + requests.utils.quote(doi, safe="/:")
        last_error = ""
        for attempt in range(retries):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                if response.status_code == 404:
                    return {"is_retracted": "0", "retraction_reason": "", "error": "DOI not found"}
                if response.status_code == 429:
                    last_error = "HTTP 429"
                elif response.ok:
                    flag, reason = check_retraction(response.json())
                    return {"is_retracted": str(flag), "retraction_reason": reason, "error": ""}
                else:
                    last_error = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(backoff * (2 ** attempt))
        return {"is_retracted": "0", "retraction_reason": "", "error": last_error[:500]}
    return fetch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with a required doi column")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--progress-db", type=Path, help="SQLite checkpoint path")
    parser.add_argument("--email", default=os.getenv("CROSSREF_EMAIL", ""),
                        help="Contact email for the API User-Agent")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--backoff", type=float, default=2)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("--workers and --retries must be positive")
    unique_dois, tasks = load_tasks(args.input)
    db_path = args.progress_db or args.output.with_suffix(".sqlite")
    store = ProgressStore(db_path, FIELDS)
    run_batch(
        unique_dois,
        store,
        make_fetcher(args.email, args.timeout, args.retries, args.backoff),
        args.workers,
        args.retry_errors,
    )
    write_results(args.output, tasks, store, FIELDS)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""Shared CSV, SQLite checkpoint, and concurrent processing helpers."""

from __future__ import annotations

import csv
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

Task = Tuple[str, str]
Result = Dict[str, object]


def load_tasks(path: Path) -> Tuple[List[str], List[Task]]:
    """Load a CSV containing a required ``doi`` column."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "doi" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'doi' column")
        unique_dois: List[str] = []
        tasks: List[Task] = []
        seen = set()
        for row in reader:
            doi = (row.get("doi") or "").strip()
            if not doi:
                continue
            name = (row.get("markdown_name") or row.get("article") or "").strip()
            tasks.append((doi, name))
            if doi not in seen:
                unique_dois.append(doi)
                seen.add(doi)
    return unique_dois, tasks


class ProgressStore:
    """SQLite checkpoint store safe for concurrent workers."""

    def __init__(self, path: Path, columns: Iterable[str]):
        self.path = path
        self.columns = list(columns)
        self.lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        definitions = ", ".join(
            f'"{column}" TEXT NOT NULL DEFAULT ""' for column in self.columns
        )
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                f'CREATE TABLE IF NOT EXISTS progress '
                f'(doi TEXT PRIMARY KEY, {definitions}, queried_at TEXT NOT NULL)'
            )

    def get(self, doi: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM progress WHERE doi = ?", (doi,)
            ).fetchone()
        return dict(row) if row else None

    def save(self, doi: str, result: Result):
        values = [str(result.get(column, "")) for column in self.columns]
        names = ", ".join(f'"{column}"' for column in self.columns)
        placeholders = ", ".join("?" for _ in range(len(values) + 2))
        with self.lock, self._connect() as connection:
            connection.execute(
                f'INSERT OR REPLACE INTO progress '
                f'(doi, {names}, queried_at) VALUES ({placeholders})',
                [doi, *values, datetime.now().isoformat()],
            )

    def all(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM progress").fetchall()
        return {row["doi"]: dict(row) for row in rows}


def run_batch(
    dois: List[str],
    store: ProgressStore,
    fetch: Callable[[str], Result],
    workers: int,
    retry_errors: bool = False,
):
    pending = []
    for doi in dois:
        cached = store.get(doi)
        if cached is None or (retry_errors and cached.get("error")):
            pending.append(doi)
    if not pending:
        return

    print(f"Processing {len(pending)} DOI(s) with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, doi): doi for doi in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            doi = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            store.save(doi, result)
            print(f"[{index}/{len(pending)}] {doi}")


def write_results(path: Path, tasks: List[Task], store: ProgressStore, fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = store.all()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doi", "markdown_name", *fields])
        writer.writeheader()
        for doi, name in tasks:
            row = progress.get(doi, {})
            writer.writerow({
                "doi": doi,
                "markdown_name": name,
                **{field: row.get(field, "") for field in fields},
            })

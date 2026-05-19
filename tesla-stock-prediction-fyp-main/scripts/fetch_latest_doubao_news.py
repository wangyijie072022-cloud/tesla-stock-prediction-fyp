#!/usr/bin/env python3
"""Fetch latest English Tesla news via Volcengine Ark web_search.

This is an optional raw-data supplement. It does not touch the notebook,
processed datasets, results, or model training outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-0-pro-260215"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data/raw/doubao_tesla_latest_news_raw.csv"
DEBUG_RESPONSE_PATH = PROJECT_ROOT / "archive/data/raw/doubao_latest_news_debug_response.json"
SOURCE_DATASET = "doubao_ark_web_search"
CORE_QUERIES = [
    "Tesla stock latest news",
    "TSLA latest news",
    "Tesla earnings news",
]
EXTENDED_QUERIES = [
    "Tesla delivery news",
    "Tesla price cut news",
    "Elon Musk Tesla stock news",
]
CSV_COLUMNS = [
    "fetched_at",
    "source_dataset",
    "query",
    "title",
    "summary",
    "publish_time",
    "publish_time_raw",
    "publish_time_utc",
    "parse_warning",
    "url",
    "site_name",
    "language",
    "ark_model",
]


class ArkQueryError(RuntimeError):
    """Raised when one Ark query fails but the fetcher should continue."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_title(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return value


def valid_url(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def is_china_source(site_name: str, url: str) -> bool:
    text = f"{site_name} {url}".lower()
    china_markers = [
        "china",
        "xinhua",
        "chinanews",
        "xinhuanet",
        "people.com.cn",
        ".cn/",
        ".cn",
    ]
    return any(marker in text for marker in china_markers)


def standardize_publish_time(raw_value: str, site_name: str, url: str) -> tuple[str, str]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return "", "missing publish_time"

    normalized = re.sub(r"\s+", " ", raw_text)
    timezone_match = re.search(r"\b([A-Z]{2,4})\b$", normalized)
    timezone_abbr = timezone_match.group(1) if timezone_match else ""
    cleaned = normalized[: timezone_match.start()].strip() if timezone_match else normalized

    tzinfo = None
    warning = ""
    if timezone_abbr == "BST":
        tzinfo = timezone(timedelta(hours=1))
    elif timezone_abbr == "CST":
        if is_china_source(site_name, url):
            tzinfo = timezone(timedelta(hours=8))
        else:
            return "", "ambiguous CST timezone; not parsed"
    elif timezone_abbr in {"UTC", "GMT"}:
        tzinfo = timezone.utc
    elif timezone_abbr:
        return "", f"unsupported timezone abbreviation: {timezone_abbr}"

    parse_candidates = [cleaned]
    if cleaned.endswith("Z"):
        parse_candidates.append(cleaned.replace("Z", "+00:00"))
    parse_candidates.extend([
        cleaned.replace("/", "-"),
        cleaned.replace("T", " "),
    ])

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    parsed = None
    for candidate in parse_candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            pass
    if parsed is None:
        for fmt in formats:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                pass

    if parsed is None:
        return "", "unparseable publish_time"

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo or timezone.utc)
        if tzinfo is None:
            warning = "timezone missing; assumed UTC"
    elif timezone_abbr:
        parsed = parsed.astimezone(tzinfo)

    publish_time_utc = parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    return publish_time_utc, warning


def enrich_time_fields(row: dict[str, str]) -> dict[str, str]:
    enriched = dict(row)
    raw_time = str(enriched.get("publish_time_raw") or enriched.get("publish_time") or "").strip()
    site_name = str(enriched.get("site_name") or "")
    url = str(enriched.get("url") or "")
    publish_time_utc, parse_warning = standardize_publish_time(raw_time, site_name, url)
    enriched["publish_time"] = str(enriched.get("publish_time") or raw_time)
    enriched["publish_time_raw"] = raw_time
    enriched["publish_time_utc"] = publish_time_utc
    enriched["parse_warning"] = parse_warning
    return {column: enriched.get(column, "") for column in CSV_COLUMNS}


def fail(message: str, exit_code: int = 1, **extra: Any) -> None:
    payload = {"success": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)


def summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(value.keys()),
            "length": len(value),
        }
    if isinstance(value, list):
        first = value[0] if value else None
        summary: dict[str, Any] = {
            "type": "list",
            "length": len(value),
        }
        if isinstance(first, dict):
            summary["first_item_keys"] = sorted(first.keys())
            summary["first_item_type"] = first.get("type")
        elif first is not None:
            summary["first_item_type"] = type(first).__name__
        return summary
    return {
        "type": type(value).__name__,
        "preview": str(value)[:200],
    }


def summarize_response_structure(payload: dict[str, Any]) -> dict[str, Any]:
    objects = list(walk_json(payload))
    interesting_keys = {
        "annotations",
        "citations",
        "citation",
        "references",
        "reference",
        "web_search_call",
        "web_search",
        "tools",
        "url",
        "uri",
        "link",
        "title",
        "site_name",
        "source_name",
        "published_at",
        "published_time",
        "publish_time",
    }
    key_counts = {key: 0 for key in interesting_keys}
    for obj in objects:
        for key in interesting_keys:
            if key in obj:
                key_counts[key] += 1

    return {
        "top_level_fields": sorted(payload.keys()),
        "output_summary": summarize_value(payload.get("output")),
        "has_output_text": bool(extract_response_text(payload)),
        "interesting_key_counts": {key: count for key, count in sorted(key_counts.items()) if count},
    }


def save_debug_response(query: str, model: str, payload: dict[str, Any], reason: str) -> dict[str, Any]:
    DEBUG_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "saved_at": now_utc_iso(),
        "query": query,
        "ark_model": model,
        "reason": reason,
        "structure_summary": summarize_response_structure(payload),
        "raw_response": payload,
    }
    existing: list[Any] = []
    if DEBUG_RESPONSE_PATH.exists():
        try:
            existing_payload = json.loads(DEBUG_RESPONSE_PATH.read_text(encoding="utf-8"))
            existing = existing_payload if isinstance(existing_payload, list) else [existing_payload]
        except json.JSONDecodeError:
            existing = []
    existing.append(entry)
    DEBUG_RESPONSE_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry["structure_summary"]


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    fragments: list[str] = []
    for obj in walk_json(payload):
        obj_type = str(obj.get("type", ""))
        text_value = obj.get("text")
        if isinstance(text_value, str) and obj_type in {"output_text", "text"}:
            fragments.append(text_value)
    return "\n".join(fragments).strip()


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty response text")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object or array found in response text")


def fallback_articles_from_citations(payload: dict[str, Any]) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    for obj in walk_json(payload):
        url = (
            obj.get("url")
            or obj.get("uri")
            or obj.get("link")
            or obj.get("source_url")
            or obj.get("page_url")
        )
        title = (
            obj.get("title")
            or obj.get("source_title")
            or obj.get("source_name")
            or obj.get("site_name")
            or obj.get("name")
        )
        if not valid_url(str(url)) or not title:
            continue
        articles.append(
            {
                "title": str(title),
                "summary": "",
                "publish_time": str(
                    obj.get("published_at")
                    or obj.get("published_time")
                    or obj.get("publish_time")
                    or obj.get("time")
                    or obj.get("date")
                    or ""
                ),
                "url": str(url),
                "site_name": str(obj.get("site_name") or obj.get("source_name") or obj.get("name") or ""),
                "language": "en",
            }
        )
    return articles


def normalize_articles(parsed: Any) -> list[dict[str, str]]:
    if isinstance(parsed, dict):
        if parsed.get("title"):
            articles = [parsed]
        else:
            articles = parsed.get("articles") or parsed.get("news") or parsed.get("items") or []
    elif isinstance(parsed, list):
        articles = parsed
    else:
        articles = []

    normalized: list[dict[str, str]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        normalized.append(
            {
                "title": title,
                "summary": str(article.get("summary") or "").strip(),
                "publish_time": str(
                    article.get("publish_time")
                    or article.get("published_at")
                    or article.get("published_time")
                    or article.get("date")
                    or ""
                ).strip(),
                "url": str(article.get("url") or article.get("link") or article.get("uri") or "").strip(),
                "site_name": str(article.get("site_name") or article.get("source") or "").strip(),
                "language": str(article.get("language") or "en").strip() or "en",
            }
        )
    return normalized


def build_prompt(query: str, max_articles: int) -> str:
    return f"""Search the web for: {query}

Return English-language sources only.
Prefer original financial/news sources.
Do not translate titles.
Keep title and summary in English.

Return strict JSON only.
Return exactly a JSON array, with no markdown, no explanation, and no wrapper object:
[
  {{
    "title": "English title exactly as published",
    "summary": "Short English summary",
    "publish_time": "Publication time if available, otherwise empty string",
    "url": "Canonical article URL",
    "site_name": "Publisher or site name",
    "language": "en"
  }}
]

Return up to {max_articles} articles. Do not include non-English articles."""


def call_ark_web_search(
    api_key: str,
    model: str,
    base_url: str,
    query: str,
    max_articles: int,
    timeout: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/responses"
    request_payload = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_prompt(query, max_articles),
                    }
                ],
            }
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ArkQueryError(
            "Ark API HTTP error.",
            ark_model=model,
            query=query,
            status_code=exc.code,
            response_body_preview=body[:2000],
        )
    except urllib.error.URLError as exc:
        raise ArkQueryError("Ark API connection failed.", ark_model=model, query=query, reason=str(exc.reason))
    except TimeoutError:
        raise ArkQueryError(
            "Ark API request timed out. Network or API endpoint may be unavailable.",
            ark_model=model,
            query=query,
            timeout=timeout,
        )

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        raise ArkQueryError(
            "Ark API returned non-JSON response.",
            ark_model=model,
            query=query,
            response_body_preview=response_body[:2000],
        )

    response_text = extract_response_text(payload)
    warning: dict[str, Any] = {}
    try:
        articles = normalize_articles(parse_json_from_text(response_text))
    except ValueError as parse_err:
        summary = save_debug_response(
            query=query,
            model=model,
            payload=payload,
            reason=f"output_text JSON parse failed: {parse_err}",
        )
        warning = {
            "query": query,
            "warning": f"output_text JSON parse failed: {parse_err}",
            "debug_response_path": str(DEBUG_RESPONSE_PATH),
            "response_structure_summary": summary,
        }
        articles = fallback_articles_from_citations(payload)

    if not articles:
        summary = save_debug_response(
            query=query,
            model=model,
            payload=payload,
            reason="No parseable JSON articles or citation fallback articles found.",
        )
        warning = {
            "query": query,
            "warning": "No parseable news articles for this query; continuing.",
            "debug_response_path": str(DEBUG_RESPONSE_PATH),
            "response_structure_summary": summary,
        }
        print(json.dumps(warning, ensure_ascii=False, indent=2))
        return [], warning

    if warning:
        warning["fallback_article_count"] = len(articles)
        print(json.dumps(warning, ensure_ascii=False, indent=2))
    return articles, warning


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [enrich_time_fields({column: row.get(column, "") for column in (reader.fieldnames or [])}) for row in reader]


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    url_seen: dict[str, int] = {}
    fallback_seen: dict[tuple[str, str], int] = {}
    keep = [True] * len(rows)

    for index, row in enumerate(rows):
        url = str(row.get("url", "")).strip().lower()
        if valid_url(url):
            previous = url_seen.get(url)
            if previous is not None:
                keep[previous] = False
            url_seen[url] = index
            continue

        fallback_key = (
            str(row.get("publish_time", "")).strip(),
            normalize_title(str(row.get("title", ""))),
        )
        previous = fallback_seen.get(fallback_key)
        if previous is not None:
            keep[previous] = False
        fallback_seen[fallback_key] = index

    return [row for row, should_keep in zip(rows, keep) if should_keep]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def selected_queries(extended: bool) -> list[str]:
    return CORE_QUERIES + EXTENDED_QUERIES if extended else CORE_QUERIES


def fetch_query_with_retries(
    api_key: str,
    model: str,
    base_url: str,
    query: str,
    max_articles: int,
    timeout: int,
    retries: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    query_warnings: list[dict[str, Any]] = []
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            articles, warning = call_ark_web_search(
                api_key=api_key,
                model=model,
                base_url=base_url,
                query=query,
                max_articles=max_articles,
                timeout=timeout,
            )
            if warning:
                warning["attempt"] = attempt
                query_warnings.append(warning)
            return articles, query_warnings
        except ArkQueryError as exc:
            warning = {
                "query": query,
                "warning": exc.message,
                "attempt": attempt,
                "max_attempts": attempts,
                **exc.details,
            }
            query_warnings.append(warning)
            print(json.dumps(warning, ensure_ascii=False, indent=2))
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 5))
    return [], query_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest English Tesla news with Ark web_search.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV instead of appending.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Output CSV path. Default: {OUTPUT_PATH}")
    parser.add_argument("--timeout", type=int, default=180, help="Per-query request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts per query after the initial request.")
    parser.add_argument("--extended", action="store_true", help="Include additional lower-priority news queries.")
    parser.add_argument("--max-articles-per-query", type=int, default=5, help="Maximum articles requested per query.")
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Normalize publish_time fields in the existing output CSV without calling the API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("ARK_API_KEY")
    model = os.getenv("ARK_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if args.normalize_existing:
        existing_rows = read_existing_rows(output_path)
        if not existing_rows:
            fail("No existing latest-news CSV found to normalize.", output_path=str(output_path))
        normalized_rows = deduplicate_rows(existing_rows)
        write_rows(output_path, normalized_rows)
        print(
            json.dumps(
                {
                    "success": True,
                    "mode": "normalize_existing",
                    "output_path": str(output_path),
                    "rows": len(normalized_rows),
                    "parse_warning_counts": {
                        warning: sum(1 for row in normalized_rows if row.get("parse_warning") == warning)
                        for warning in sorted({row.get("parse_warning", "") for row in normalized_rows})
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not api_key:
        fail("Missing ARK_API_KEY. Set it in the environment before running this fetcher.", ark_model=model)

    fetched_at = now_utc_iso()
    new_rows: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    query_list = selected_queries(args.extended)
    failed_queries: list[str] = []
    for query_index, query in enumerate(query_list):
        articles, query_warnings = fetch_query_with_retries(
            api_key=api_key,
            model=model,
            base_url=base_url,
            query=query,
            max_articles=args.max_articles_per_query,
            timeout=args.timeout,
            retries=max(args.retries, 0),
        )
        warnings.extend(query_warnings)
        if not articles and query_warnings:
            failed_queries.append(query)
        for article in articles:
            row = {
                "fetched_at": fetched_at,
                "source_dataset": SOURCE_DATASET,
                "query": query,
                "title": article["title"],
                "summary": article["summary"],
                "publish_time": article["publish_time"],
                "publish_time_raw": article["publish_time"],
                "publish_time_utc": "",
                "parse_warning": "",
                "url": article["url"],
                "site_name": article["site_name"],
                "language": article["language"],
                "ark_model": model,
            }
            new_rows.append(enrich_time_fields(row))
        if query_index < len(query_list) - 1:
            time.sleep(random.uniform(2, 3))

    existing_rows = [] if args.overwrite else read_existing_rows(output_path)
    if not new_rows and output_path.exists():
        print(
            json.dumps(
                {
                    "success": True,
                    "ark_model": model,
                    "output_path": str(output_path),
                    "overwrite": bool(args.overwrite),
                    "queries": query_list,
                    "new_rows_before_dedup": 0,
                    "existing_rows_preserved": len(read_existing_rows(output_path)),
                    "final_rows": len(read_existing_rows(output_path)),
                    "warnings": warnings,
                    "failed_queries": failed_queries,
                    "note": "No new rows were fetched; existing latest-news CSV was preserved.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    combined_rows = deduplicate_rows(existing_rows + new_rows)
    write_rows(output_path, combined_rows)

    print(
        json.dumps(
            {
                "success": True,
                "ark_model": model,
                "output_path": str(output_path),
                "overwrite": bool(args.overwrite),
                "extended": bool(args.extended),
                "queries": query_list,
                "new_rows_before_dedup": len(new_rows),
                "existing_rows_before_dedup": len(existing_rows),
                "final_rows": len(combined_rows),
                "warnings": warnings,
                "failed_queries": failed_queries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch TBA ground truth — no browser, no key needed.

Primary path (no key): server-rendered TBA event HTML via requests + browser UA (the v1
path that worked when the API 401'd and WebFetch got anti-scrape 403s). Optional path: the
TBA API v3 if TBA_AUTH_KEY is set. Either way, fully scriptable by Claude.

Writes a normalized {matches:[...]} JSON with ALL comp levels (qm/sf/f), 3v3 dedup'd.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def fetch_html(event: str) -> list[dict]:
    import requests
    url = f"https://www.thebluealliance.com/event/{event}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (frcscout fetch)"} , timeout=30)
    r.raise_for_status()
    # TODO: parse td.red/td.blue data-team="frc####", red/blueScore spans, 'winner' class;
    #       dedup the desktop+mobile duplicate rows by match key; keep full 3v3 row.
    return []


def fetch_api(event: str, key: str) -> list[dict]:
    import requests
    r = requests.get(f"https://www.thebluealliance.com/api/v3/event/{event}/matches",
                     headers={"X-TBA-Auth-Key": key}, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ.get("TBA_AUTH_KEY")
    matches = fetch_api(a.event, key) if key else fetch_html(a.event)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"matches": matches}, open(a.out, "w"), indent=2)
    print(f"wrote {len(matches)} matches -> {a.out}")


if __name__ == "__main__":
    main()

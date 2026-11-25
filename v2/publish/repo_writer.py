"""Live metric publisher — incremental, idempotent, online.

Takes a per-match metrics dict from the aggregator and publishes it to a store that
can be served online. Design goals (RUBRIC_v2 D5):
  * incremental: callable repeatedly during a live match; each call upserts the
    latest snapshot for that match_key (in-progress snapshots are monotonic +
    timestamped, never duplicated).
  * idempotent: re-running a finished match overwrites cleanly (no dupes).
  * online: optional git push to a configured remote; degrades to local-only if no
    remote/creds (never crashes the live loop).
  * consumable: regenerates a human-readable index (README.md + index.html) from the
    store so a static host renders it.

No network is required for the core store; git push is best-effort and isolated.
Pure-stdlib (json/os/subprocess/time) so it has zero heavy deps and is unit-testable.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _utc_iso(ts: float) -> str:
    """ISO-8601 UTC from an epoch seconds value (caller supplies time; testable)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


@dataclass
class PublishConfig:
    store_dir: str                      # local repo working dir (the published store)
    git_remote: Optional[str] = None    # e.g. git@github.com:user/frc-live.git
    git_branch: str = "main"
    push: bool = False                  # attempt git push on publish (best-effort)
    author: str = "frc-vision-scout"
    # snapshot retention for in-progress matches (keep last N timestamped snapshots)
    keep_snapshots: int = 20


@dataclass
class PublishResult:
    match_key: str
    wrote: list[str] = field(default_factory=list)
    pushed: bool = False
    error: Optional[str] = None


class LivePublisher:
    """Upserts match-metric snapshots into a git-backed JSON store + dashboard."""

    def __init__(self, cfg: PublishConfig):
        self.cfg = cfg
        self.root = Path(cfg.store_dir)
        self.matches_dir = self.root / "matches"
        self.snaps_dir = self.root / "snapshots"
        self._git_ready = False

    # ---------------------------------------------------------------- store io
    def _ensure_store(self) -> None:
        self.matches_dir.mkdir(parents=True, exist_ok=True)
        self.snaps_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _read_index(self) -> dict:
        p = self._index_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {"matches": {}}
        return {"matches": {}}

    # ---------------------------------------------------------------- publish
    def publish(self, match_key: str, metrics: dict[str, Any], *,
                now: Optional[float] = None, final: bool = False) -> PublishResult:
        """Upsert a metrics snapshot for `match_key`.

        `metrics` is schema-agnostic (the aggregator owns the schema); we only add a
        publish envelope. `final=True` marks the match complete (stops snapshotting).
        `now` is injectable for deterministic tests.
        """
        res = PublishResult(match_key=match_key)
        try:
            self._ensure_store()
            ts = time.time() if now is None else now
            envelope = {
                "match_key": match_key,
                "updated_at": _utc_iso(ts),
                "final": bool(final),
                "metrics": metrics,
            }
            # 1) canonical latest-snapshot per match (idempotent upsert)
            latest = self.matches_dir / f"{match_key}.json"
            latest.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
            res.wrote.append(str(latest))

            # 2) monotonic timestamped snapshot trail (in-progress history)
            if not final:
                snap = self.snaps_dir / f"{match_key}__{int(ts)}.json"
                snap.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
                res.wrote.append(str(snap))
                self._prune_snapshots(match_key)

            # 3) index + dashboard regen
            self._update_index(match_key, envelope, ts)
            self._render_dashboard()
            res.wrote.append(str(self._index_path()))

            # 4) best-effort online push
            if self.cfg.push:
                res.pushed = self._git_publish(f"update {match_key} @ {envelope['updated_at']}")
        except Exception as e:  # never crash the live loop
            res.error = f"{type(e).__name__}: {e}"
        return res

    def _prune_snapshots(self, match_key: str) -> None:
        snaps = sorted(self.snaps_dir.glob(f"{match_key}__*.json"))
        excess = len(snaps) - self.cfg.keep_snapshots
        for p in snaps[:max(0, excess)]:
            try:
                p.unlink()
            except OSError:
                pass

    def _update_index(self, match_key: str, envelope: dict, ts: float) -> None:
        idx = self._read_index()
        m = idx.setdefault("matches", {})
        m[match_key] = {
            "match_key": match_key,
            "updated_at": envelope["updated_at"],
            "final": envelope["final"],
            "n_metrics": _count_metrics(envelope["metrics"]),
        }
        idx["generated_at"] = _utc_iso(ts)
        idx["count"] = len(m)
        self._index_path().write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")

    # ---------------------------------------------------------------- dashboard
    def _render_dashboard(self) -> None:
        idx = self._read_index()
        rows = sorted(idx.get("matches", {}).values(), key=lambda r: r["match_key"])
        # README.md
        lines = ["# FRC Vision Scout — Live Metrics", "",
                 f"_Auto-generated. {len(rows)} match(es)._", "",
                 "| Match | Updated (UTC) | Final | #Metrics |",
                 "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| `{r['match_key']}` | {r['updated_at']} | "
                         f"{'✓' if r['final'] else '…'} | {r['n_metrics']} |")
        (self.root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # index.html (self-contained, reads the per-match JSON on click)
        html = _DASHBOARD_HTML.replace("/*ROWS*/", json.dumps(rows))
        (self.root / "index.html").write_text(html, encoding="utf-8")

    # ---------------------------------------------------------------- git
    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.root), *args],
                              capture_output=True, text=True, timeout=60)

    def _ensure_git(self) -> bool:
        if self._git_ready:
            return True
        try:
            if not (self.root / ".git").exists():
                self._git("init")
                self._git("checkout", "-B", self.cfg.git_branch)
                if self.cfg.git_remote:
                    self._git("remote", "add", "origin", self.cfg.git_remote)
            self._git("config", "user.name", self.cfg.author)
            self._git("config", "user.email", f"{self.cfg.author}@local")
            self._git_ready = True
        except Exception:
            self._git_ready = False
        return self._git_ready

    def _git_publish(self, message: str) -> bool:
        if not self._ensure_git():
            return False
        try:
            self._git("add", "-A")
            c = self._git("commit", "-m", message)
            if c.returncode != 0 and "nothing to commit" in (c.stdout + c.stderr):
                return False
            if self.cfg.git_remote:
                p = self._git("push", "-u", "origin", self.cfg.git_branch)
                return p.returncode == 0
            return True
        except Exception:
            return False


def _count_metrics(metrics: Any) -> int:
    """Count leaf metric values in a (possibly nested) metrics dict."""
    if isinstance(metrics, dict):
        return sum(_count_metrics(v) for v in metrics.values())
    if isinstance(metrics, (list, tuple)):
        return sum(_count_metrics(v) for v in metrics)
    return 1


_DASHBOARD_HTML = """<!doctype html><meta charset=utf-8>
<title>FRC Vision Scout — Live</title>
<style>body{font:14px system-ui;margin:2rem;background:#0b0f14;color:#e6edf3}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #30363d;padding:.4rem .6rem}
th{background:#161b22}a{color:#58a6ff}</style>
<h1>FRC Vision Scout — Live Metrics</h1>
<table id=t><tr><th>Match<th>Updated (UTC)<th>Final<th>#Metrics</tr></table>
<script>const rows=/*ROWS*/||[];const t=document.getElementById('t');
for(const r of rows){const tr=t.insertRow();tr.innerHTML=
`<td><a href="matches/${r.match_key}.json">${r.match_key}</a><td>${r.updated_at}`+
`<td>${r.final?'\\u2713':'\\u2026'}<td>${r.n_metrics}`;}</script>
"""


if __name__ == "__main__":  # tiny offline self-test
    import tempfile
    d = tempfile.mkdtemp(prefix="frc_pub_")
    pub = LivePublisher(PublishConfig(store_dir=d, push=False))
    r1 = pub.publish("2026ctwat_qm1", {"score": {"red": 100, "blue": 90},
                                       "epa": {"red": 33.1}}, now=1_700_000_000, final=False)
    r2 = pub.publish("2026ctwat_qm1", {"score": {"red": 150, "blue": 95},
                                       "epa": {"red": 40.2}}, now=1_700_000_030, final=True)
    idx = json.loads((Path(d) / "index.json").read_text())
    assert idx["count"] == 1, idx
    assert idx["matches"]["2026ctwat_qm1"]["final"] is True
    latest = json.loads((Path(d) / "matches" / "2026ctwat_qm1.json").read_text())
    assert latest["metrics"]["score"]["red"] == 150
    assert (Path(d) / "index.html").exists() and (Path(d) / "README.md").exists()
    print("OK self-test:", d, "| wrote", len(r1.wrote), "+", len(r2.wrote), "files")

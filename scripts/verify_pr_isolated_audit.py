#!/usr/bin/env python3
"""Evaluate structured isolated-audit comments and publish an exact-head Check Run.

This script is intended to execute from trusted ``main`` code under
``pull_request_target`` / ``issue_comment``.  It never checks out or executes the
PR head.  The only head input is the SHA returned by GitHub's pull-request API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Workflow sets PYTHONPATH=src so no dependency installation or untrusted head
# package import is needed.
from quantagent.governance.github_audit_gate import (  # noqa: E402
    AUDIT_CHECK_NAME,
    evaluate_audit_comments,
)


API_VERSION = "2022-11-28"


class GitHubApiError(RuntimeError):
    pass


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "quantagent-isolated-audit-gate/1",
    }


def _api(
    method: str,
    url: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=_headers(token), method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API URL from workflow env
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubApiError(f"GitHub API {method} {url} failed {exc.code}: {detail[:1000]}") from exc
    except OSError as exc:
        raise GitHubApiError(f"GitHub API {method} {url} failed: {exc}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _event_pr_number(event: dict[str, Any]) -> int:
    if isinstance(event.get("pull_request"), dict):
        return int(event["pull_request"]["number"])
    issue = event.get("issue")
    if isinstance(issue, dict) and isinstance(issue.get("pull_request"), dict):
        return int(issue["number"])
    raise ValueError("event is not associated with a pull request")


def _list_issue_comments(api_root: str, repository: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urlencode({"per_page": 100, "page": page})
        batch = _api(
            "GET",
            f"{api_root}/repos/{repository}/issues/{pr_number}/comments?{query}",
            token=token,
        )
        if not isinstance(batch, list):
            raise GitHubApiError("issue comments endpoint did not return a list")
        comments.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            return comments
        page += 1
        if page > 100:
            raise GitHubApiError("audit comment pagination exceeded 10,000 comments")


def _existing_check_run(
    api_root: str,
    repository: str,
    head_sha: str,
    external_id: str,
    token: str,
) -> int | None:
    query = urlencode({"check_name": AUDIT_CHECK_NAME, "filter": "latest", "per_page": 100})
    payload = _api(
        "GET",
        f"{api_root}/repos/{repository}/commits/{head_sha}/check-runs?{query}",
        token=token,
    )
    if not isinstance(payload, dict):
        return None
    runs = payload.get("check_runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if isinstance(run, dict) and str(run.get("external_id", "")) == external_id:
            try:
                return int(run["id"])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _publish_check(
    *,
    api_root: str,
    repository: str,
    pr_number: int,
    head_sha: str,
    passed: bool,
    summary: str,
    token: str,
) -> None:
    external_id = f"quantagent-isolated-audit-pr-{pr_number}"
    body = {
        "name": AUDIT_CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "success" if passed else "failure",
        "external_id": external_id,
        "output": {
            "title": "Exact-head isolated audit accepted" if passed else "Exact-head isolated audit blocked",
            "summary": summary[:65000],
        },
    }
    existing = _existing_check_run(api_root, repository, head_sha, external_id, token)
    if existing is None:
        _api("POST", f"{api_root}/repos/{repository}/check-runs", token=token, payload=body)
    else:
        # Updating a Check Run does not accept head_sha/name fields in the same
        # way as creation; keep only mutable fields.
        mutable = {
            "status": body["status"],
            "conclusion": body["conclusion"],
            "output": body["output"],
        }
        _api(
            "PATCH",
            f"{api_root}/repos/{repository}/check-runs/{existing}",
            token=token,
            payload=mutable,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-path", default=os.getenv("GITHUB_EVENT_PATH", ""))
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--api-root", default=os.getenv("GITHUB_API_URL", "https://api.github.com"))
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    if not args.repository or "/" not in args.repository:
        print("GITHUB_REPOSITORY owner/name is required", file=sys.stderr)
        return 2
    if not args.event_path:
        print("GITHUB_EVENT_PATH is required", file=sys.stderr)
        return 2

    event = json.loads(Path(args.event_path).read_text(encoding="utf-8"))
    try:
        pr_number = _event_pr_number(event)
        pr = _api(
            "GET",
            f"{args.api_root}/repos/{args.repository}/pulls/{pr_number}",
            token=token,
        )
        if not isinstance(pr, dict) or not isinstance(pr.get("head"), dict):
            raise GitHubApiError("pull request endpoint did not return head metadata")
        head_sha = str(pr["head"].get("sha", "")).lower()
        comments = _list_issue_comments(args.api_root, args.repository, pr_number, token)
        evaluation = evaluate_audit_comments(comments, head_sha=head_sha)
        summary = evaluation.summary()
        _publish_check(
            api_root=args.api_root,
            repository=args.repository,
            pr_number=pr_number,
            head_sha=head_sha,
            passed=evaluation.passed,
            summary=summary,
            token=token,
        )
    except (ValueError, GitHubApiError, OSError, json.JSONDecodeError) as exc:
        print(f"isolated audit gate failed to evaluate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(summary)
    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

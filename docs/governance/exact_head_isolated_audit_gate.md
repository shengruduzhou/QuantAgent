# Exact-head isolated audit gate

This control closes the process gap recorded in issue #76.  Repository code
already knows how to decide whether isolated post-change reviews are sufficient;
this layer binds that decision to the exact GitHub pull-request head.

## Machine record

One logical reviewer publishes one maintainer-associated PR comment containing
exactly one marker block:

```text
<!-- quantagent-post-change-audit:v1
{"head_sha":"0123456789abcdef0123456789abcdef01234567","reviewer_role":"testing_expert","verdict":"approve","repo_wide":false,"evidence_checked":["CI run #123 exact head"],"notes":"optional"}
-->
```

Free-form prose is not parsed as approval.  Stale SHA records are ignored.
Malformed/unauthorised markers fail the gate.  Duplicate logical roles fail the
gate through `IsolatedAuditBoard`.

Required repository-owned policy remains:

- `testing_expert` approve;
- `quant_expert_tester` approve;
- `ai_quant_expert_auditor` approve with `repo_wide=true`;
- at least one independent domain-role approve;
- no `reject` or `needs_evidence` record.

The GitHub login that posts the record must have author association OWNER,
MEMBER, or COLLABORATOR.  This stops random public comments from creating
records, but does **not** claim that logical agent roles are cryptographically
distinct identities when one repository owner operates them.

## Trusted execution model

`.github/workflows/isolated-audit-gate.yml` runs on `pull_request_target` and
`issue_comment`.  It checks out only the trusted base/default branch and never
executes the PR head.  The workflow queries GitHub for the current PR `head.sha`,
parses structured comments, evaluates the repository-owned audit board, and
creates/updates the Check Run:

`isolated-multi-role-audit`

on that exact head SHA.  A new commit therefore cannot reuse an old approval set.

## Required one-time GitHub repository setting

The workflow/check is only an *enforcement mechanism* after GitHub is configured
to require it.  Repository administrators must protect `main` (branch protection
or a ruleset) and add `isolated-multi-role-audit` under **Require status checks
to pass before merging**.  Prefer strict/up-to-date mode if the repository's
merge cadence can support the extra reruns, and disable bypass for actors that
should not override the governance control.

The connected GitHub App used by the development agent currently receives HTTP
403 for the branch-protection endpoint, so this repository setting cannot be
truthfully asserted or changed by the current automation.  The setting must be
configured once by a repository administrator and then verified in GitHub.

GitHub requires a required status check to pass on the latest commit SHA.  That
property is intentionally paired with the exact-head audit record above.

## Threat boundary

This gate is designed to prevent accidental/process bypass such as "CI is green,
merge now" while a role still requests evidence.  It is materially stronger than
comment convention because the result is a machine check on the head commit.

It is not a malicious-owner-resistant signing system.  A repository administrator
can ultimately change branch rules, source code, or app permissions.  Protected
provenance/signer identity remains a separate trust layer.

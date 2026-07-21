# VN.PY Capability Parity Registry

## Purpose

This registry is the single machine-readable control plane for tracking how QuantAgent adopts VeighNa/vn.py platform capabilities without replacing QuantAgent's PIT data, strict A-share execution, model, risk, evidence and audit systems.

The canonical source is:

`services/quant_api/resources/vnpy_capability_parity.v1.json`

The Web workspace is available at:

`/parity`

The REST projection is:

`GET /api/system/vnpy-parity`

## Status semantics

- `not_audited`: no code-level equivalence audit has been completed.
- `missing`: no usable QuantAgent implementation was identified.
- `planned`: the adoption boundary and next action are defined.
- `in_progress`: an isolated vertical slice is under implementation.
- `partial`: some real capability exists, but the complete workflow or verification evidence is missing.
- `implemented`: the workflow exists, but one or more strict verification gates remain incomplete.
- `verified`: backend, typed contract, operable UI, real state change, realtime feedback, error states, tests, browser verification and reproducible evidence all pass.
- `blocked`: external credentials, infrastructure, safety or unresolved production-path conflicts prevent progress.
- `not_applicable`: an explicit, documented product decision excludes the capability.

A page, route, component, class or similarly named function is not sufficient evidence of implementation or verification.

## Update discipline

Each capability record must include:

- exact vn.py repository, module, version and commit when known;
- QuantAgent canonical modules, API, events, artifacts and frontend entry;
- current gap and adoption decision;
- tests, evidence and known limitations;
- one concrete next action.

Unknown details must remain explicit. Do not infer formula equivalence, plugin compatibility or production readiness from names or screenshots.

## Architecture boundaries

- The registry does not create a second runtime scanner, model registry, backtest engine, risk engine or event bus.
- The backend service validates and filters the JSON registry; it does not scan arbitrary source files per request.
- The frontend consumes the typed REST projection and does not embed a duplicate capability list.
- Live trading remains disabled. This workspace is governance and research control, not an order-entry surface.

## Validation commands

```bash
python -m pytest -q tests/quant_ui/test_vnpy_parity.py tests/quant_ui/test_api.py
python -m compileall -q services/quant_api

cd apps/quant-ui
npm test -- --run
npm run typecheck
npm run build
```

Start the API and UI, then open `/parity`. Browser verification must cover category/status filtering, search, row selection, inspector contents, empty results, API failure and layout overflow.

## Next vertical slice

The highest-priority follow-up is the DataManager query/download workflow: provider and dataset selection, symbol/date/interval filters, validated command creation, background job progress, typed events, manifest output, PIT/data-quality checks and Runtime Catalog refresh. It must extend the existing Runtime/Data workspace and existing job/event infrastructure rather than create a parallel data manager.

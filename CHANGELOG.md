# Changelog

All notable NanoCamelid changes are tracked here.

## [Unreleased]

- Made release packaging build the named release target explicitly
  (`aarch64-unknown-linux-gnu` by default), stage the target-scoped binary, and
  validate that dry-run contract.
- Promoted the local API server support matrix row to supported and expanded
  live validation coverage for structured POST errors on completion and chat
  endpoints, including output-cap enforcement.
- Added standard validation coverage for the live local API server contract:
  authenticated `/health`, `/v1/models`, `/metrics`, completion method errors,
  and structured not-found JSON responses on `127.0.0.1`.
- Switched local API completion request parsing to `serde_json` so malformed
  JSON, trailing data, escaped strings, arrays, and numeric fields follow normal
  JSON semantics before structured API validation errors are returned.
- Redacted API tokens from `nanocamelid serve --dry-run` replay plans while
  preserving an auth-enabled `NANOCAMELID_API_KEY='<token>'` command shape.
- Added `docs/CLI_CONTRACT.md` and tightened the README core CLI section around
  the stable v0.1 product commands, with compatibility commands documented
  separately; release packages now include the linked `docs/` tree.
- Added `scripts/install-systemd-user-service.sh`, release-package staging for
  the service installer, `docs/SERVICE_MODE.md`, and validation coverage for
  the loopback-only systemd user-service dry run.
- Wired `/v1/chat/completions` to bounded local chat-template generation with
  model id/alias/path resolution and OpenAI-shaped chat-completion JSON
  responses.
- Added an explicit local API `--max-request-bytes` cap, corresponding
  `NANOCAMELID_MAX_REQUEST_BYTES` env default, Prometheus metric, and
  structured `request_too_large` JSON errors for oversized HTTP requests.
- Added the `serve` CLI/API skeleton with default loopback binding, `/health`,
  `/v1/models`, `/metrics`, optional bearer-token auth, request/output cap
  settings, and structured JSON errors for unsupported completion endpoints.
- Added OpenAI-shaped request validation for `/v1/completions` and
  `/v1/chat/completions`, including required JSON fields and configured
  input/output cap enforcement before the deliberate not-implemented response.
- Added the stable non-loading `doctor` preflight and `models list`,
  `models scan`, and `models inspect` CLI namespace for model discovery and
  actionable model-directory errors.
- Reworked the repository front door into a short customer-facing README with a
  5-minute quickstart, install modes, core CLI examples, and links to detailed
  docs.
- Moved the long prototype history, Pi evidence, benchmark notes, and advanced
  launchers from the README into `docs/PRODUCT_HISTORY.md`.
- Started the v0.1 productization track: release packaging, install-from-release behavior, product README cleanup, stable CLI contract, local API server, service mode, observability, and support-matrix hardening.

## [0.1.0] - TBD

- Planned first versioned release for Raspberry Pi 5 and ARM64 edge devices.
- Release artifacts will include the `nanocamelid` binary, README quickstart, LICENSE, release notes, and SHA256 checksums.

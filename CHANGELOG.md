# Changelog

All notable NanoCamelid changes are tracked here.

## [Unreleased]

- Hardened the systemd user-service installer so generated unit values escape
  literal `%` characters before systemd can treat them as specifiers.
- Added `model_ready` and `next_action` to the local API `/health` response so
  local tools can distinguish an up server from a generation-ready model
  directory and show an actionable fix when no GGUF is discoverable.
- Added the configured request/input/output caps to the local API `/health`
  response so local tools can preflight server limits without scraping metrics.
- Hardened the release installer so `NANOCAMELID_INSTALL_MODE` must be an
  explicit `release`, `source`, or `dev` value instead of silently falling back
  to release behavior for typos.
- Added release-facing Cargo package metadata so the packaged binary has a
  public description, repository URL, README pointer, keywords, and category.
- Synced the public release notes validation checklist with the actual release
  workflow, including the full standard validation gate before packaging.
- Tightened the systemd user-service installer so model, unit, and config
  directory overrides must be absolute paths before a dry-run or install plan
  is accepted.
- Tightened local API method handling so every known endpoint path returns a
  structured `method_not_allowed` error for unsupported HTTP methods, while
  unknown paths remain structured `not_found` errors.
- Made the local API server drain a bounded oversized request body before
  returning `request_too_large`, and extended the live validation smoke to
  exercise that configured HTTP request byte cap.
- Tightened the stable `models inspect` namespace so it rejects missing,
  non-GGUF, or unknown shorthand model arguments before printing an inspect
  plan.
- Made the GitHub release workflow run the standard validation gate before
  packaging/uploading versioned aarch64 Linux artifacts, with local validation
  coverage for that release guard.
- Added local API response-status counters to `/metrics`, documented the
  Prometheus-style output, and extended the live API validation smoke to verify
  status-bucket metrics.
- Added `docs/API.md` as the public v0.1 local API reference covering server
  defaults, auth, request/output caps, `/health`, `/v1/models`,
  `/v1/completions`, `/v1/chat/completions`, `/metrics`, model id resolution,
  response shapes, and structured JSON errors.
- Made GitHub CI run the standard validation gate instead of only its dry-run
  plan, and wired GitHub releases to publish `RELEASE_NOTES.md` as the release
  body.
- Added `CHANGELOG.md` to versioned release archives and release install
  companion files so packaged installs carry both per-release notes and the
  project change history.
- Hardened `nanocamelid serve` and the systemd user-service installer so
  unauthenticated API serving remains loopback-only; non-loopback binds now
  require `--api-key` or `NANOCAMELID_API_KEY`.
- Added a packaged `VERSION` manifest and release-install verification that the
  archive manifest and `nanocamelid --version` output match the requested
  version before the binary is installed.
- Made model discovery expose active `1b`/`3b` aliases in `models list`,
  `models scan`, and `/v1/models`, with catalog docs for the default alias
  resolution rules.
- Made release installs persist the bundled README, docs, release notes, and
  service installer in a versioned companion directory while still installing
  the executable onto `PATH`.
- Added a release packaging guard that rejects `NANOCAMELID_VERSION` values
  that do not match the binary version in `Cargo.toml`, preventing versioned
  artifacts from disagreeing with `nanocamelid --version`.
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

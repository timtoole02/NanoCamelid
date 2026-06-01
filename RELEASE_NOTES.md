# NanoCamelid v0.1.0 Release Notes

## Summary

NanoCamelid v0.1.0 is the first product release of the Rust-native local AI runtime for Raspberry Pi 5 and ARM64 edge devices.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
```

## Included Artifacts

- `nanocamelid`
- `VERSION`
- `README.md`
- `docs/`
- `LICENSE`
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- `scripts/install-systemd-user-service.sh`
- `SHA256SUMS`

## Supported Models

See `docs/MODEL_CATALOG.md` and `docs/SUPPORT_MATRIX.md`.

## Known Limits

- Cluster mode remains experimental/labs until the single-node product path is complete.
- Streamed API responses are planned after the first non-streaming local API server.
- Service mode is supported for systemd user services; system services and launchd
  are not claimed in v0.1.

## Validation Checklist

- `./scripts/validate.sh`
- `cargo fmt -- --check`
- `cargo test`
- `cargo clippy --all-targets -- -D warnings`
- `cargo build --release --bins --target aarch64-unknown-linux-gnu`
- `./scripts/package-release.sh --dry-run`
- `./scripts/release-preflight.sh --check-remote --require-unpublished`
- `./scripts/validate.sh --dry-run`

GitHub releases run the standard validation gate before packaging the aarch64
Linux release archive, then publish the archive and `SHA256SUMS` from
`scripts/package-release.sh`. The final publish step is to tag and push the
validated commit named by `scripts/release-preflight.sh`.

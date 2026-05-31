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

- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo build --release --bins --target aarch64-unknown-linux-gnu`
- `./scripts/package-release.sh --dry-run`
- `./scripts/validate.sh --dry-run`

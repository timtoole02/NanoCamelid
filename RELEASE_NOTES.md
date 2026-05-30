# NanoCamelid v0.1.0 Release Notes

## Summary

NanoCamelid v0.1.0 is the first product release of the Rust-native local AI runtime for Raspberry Pi 5 and ARM64 edge devices.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
```

## Included Artifacts

- `nanocamelid`
- `README.md`
- `LICENSE`
- `RELEASE_NOTES.md`
- `SHA256SUMS`

## Supported Models

See `docs/MODEL_CATALOG.md` and `docs/SUPPORT_MATRIX.md`.

## Known Limits

- Cluster mode remains experimental/labs until the single-node product path is complete.
- Streamed API responses are planned after the first non-streaming local API server.

## Validation Checklist

- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo build --release --bins`
- `./scripts/validate.sh --dry-run`

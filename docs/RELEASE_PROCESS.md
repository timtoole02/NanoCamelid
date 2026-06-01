# NanoCamelid Release Process

This runbook keeps the v0.1 release path reproducible and public-safe. It
assumes the release version in `Cargo.toml`, `CHANGELOG.md`, and
`RELEASE_NOTES.md` already names the version being published.

## Preflight

Run the local release plan from a clean checkout:

```bash
./scripts/release-preflight.sh --dry-run
```

Before publishing a new tag, include the remote unpublished check:

```bash
./scripts/release-preflight.sh --check-remote --require-unpublished
```

The preflight verifies:

- the requested version matches `Cargo.toml`
- `CHANGELOG.md` has a matching version entry
- `RELEASE_NOTES.md` has matching release notes
- the working tree is clean for non-dry-run publication checks
- the local and optional remote release tag state is safe
- the packaging and installer dry-run contracts still resolve

## Publish

Use the exact `next_action` printed by the preflight. For a first publication,
that action has this shape:

```bash
git tag -a v0.1.0 -m 'NanoCamelid v0.1.0'
git push origin v0.1.0
```

Pushing the version tag starts the GitHub release workflow. Manual workflow
dispatch is reserved for retrying a release workflow for the same version after
inspecting the failed run.

## GitHub Workflow

The release workflow runs on the tagged commit and performs these gates before
uploading artifacts:

1. `./scripts/release-preflight.sh --dry-run`
2. `./scripts/validate.sh`
3. `rustup target add aarch64-unknown-linux-gnu`
4. `./scripts/package-release.sh`

The workflow publishes:

- `nanocamelid-v<version>-aarch64-unknown-linux-gnu.tar.gz`
- `SHA256SUMS`
- release body from `RELEASE_NOTES.md`

The archive contains `nanocamelid`, `VERSION`, `README.md`, `docs/`, `LICENSE`,
`CHANGELOG.md`, `RELEASE_NOTES.md`, and the systemd user-service installer.

## Post-Publish Smoke

After the GitHub release is visible, test the release installer on an ARM64
Linux host:

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | \
  bash -s -- --version v0.1.0
nanocamelid --version
nanocamelid doctor
nanocamelid serve --dry-run
```

The installer downloads the versioned archive, verifies `SHA256SUMS`, checks
the archive `VERSION` manifest, verifies `nanocamelid --version`, installs the
binary onto `PATH`, and keeps bundled docs in the versioned companion
directory.

## Public Hygiene

Do not add private hostnames, local IP addresses, personal filesystem paths,
key paths, credentials, raw remote commands, or raw operator stderr to release
notes, docs, changelogs, or GitHub release bodies. Keep machine-specific
validation notes in private operator logs and summarize only product-safe
evidence in public docs.

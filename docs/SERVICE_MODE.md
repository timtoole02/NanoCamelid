# NanoCamelid Service Mode

NanoCamelid v0.1 includes a systemd user-service installer for the local API
server. The default service is intentionally local-only:

- bind address: `127.0.0.1`
- port: `8080`
- model directory: `/mnt/nanocamelid/models`
- request cap: `65536` bytes
- input cap: `2048` tokens
- output cap: `256` tokens
- optional bearer-token auth through `NANOCAMELID_API_KEY`

## Install

Install NanoCamelid first. The release installer keeps the bundled service
installer with the release companion files. The default v0.1.0 path is:

```bash
$HOME/.local/share/nanocamelid/NanoCamelid/releases/v0.1.0-aarch64-unknown-linux-gnu/scripts/install-systemd-user-service.sh
```

From a checkout or installed release companion directory, inspect the service
plan:

```bash
nanocamelid --version
./scripts/install-systemd-user-service.sh --dry-run
```

Install the user service without starting it:

```bash
./scripts/install-systemd-user-service.sh
systemctl --user start nanocamelid.service
```

Install and start it in one step:

```bash
./scripts/install-systemd-user-service.sh --enable-now
```

To require bearer-token auth, pass a token at install time or set
`NANOCAMELID_API_KEY` in the environment before running the installer. The
installer writes the token to a user-owned `0600` EnvironmentFile:

```bash
./scripts/install-systemd-user-service.sh --api-key replace-with-a-token
```

For a direct `nanocamelid serve --dry-run`, auth-enabled plans keep the replay
command authenticated with a `NANOCAMELID_API_KEY='<token>'` placeholder instead
of echoing the configured token. Replace the placeholder in your shell before
starting the server:

```bash
NANOCAMELID_API_KEY=replace-with-a-token nanocamelid serve --dry-run
```

## Check

```bash
systemctl --user status nanocamelid.service
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

If auth is enabled, include `Authorization: Bearer <token>` on API requests.

## Defaults and Hardening

The generated unit runs `nanocamelid serve` with explicit loopback host, port,
model directory, request cap, input cap, and output cap values. It also enables
basic user-service hardening:

- restart on failure
- no new privileges
- private temporary directory
- read-only system paths
- read-only home access
- read-only model directory access
- address-family restriction to Unix, IPv4, and IPv6 sockets
- localhost-only systemd IP allowlist

The service does not configure system login lingering. On systems that should
start the user service at boot before login, enable lingering for the target
user with the host's normal system administration process.

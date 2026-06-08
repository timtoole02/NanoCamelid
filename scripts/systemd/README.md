# Pipeline-stage systemd user units

Keep the distributed NanoCamelid pipeline **stage servers** (`serve-stage`, listening on
`:9100`) alive across reboots and crashes. These are the per-node units for a multi-node
pipeline launched from `config/nodes.local.toml`.

This is distinct from [`../install-systemd-user-service.sh`](../install-systemd-user-service.sh),
which installs the **single-node HTTP API server** (`serve`, `:8080`).

Topology: `node0` (head) runs `generate-distributed` on demand and needs no unit;
`node1` / `node2` (and any further stages) run persistent `serve-stage` servers — those
are what these units keep running.

## Install (per stage node — no root required)

```sh
mkdir -p ~/.config/systemd/user
# on the node1 host:
cp nanocamelid-stage-node1.service ~/.config/systemd/user/nanocamelid-stage.service
loginctl enable-linger                         # start at boot without an interactive login
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user daemon-reload
systemctl --user enable --now nanocamelid-stage
```

Use `nanocamelid-stage-node2.service` on the node2 host, and so on. `%h` expands to the
service user's home, so the units are user-agnostic as long as the repo is deployed to
`~/nanocamelid` (the layout `deploy.sh` / `remote_build.sh` produce): binary at
`~/nanocamelid/target/release/nanocamelid` (`CARGO_TARGET_DIR`, see `env.sh`), source at
`~/nanocamelid/src/NanoCamelid`.

## Notes

- `Restart=always` recovers from crashes; `loginctl enable-linger` makes the stages come
  back after a power cycle. No passwordless sudo is needed — these are user units.
- Stop with `systemctl --user stop nanocamelid-stage`. To kill a stray *manual* process,
  match by name (`pkill -x nanocamelid`) — `pkill -f "nanocamelid serve-stage"` over SSH
  also matches your own shell and kills the session.

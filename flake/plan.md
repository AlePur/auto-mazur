# Deployment notes — Mazur

## GPU machine (one-time setup)

Install Tailscale and join your tailnet:
```sh
curl -fsSL https://tailscale.com/install.sh | sh
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /dev/null 2>&1 &
tailscale login --auth-key <YOUR_TAILSCALE_AUTH_KEY>
tailscale serve --bg --tcp=8000 tcp://localhost:8000 &
```

Install and start vLLM:
```sh
uv pip install https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl
uv pip install git+https://github.com/huggingface/transformers.git
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/tool_chat_template_gemma4.jinja -O tool_chat_template_gemma4.jinja

nohup vllm serve LilaRest/gemma-4-31B-it-NVFP4-turbo \
  --quantization modelopt \
  --max-model-len 180000 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template tool_chat_template_gemma4.jinja \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --host localhost \
  --port 8000 \
  --trust-remote-code &
```

Model card: https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo#compatibility

---

## EC2 (NixOS deploy)

### Architecture note

The Python source lives in the **same git repo** as this flake (`auto-mazur/`).
`flake/flake.nix` builds the `auto-mazur` package directly from `self` — no
separate git-URL or external input needed.

This means:
- No separate repo or `git+file://` hack required.
- `--build-host localhost` compiles `auto-mazur` locally and ships the closure.
- The resulting store path (`/nix/store/...-auto-mazur-0.1.0/`) is immutable:
  the `mazur` service user can execute the code but can never overwrite it.
- The `mazur` user is a normal user in `/home/mazur` with no restrictions,
  but has no sudo — the daemon is a system service only root can stop.
  `Restart=always` means even a self-kill is recovered automatically.

### Before first deploy

1. **Fill in your env file** — copy the example, edit your values, and source it
   before every nix command (`--impure` is required because the flake reads
   deployment values from environment variables at eval time):

   ```sh
   cp flake/env.sh.example flake/env.sh
   $EDITOR flake/env.sh          # set MAZUR_GPU_IP, MAZUR_MODEL, MAZUR_SSH_KEY …
   source flake/env.sh
   ```

   `flake/env.sh` is in `.gitignore` — **never commit it**.

2. Lock the flake (only needed once, or after input changes):

   ```sh
   cd flake/
   nix flake lock --impure
   cd ..
   ```

3. Stage the Tailscale auth key — nixos-anywhere injects it via `--extra-files`,
   so **no manual SSH step is needed on first boot**:

   Get a reusable auth key from https://login.tailscale.com/admin/settings/keys, then:

   ```sh
   mkdir -p flake/secrets/etc/tailscale
   echo 'tskey-auth-XX' > flake/secrets/etc/tailscale/authkey
   chmod 600 flake/secrets/etc/tailscale/authkey
   ```

   `flake/secrets/` is in `.gitignore` — **never commit it**.

### First deploy (bootstraps NixOS from scratch)

nixos-anywhere copies `secrets/` onto the target before NixOS activates, so
Tailscale finds `/etc/tailscale/authkey` on first boot without any prior SSH.

```sh
source flake/env.sh
cd flake/
nix run github:nix-community/nixos-anywhere -- --flake .#aws-agent --extra-files ./secrets --impure claw
```

### Subsequent config changes

```sh
source flake/env.sh
cd flake/
nixos-rebuild switch --flake .#aws-agent --impure --target-host claw
```

### Checking the agent

```sh
# View live logs
ssh root@<EC2-IP> journalctl -u mazur -f

# Check service status
ssh root@<EC2-IP> systemctl status mazur

# Drop into the mazur user's shell (admin only)
ssh root@<EC2-IP> machinectl shell mazur@
```

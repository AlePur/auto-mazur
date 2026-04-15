# flake/flake.nix — Mazur autonomous agent AWS NixOS deployment
#
# Usage (from the flake/ directory):
#
#   # First deploy onto a freshly provisioned NixOS instance:
#   nix run github:nix-community/nixos-anywhere -- \
#     --flake .#aws-agent --extra-files ./secrets root@<EC2-IP>
#
#   # Subsequent config changes:
#   nixos-rebuild switch \
#     --flake .#aws-agent \
#     --target-host root@<EC2-IP> \
#     --build-host localhost
#
# Before deploying:
#   1. Copy flake/env.sh.example → flake/env.sh and fill in your values
#   2. source flake/env.sh
#   3. Place your Tailscale auth key at flake/secrets/etc/tailscale/authkey
#   4. All nix commands must be run with --impure (env vars require it)
{
  description = "Mazur autonomous LLM agent — AWS NixOS deployment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";

    # disko: declarative disk partitioning — required by nixos-anywhere
    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, disko, ... }:
  let
    system = "x86_64-linux";
    pkgs   = import nixpkgs { inherit system; };
    lib    = nixpkgs.lib;

    # ---------------------------------------------------------------------------
    # Deployment-specific values — read from environment variables at eval time.
    #
    # These are NOT hardcoded here so the flake is safe to publish.
    # Before running any nix command, source your local env file:
    #
    #   source flake/env.sh      # copy from flake/env.sh.example and fill in
    #   nix flake lock --impure  # --impure is required for builtins.getEnv
    #
    # ---------------------------------------------------------------------------
    getEnv    = var: def: let v = builtins.getEnv var; in if v == "" then def else v;
    getEnvInt = var: def: let v = builtins.getEnv var;
                          in if v == "" then def else lib.toInt v;

    # GPU machine running vLLM over Tailscale (no API authentication needed)
    gpuIp    = getEnv    "MAZUR_GPU_IP"       "UNSET_GPU_IP";
    model    = getEnv    "MAZUR_MODEL"         "UNSET_MODEL";
    ctxLen   = getEnvInt "MAZUR_CONTEXT_LENGTH" 180000;

    # Admin SSH public key for root login
    sshKey   = getEnv "MAZUR_SSH_KEY" "";

    # Build auto-mazur directly from this repo.
    #
    # `self` refers to the git root of the auto-mazur repo (the directory that
    # contains pyproject.toml, src/, flake/, etc.).  Because this flake lives
    # inside the same git repo as the Python source, there is no separate
    # git-URL hack needed — Nix copies the whole tracked tree into the store
    # and compiles the package there.  The resulting store path is immutable:
    # the mazur service user can run the code but can never overwrite it.
    mazurPkg = pkgs.python3Packages.buildPythonApplication {
      pname   = "auto-mazur";
      version = "0.1.0";

      # The git root contains pyproject.toml and the src/ package.
      # Because this flake lives in the flake/ subdirectory, self.outPath
      # resolves to the flake/ dir inside the store copy of the repo.
      # dirOf self.outPath gives the repo root where pyproject.toml lives.
      src = /. + builtins.unsafeDiscardStringContext (dirOf self.outPath);

      pyproject = true;

      build-system = with pkgs.python3Packages; [
        setuptools
      ];

      dependencies = with pkgs.python3Packages; [
        httpx
        pyyaml
      ];

      # Tests require a live database and running agent; skip during Nix build.
      doCheck = false;
    };

  in {
    # -------------------------------------------------------------------------
    # NixOS configuration for the AWS EC2 instance
    # -------------------------------------------------------------------------
    nixosConfigurations.aws-agent = nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        # Declarative disk partitioning (required by nixos-anywhere)
        disko.nixosModules.disko

        # The Mazur service module
        ./mazur-service.nix

        ({ ... }: {
          # ------------------------------------------------------------------
          # Mazur service — driven by env vars from flake/env.sh
          # ------------------------------------------------------------------
          services.mazur = {
            enable              = true;
            package             = mazurPkg;
            vllmUrl             = "http://${gpuIp}:8000";
            model               = model;
            contextLengthTokens = ctxLen;
          };

          # ------------------------------------------------------------------
          # Tailscale — connects this EC2 to the GPU machine running vLLM
          # ------------------------------------------------------------------
          services.tailscale = {
            enable      = true;
            # Auth key provisioned out-of-band at /etc/tailscale/authkey.
            # Never embed the key directly here; the Nix store is world-readable.
            authKeyFile = "/etc/tailscale/authkey";
          };

          # Trust Tailscale traffic; open the WireGuard UDP port
          networking.firewall.trustedInterfaces = [ "tailscale0" ];
          networking.firewall.allowedUDPPorts   = [ 41641 ];

          # ------------------------------------------------------------------
          # Disk layout (disko — used by nixos-anywhere on first deploy)
          # nvme0n1p1: 1 MB GRUB BIOS-boot partition (type EF02)
          # nvme0n1p2: remainder as ext4 root
          # ------------------------------------------------------------------
          disko.devices.disk.main = {
            device = "/dev/nvme0n1";
            type   = "disk";
            content = {
              type = "gpt";
              partitions = {
                bios = {
                  size     = "1M";
                  type     = "EF02";   # GRUB BIOS-boot — no filesystem, no mountpoint
                  priority = 1;
                };
                root = {
                  size    = "100%";
                  content = {
                    type       = "filesystem";
                    format     = "ext4";
                    mountpoint = "/";
                  };
                };
              };
            };
          };

          boot.loader.grub.enable = true;

          # ------------------------------------------------------------------
          # Basic system
          # ------------------------------------------------------------------
          system.stateVersion = "25.11";   # Match your NixOS AMI version

          # Nix flakes + allow mazur user to use the Nix daemon
          nix.settings.experimental-features = [ "nix-command" "flakes" ];
          nix.settings.allowed-users         = [ "mazur" ];

          # Weekly garbage collection
          nix.gc = {
            automatic  = true;
            dates      = "weekly";
            options    = "--delete-older-than 14d";
          };

          # ------------------------------------------------------------------
          # SSH — key-only, no passwords
          # ------------------------------------------------------------------
          services.openssh = {
            enable = true;
            settings = {
              PasswordAuthentication = false;
              PermitRootLogin        = "prohibit-password";
            };
          };

          # Loaded from MAZUR_SSH_KEY in flake/env.sh (gitignored).
          users.users.root.openssh.authorizedKeys.keys =
            nixpkgs.lib.optionals (sshKey != "") [ sshKey ];

          # ------------------------------------------------------------------
          # Locale / timezone
          # ------------------------------------------------------------------
          time.timeZone = "UTC";
          i18n.defaultLocale = "en_US.UTF-8";
        })
      ];
    };
  };
}

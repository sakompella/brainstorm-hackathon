{
  description = "BrainStorm 2026 Track 2 - Neural Data Viewer";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    systems,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs (import systems);

    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };

    editableOverlay = workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

    pythonSets = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
        # Override evdev to use pre-built wheel (avoids kernel header requirement)
        evdevOverlay = final: prev: {
          evdev = prev.evdev.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
            # Point to linux headers for building
            preBuild = (old.preBuild or "") + ''
              export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include:$C_INCLUDE_PATH"
            '';
          });
        };
      in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
        (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
            evdevOverlay
          ]
        )
    );
  in {
    devShells = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonSet = pythonSets.${system}.overrideScope editableOverlay;
        virtualenv = pythonSet.mkVirtualEnv "brainstorm-dev-env" workspace.deps.all;
      in {
        default = pkgs.mkShell {
          packages = [
            virtualenv
            pkgs.uv
          ];
          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = pythonSet.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
          '';
        };
      }
    );

    packages = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonSet = pythonSets.${system};
        venv = pythonSet.mkVirtualEnv "brainstorm-env" workspace.deps.default;

        frontend = pkgs.stdenvNoCC.mkDerivation {
          name = "brainstorm-frontend";
          src = ./frontend;
          installPhase = ''
            mkdir -p $out
            cp -r $src/* $out/
          '';
        };
        
        brainstorm-py-app = name:
          pkgs.writeShellScriptBin "${name}" ''
            exec ${venv}/bin/${name} "$@"
          '';
        brainstorm-stream = brainstorm-py-app "brainstorm-stream";
        brainstorm-backend = brainstorm-py-app "brainstorm-backend";

        brainstorm-all = pkgs.writeShellScriptBin "brainstorm-all" ''
          export BRAINSTORM_STATIC_DIR="${frontend}"
          exec ${venv}/bin/brainstorm-all "$@"
        '';
      in {
        inherit frontend brainstorm-backend brainstorm-stream brainstorm-all;
        default = brainstorm-all;
      }
    );

    formatter = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in
        pkgs.writeShellScriptBin "alejandra-wrapped" ''
          ${pkgs.alejandra}/bin/alejandra .
        ''
    );
  };
}

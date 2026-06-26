{ stdenv, lib, bun2nix, ... }:
stdenv.mkDerivation {
  pname = "brainstorm-frontend";
  version = "0.0.0";

  src = ../frontend;

  nativeBuildInputs = [
    bun2nix.hook
  ];

  bunDeps = bun2nix.fetchBunDeps {
    bunNix = ../frontend/bun.nix;
  };

  # Use hoisted linker + hardlink backend to avoid EPERM in Nix sandbox
  bunInstallFlags =
    if stdenv.hostPlatform.isDarwin
    then ["--linker=hoisted" "--backend=copyfile"]
    else ["--linker=hoisted" "--backend=symlink"];

  buildPhase = ''
    bun run build
  '';

  installPhase = ''
    mkdir -p $out
    cp -r dist/* $out/
  '';
}

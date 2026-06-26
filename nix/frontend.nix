{ stdenv, bun2nix, ... }:
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

  buildPhase = ''
    bun run build
  '';

  installPhase = ''
    mkdir -p $out
    cp -r dist/* $out/
  '';
}

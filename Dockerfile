# From https://mitchellh.com/writing/nix-with-dockerfiles
# Nix builder
FROM nixos/nix:latest AS builder

# Add nix-community cache for bun2nix
RUN mkdir -p /etc/nix && \
    echo 'extra-substituters = https://nix-community.cachix.org' >> /etc/nix/nix.conf && \
    echo 'extra-trusted-public-keys = nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=' >> /etc/nix/nix.conf

# Copy our source and setup our working dir.
COPY . /tmp/build
WORKDIR /tmp/build

RUN nix \
    --extra-experimental-features "nix-command flakes" \
    # filter-syscalls option lets Apple Silicon cross-compile to Intel
    --option filter-syscalls false \
    build nixpkgs#git

RUN mv result git-result

# Build CA certificates (needed for HTTPS downloads at runtime via huggingface_hub)
RUN nix \
    --extra-experimental-features "nix-command flakes" \
    --option filter-syscalls false \
    build nixpkgs#cacert

RUN mv result cacert-result

# Build our Nix environment
RUN nix \
    --extra-experimental-features "nix-command flakes" \
    # filter-syscalls option lets Apple Silicon cross-compile to Intel
    --option filter-syscalls false \
    build

# Copy the Nix store closure into a directory. The Nix store closure is the
# entire set of Nix store values that we need for our build.
RUN mkdir /tmp/nix-store-closure
RUN cp -R $(nix-store -qR result/) $(nix-store -qR git-result/) $(nix-store -qR cacert-result/) /tmp/nix-store-closure

# Create the minimal filesystem structure that scratch lacks:
#   /tmp         - needed for temp files
#   /etc         - needed for nsswitch, passwd, and SSL certs
#   /app/data    - writable data directory for downloaded datasets
RUN mkdir -p /tmp/rootfs/tmp \
             /tmp/rootfs/etc/ssl/certs \
             /tmp/rootfs/app/data \
             /tmp/rootfs/root

# Minimal /etc/passwd: root user + nobody (some Python libs call getpwuid)
RUN printf 'root:x:0:0:root:/root:/bin/sh\nnobody:x:65534:65534:nobody:/:/bin/false\n' \
    > /tmp/rootfs/etc/passwd

# Minimal /etc/group
RUN printf 'root:x:0:\nnobody:x:65534:\n' \
    > /tmp/rootfs/etc/group

# nsswitch.conf: use files only (no NIS/LDAP) — required for getpwuid to work
RUN printf 'passwd: files\ngroup: files\nhosts: files dns\n' \
    > /tmp/rootfs/etc/nsswitch.conf

# Point SSL cert bundle to the cacert store path (symlink to the real file)
RUN ln -s $(readlink -f cacert-result/etc/ssl/certs/ca-bundle.crt) \
         /tmp/rootfs/etc/ssl/certs/ca-certificates.crt

# Final image is based on scratch. We copy a bunch of Nix dependencies
# but they're fully self-contained so we don't need Nix anymore.
FROM scratch

WORKDIR /app

# Copy /nix/store
COPY --from=builder /tmp/nix-store-closure /nix/store
COPY --from=builder /tmp/build/result /app
COPY --from=builder /tmp/build/git-result /git
COPY --from=builder /tmp/build/cacert-result /cacert

# Filesystem structure: /tmp, /etc (with certs, passwd, nsswitch), /app/data
COPY --from=builder /tmp/rootfs /

ENV PATH="/git/bin:/app/bin:${PATH}"
# Point Python's SSL stack to our CA bundle
ENV SSL_CERT_FILE="/cacert/etc/ssl/certs/ca-bundle.crt"
ENV NIX_SSL_CERT_FILE="/cacert/etc/ssl/certs/ca-bundle.crt"
ENV REQUESTS_CA_BUNDLE="/cacert/etc/ssl/certs/ca-bundle.crt"
ENV CURL_CA_BUNDLE="/cacert/etc/ssl/certs/ca-bundle.crt"
ENV HOME="/root"
ENV HF_HOME="/tmp/hf_cache"
ENV CONTAINER=1
EXPOSE 8000
# /app/data is where downloaded datasets land (repo_root falls back to cwd=/app)
VOLUME ["/app/data"]
CMD ["/app/bin/brainstorm-all"]

#!/usr/bin/env bash
set -euo pipefail

DOCKER_CMD="${DOCKER_CMD:-$(command -v podman || command -v docker || true)}"
if [ -z "$DOCKER_CMD" ]; then
  echo "Error: podman or docker command is required."
  exit 1
fi

rm -f volpe_img.tar
"$DOCKER_CMD" build -t volpe_grpc_test .
"$DOCKER_CMD" save -o volpe_img.tar volpe_grpc_test

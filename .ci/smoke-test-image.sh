#!/usr/bin/env bash
# Smoke-test a built image: the CLI starts and reports the commit it was built from.
# Usage: .ci/smoke-test-image.sh <image-ref> <expected-git-sha>
set -euo pipefail

IMAGE=${1:?usage: smoke-test-image.sh <image-ref> <expected-git-sha>}
EXPECTED_SHA=${2:?usage: smoke-test-image.sh <image-ref> <expected-git-sha>}

# Run like production: read-only root filesystem, no capabilities, no network.
RUN=(docker run --rm --read-only --cap-drop ALL --network none)

"${RUN[@]}" "$IMAGE" --help > /dev/null

ACTUAL_SHA=$("${RUN[@]}" --entrypoint python "$IMAGE" \
    -c 'from battery_optimizer.manifest import code_version; print(code_version())')

if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "code_version() is '$ACTUAL_SHA', expected '$EXPECTED_SHA'" >&2
    exit 1
fi
echo "OK: $IMAGE reports code_version() $ACTUAL_SHA"

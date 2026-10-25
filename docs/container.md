# Container image

Multi-arch (`linux/amd64`, `linux/arm64`) images are published to
`ghcr.io/pando85/battery-optimizer` with the tags `latest` (main), `sha-<short-sha>` (main),
`X.Y.Z` and `X.Y` (releases). The image runs as uid/gid `65532`, works with a read-only root
filesystem, and contains no data and no git.

A one-shot run of the batch commands, with the data directory and tariff config mounted and the token passed
by name, is in the [README quick start](../README.md#quick-start).

## Shadow service

The shadow service runs from the same image (`serve`, port 8080, read-only root filesystem, writes
only under the data directory); see [service.md](service.md):

```bash
docker run --rm --read-only --cap-drop ALL -p 8080:8080 \
  -v "$HOME/data/energy-baseline:/data" -v "$PWD/config:/config:ro" \
  -e BATTERY_OPTIMIZER_DATA_DIR=/data \
  -e BATTERY_OPTIMIZER_CONFIG=/config/tariff.yaml \
  -e BATTERY_OPTIMIZER_SERVICE_CONFIG=/config/service.yaml \
  -e BATTERY_OPTIMIZER_HA_URL -e BATTERY_OPTIMIZER_HA_TOKEN -e BATTERY_OPTIMIZER_PROMETHEUS_URL \
  ghcr.io/pando85/battery-optimizer:latest serve
```

## Data directory, provenance and code identity

The data directory must be writable by uid `65532`. Verify an image's provenance with
`gh attestation verify oci://ghcr.io/pando85/battery-optimizer:<tag> --owner pando85`.
The image records the commit it was built from in `BATTERY_OPTIMIZER_GIT_SHA` (see [Configuration](configuration.md#data-directory)).

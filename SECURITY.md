# Security policy

## Reporting a vulnerability

Please use GitHub private vulnerability reporting:
<https://github.com/pando85/battery-optimizer/security/advisories/new>

Do not open a public issue for a vulnerability. You can expect an acknowledgement within a few
days. Only the latest release is supported.

## Never post private data

This tool handles household energy data and credentials (Home Assistant and Prometheus access).
Never include any of the following in an issue, pull request, discussion or log excerpt:

- energy measurements, datasets or parquet files;
- Home Assistant or Prometheus URLs and tokens (`BATTERY_OPTIMIZER_HA_TOKEN`, ...);
- anything else that identifies an installation.

If you pasted something sensitive by mistake, revoke the token first, then edit or delete the post.

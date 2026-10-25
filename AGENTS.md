# battery-optimizer

Python toolkit for household battery optimisation: Prometheus / Home Assistant extraction, dataset
build and battery MPC benchmarks. This file is the entry point for coding agents. Detailed workflows
live in `.opencode/skills/` (also exposed to Claude Code through the `.claude/skills` symlink).

## Essential commands

Local uv may be stale: use `make UV="uvx uv@<version>" ...` with the version pinned in
`.github/actions/setup-uv/action.yml`.

```bash
make help              # list targets
make git-hooks         # install pre-commit + commit-msg hooks (once)
make lint              # every pre-commit hook on all files (ruff, yamllint, shellcheck, uv.lock)
make fmt               # ruff check --fix + ruff format
make test              # pytest (~1900 tests, several minutes)
make image             # build the container image locally
make release           # cut a release (see the release skill)
```

## Repository map

- `src/battery_optimizer/`: the package and the `battery-optimizer` CLI (`cli.py`).
- `tests/`: pytest suite; `hack/`: one-off analysis scripts; `config/tariff.yaml`: tariff config.
- `docs/adr/`: architecture decision records; `docs/planning/`: plans and progress notes.
- `.ci/`, `Dockerfile`, `.github/`: release script, image, workflows.

## Rules

- Python 3.14, ruff (line length 100; rules E, F, I, N, W, UP). Run `make lint` and `make test`
  before reporting completion and include the actual results.
- Dependencies are unconstrained in `pyproject.toml`; `uv.lock` is the source of truth and Renovate
  refreshes it weekly. Commit `uv.lock` with any dependency change.
- One place per version pin: the project version is `[project] version` in `pyproject.toml` (changed
  only by `make release`), the uv version is pinned in `.github/actions/setup-uv/action.yml` and the
  `Dockerfile`. Pin GitHub actions to a full commit SHA with a `# vX.Y.Z` comment.
- Commits: Conventional Commits (`type(scope): Subject`, types in `.commitlintrc.json`), GPG-signed
  and `Signed-off-by` (`git commit -s`). Never pass `--no-gpg-sign` or `--no-verify`.
- Never edit `CHANGELOG.md` by hand and never create tags: `make release` and CI do it.
  Read `.opencode/skills/release/SKILL.md` before any release work.
- Behavioral changes need tests. Keep the smallest correct diff.

## Privacy and safety

- Never commit energy data, datasets, parquet files, tokens, Home Assistant or Prometheus URLs of a
  real installation. Use environment variables; `.envrc` is gitignored (`.envrc.example` is the template).
- The data directory (`BATTERY_OPTIMIZER_DATA_DIR`, default `~/data/energy-baseline`) must stay outside
  the repository; the code refuses a path inside a git tree. The default keeps the pre-rename name on
  purpose.
- Shadow phase: the project reads and benchmarks only. Do not add write credentials, service calls to
  Home Assistant, or any actuation of the battery or other devices. Home Assistant tokens must be
  read-only.
- The container image holds no data and no git; `BATTERY_OPTIMIZER_GIT_SHA` is the build-time commit
  identity.

## Further reading

- Decisions: `docs/adr/` (start with ADR-0003 for the measurement baseline and ADR-0004 for the
  production architecture).
- Plans and progress: `docs/planning/`, provenance of the extraction in `docs/provenance.md`.
- CLI behavior, environment variables and secrets policy: `README.md`.

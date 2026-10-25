# battery-optimizer - lint, test, build and release commands
#
# Thin wrappers around uv; the logic lives in pyproject.toml, .pre-commit-config.yaml and .ci/.
# Use another uv with: UV="uvx uv@0.13.5" just <recipe>

# =============================================================================
# Configuration
# =============================================================================

# uv command to use (override with the UV environment variable)
UV := env("UV", "uv")

# Tag of the locally built container image
IMAGE := "battery-optimizer:dev"

# Version from pyproject.toml; only `just release` changes it
PROJECT_VERSION := `sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -n1`

# Default recipe
default:
    @just --list

# =============================================================================
# Development
# =============================================================================

# Install the pre-commit and commit-msg git hooks (once per clone)
git-hooks:
    {{UV}} run --only-group tools pre-commit install --install-hooks -t pre-commit -t commit-msg

# Run all pre-commit checks (ruff, yamllint, shellcheck, uv.lock, ...) on every file
lint:
    {{UV}} run --only-group tools pre-commit run --all-files

# Format and auto-fix with ruff
fmt:
    {{UV}} run --extra dev ruff check --fix .
    {{UV}} run --extra dev ruff format .

# Run the test suite (extra arguments are passed to pytest)
test *args:
    {{UV}} run --extra dev pytest {{args}}

# =============================================================================
# Build
# =============================================================================

# Build the sdist and wheel into dist/
build:
    {{UV}} build

# Build the container image locally
image:
    docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t {{IMAGE}} .

# =============================================================================
# Release
# =============================================================================

# Prepend the unreleased commits to CHANGELOG.md as the next version
update-changelog:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f CHANGELOG.md ]; then
        {{UV}} run --only-group tools git-cliff -t v{{PROJECT_VERSION}} -u -p CHANGELOG.md
    else
        {{UV}} run --only-group tools git-cliff -t v{{PROJECT_VERSION}} -u -o CHANGELOG.md
    fi
    # git-cliff leaves trailing blank lines, which the end-of-file-fixer hook rejects.
    printf '%s\n' "$(cat CHANGELOG.md)" > CHANGELOG.md.tmp
    mv CHANGELOG.md.tmp CHANGELOG.md

# Cut a release: bump the version, update the changelog, push release/vX.Y.Z (version is prompted for if omitted)
release version="":
    UV="{{UV}}" .ci/release.sh {{version}}

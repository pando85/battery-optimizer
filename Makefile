# Thin wrappers around uv; the logic lives in pyproject.toml, .pre-commit-config.yaml and .ci/.
# Use another uv with: make UV="uvx uv@0.13.5" <target>
UV ?= uv
export UV

IMAGE ?= battery-optimizer:dev
PROJECT_VERSION = $(shell $(UV) version --short)

.DEFAULT_GOAL := help

.PHONY: help
help:	## show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

.PHONY: git-hooks
git-hooks:	## install the pre-commit and commit-msg git hooks
	$(UV) run --only-group tools pre-commit install --install-hooks -t pre-commit -t commit-msg

.PHONY: lint
lint:	## run all pre-commit checks (ruff, yamllint, shellcheck, ...) on every file
	$(UV) run --only-group tools pre-commit run --all-files

.PHONY: fmt
fmt:	## format and auto-fix with ruff
	$(UV) run --extra dev ruff check --fix .
	$(UV) run --extra dev ruff format .

.PHONY: test
test:	## run the test suite
	$(UV) run --extra dev pytest

.PHONY: build
build:	## build the sdist and wheel into dist/
	$(UV) build

.PHONY: image
image:	## build the container image locally (IMAGE=name:tag)
	docker build --build-arg GIT_SHA=$$(git rev-parse HEAD) -t $(IMAGE) .

.PHONY: update-changelog
update-changelog:	## prepend the unreleased commits to CHANGELOG.md as the next version
	@if [ -f CHANGELOG.md ]; then \
		$(UV) run --only-group tools git-cliff -t v$(PROJECT_VERSION) -u -p CHANGELOG.md; \
	else \
		$(UV) run --only-group tools git-cliff -t v$(PROJECT_VERSION) -u -o CHANGELOG.md; \
	fi
	@# git-cliff leaves trailing blank lines, which the end-of-file-fixer hook rejects.
	@printf '%s\n' "$$(cat CHANGELOG.md)" > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md

.PHONY: release
release:	## cut a release: make release [VERSION=X.Y.Z]
	.ci/release.sh $(VERSION)

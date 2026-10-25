#!/usr/bin/env bash
# Prepare a release: bump the version, update the changelog and push a release/vX branch.
#
# Usage: .ci/release.sh [X.Y.Z]     (prompts when the version is omitted)
#        make release [VERSION=X.Y.Z]
#
# The version lives in [project] version of pyproject.toml. After the release PR is merged,
# .github/workflows/auto-tag.yml creates the signed tag and images.yml publishes everything.
set -euo pipefail

UV=${UV:-uv}
REMOTE=origin

die() {
    echo "error: $*" >&2
    exit 1
}

cd "$(git rev-parse --show-toplevel)"

if ! git remote | grep -qx "$REMOTE"; then
    die "remote '$REMOTE' does not exist"
fi

# git-cliff needs the full history and all tags to build an accurate changelog.
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
    echo "Shallow clone detected. Fetching full history..."
    git fetch --unshallow --quiet "$REMOTE"
fi
git fetch --tags --quiet "$REMOTE"

DEFAULT_BRANCH=$(git symbolic-ref --short "refs/remotes/$REMOTE/HEAD" 2>/dev/null | sed "s@^$REMOTE/@@" || true)
if [ -z "$DEFAULT_BRANCH" ]; then
    DEFAULT_BRANCH=$(git remote show "$REMOTE" 2>/dev/null | sed -n 's/.*HEAD branch: //p')
fi
[ -n "$DEFAULT_BRANCH" ] || die "could not determine the default branch of '$REMOTE'"
git show-ref --quiet "refs/remotes/$REMOTE/$DEFAULT_BRANCH" ||
    die "branch '$DEFAULT_BRANCH' does not exist on '$REMOTE'"

[ -z "$(git status --porcelain)" ] || die "working tree is not clean; commit or stash your changes"

LOCAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$LOCAL_BRANCH" = "$DEFAULT_BRANCH" ] ||
    die "releases are cut from '$DEFAULT_BRANCH', but you are on '$LOCAL_BRANCH'"

AHEAD=$(git rev-list --count "$REMOTE/$DEFAULT_BRANCH..HEAD")
[ "$AHEAD" -eq 0 ] ||
    die "$AHEAD commit(s) are not in '$REMOTE/$DEFAULT_BRANCH'; merge them first (the changelog links to merged commits)"
BEHIND=$(git rev-list --count "HEAD..$REMOTE/$DEFAULT_BRANCH")
[ "$BEHIND" -eq 0 ] ||
    die "'$DEFAULT_BRANCH' is $BEHIND commit(s) behind '$REMOTE/$DEFAULT_BRANCH'; run: git pull --ff-only"

CURRENT=$($UV version --short)
LAST_TAG=$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)
echo "Current version: $CURRENT (last tag: ${LAST_TAG:-none})"
if [ -n "$LAST_TAG" ]; then
    git --no-pager log --oneline "$LAST_TAG..HEAD"
fi

VERSION=${1:-}
if [ -z "$VERSION" ]; then
    read -r -p "New version (X.Y.Z): " VERSION
fi
VERSION=${VERSION#v}
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$VERSION' is not a valid X.Y.Z version"

if git rev-parse --quiet --verify "refs/tags/v$VERSION" >/dev/null; then
    die "tag v$VERSION already exists"
fi
if [ -n "$LAST_TAG" ] && [ "$(printf '%s\n%s\n' "${LAST_TAG#v}" "$VERSION" | sort -V | tail -n1)" != "$VERSION" ]; then
    die "version $VERSION is not newer than $LAST_TAG"
fi

BRANCH="release/v$VERSION"
git checkout -b "$BRANCH"

$UV version --no-sync "$VERSION"
$UV lock
UV="$UV" make update-changelog

git add pyproject.toml uv.lock CHANGELOG.md
git commit -s -m "release: Version $VERSION"
git push -u "$REMOTE" "$BRANCH"

echo
echo "Open the release PR (after merging, the tag, image and GitHub release are automatic):"
echo "  gh pr create --base $DEFAULT_BRANCH --head $BRANCH --title 'release: Version $VERSION' --body 'Release v$VERSION'"

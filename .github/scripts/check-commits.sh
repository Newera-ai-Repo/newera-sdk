#!/usr/bin/env bash
# Fails if any commit in RANGE carries a co-author trailer, an AI-tool footer,
# or an AI-tool author/committer identity. Also checks PR_TITLE / PR_BODY if set.
# Usage: check-commits.sh <git-range>     e.g. origin/main..HEAD
set -euo pipefail

RANGE="${1:?usage: check-commits.sh <git-range>}"
MSG_PATTERN='^(co-authored-by:|claude-session:)|generated with \[?claude code|claude\.com/claude-code'
ID_PATTERN='claude|anthropic'

fail=0
count=0
while IFS= read -r sha; do
  [ -z "$sha" ] && continue
  count=$((count + 1))
  ids=$(git log -1 --format='%an <%ae>%n%cn <%ce>' "$sha")
  if grep -qiE "$ID_PATTERN" <<<"$ids"; then
    echo "::error::commit ${sha:0:7} has a blocked author/committer identity:"; echo "$ids"; fail=1
  fi
  hits=$(git log -1 --format=%B "$sha" | grep -iE "$MSG_PATTERN" || true)
  if [ -n "$hits" ]; then
    echo "::error::commit ${sha:0:7} message has a blocked line:"; echo "$hits"; fail=1
  fi
done < <(git rev-list "$RANGE")

for field in PR_TITLE PR_BODY; do
  text="${!field:-}"
  hits=$(printf '%s\n' "$text" | grep -iE "$MSG_PATTERN" || true)
  if [ -n "$hits" ]; then
    echo "::error::$field has a blocked line:"; echo "$hits"; fail=1
  fi
done

echo "checked $count commit(s) in $RANGE"
if [ "$fail" -ne 0 ]; then
  echo "Remove Co-Authored-By / AI-tool lines and use your own git identity, then push again."
  exit 1
fi
echo "ok: no co-author trailers or AI-tool identities"

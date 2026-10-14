#!/usr/bin/env bash
# Commit whatever changed under results/ and push, retrying over concurrent pushes.
set -euo pipefail
msg="${1:-update results}"
git config user.name  "replay-bot"
git config user.email "replay-bot@users.noreply.github.com"
git add results/
if git diff --cached --quiet; then
  echo "no result changes to commit"; exit 0
fi
git commit -m "$msg"
for i in 1 2 3 4 5; do
  if git pull --rebase --quiet && git push --quiet; then exit 0; fi
  echo "push failed, retry $i"; sleep $((i * 7))
done
echo "could not push results"; exit 1

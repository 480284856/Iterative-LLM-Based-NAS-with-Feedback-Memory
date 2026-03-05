#!/usr/bin/env bash
# Push this repo to GitHub (add origin, rename branch to main, push).

set -e
REPO_URL="https://github.com/480284856/Iterative-LLM-Based-NAS-with-Feedback-Memory.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git branch -M main
git push -u origin main

echo "Done. Pushed to $REPO_URL (branch main)."

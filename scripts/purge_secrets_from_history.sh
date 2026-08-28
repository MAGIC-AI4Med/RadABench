#!/usr/bin/env bash
# ============================================================================
# Rewrite this repository's history so the 19 leaked API keys are unreachable.
#
# READ BEFORE RUNNING. This rewrites every commit hash and requires a
# force-push. Anyone with an existing clone must re-clone afterwards.
#
# Revoking the keys at the provider is the step that actually stops the
# leak, so do that FIRST. History rewriting only stops future discovery.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "==> Repository: $REPO_ROOT"
echo "==> Current HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo

# ---------------------------------------------------------------- 0. 前置检查
if ! command -v git-filter-repo >/dev/null 2>&1; then
  cat <<'MSG'
git-filter-repo is not installed. Install it with one of:

    pip install git-filter-repo
    brew install git-filter-repo
    apt install git-filter-repo

MSG
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean. Commit or stash your changes first," >&2
  echo "otherwise they will be lost during the rewrite." >&2
  exit 1
fi

# ------------------------------------------------------------ 1. 完整备份
BACKUP="../$(basename "$REPO_ROOT")-backup-$(date +%Y%m%d-%H%M%S).bundle"
git bundle create "$BACKUP" --all
echo "==> Full backup written to $BACKUP"
echo "    Restore with: git clone $BACKUP recovered-repo"
echo

# ------------------------------------------- 2. 收集历史中出现过的所有密钥
echo "==> Scanning every commit for credentials..."
PATTERNS_FILE="$(mktemp)"
trap 'rm -f "$PATTERNS_FILE"' EXIT

git rev-list --all | while read -r commit; do
  git grep -hoE 'sk-[A-Za-z0-9_-]{20,}' "$commit" -- '*.py' 2>/dev/null || true
done | sort -u | while read -r key; do
  # `literal:` avoids regex-escaping every key; ==>REDACTED<== marks the spot
  printf 'literal:%s==>***REMOVED***\n' "$key"
done > "$PATTERNS_FILE"

# The third-party relay host is not a credential, but it identifies the
# leaked account and should not survive either.
printf 'literal:https://api.claudeshop.top==>https://api.example.invalid\n' >> "$PATTERNS_FILE"

FOUND=$(grep -c . "$PATTERNS_FILE" || true)
echo "    Found $FOUND distinct strings to redact across history."
if [ "$FOUND" -eq 0 ]; then
  echo "    Nothing to do."
  exit 0
fi
echo

# ------------------------------------------------------------- 3. 重写历史
read -r -p "Rewrite all history now? This cannot be undone. [y/N] " CONFIRM
[[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]] || { echo "Aborted."; exit 1; }

git filter-repo --replace-text "$PATTERNS_FILE" --force

echo
echo "==> History rewritten."
echo

# --------------------------------------------------------------- 4. 复查
echo "==> Verifying..."
REMAIN=$(git rev-list --all | while read -r c; do
  git grep -qE 'sk-[A-Za-z0-9_-]{20,}' "$c" 2>/dev/null && echo "$c"
done | wc -l)
echo "    Commits still containing a key pattern: $REMAIN"
[ "$REMAIN" -eq 0 ] && echo "    Clean." || echo "    STILL DIRTY. Inspect before pushing."
echo

# ------------------------------------------------------------- 5. 后续步骤
cat <<'NEXT'
Remaining steps, in order:

  1. Confirm all 19 keys are revoked at their providers. If any is still
     live, the rewrite has not protected you.

  2. filter-repo drops the remote. Re-add it:
         git remote add origin https://github.com/MAGIC-AI4Med/RadABench.git

  3. Force-push every branch and tag:
         git push --force --all origin
         git push --force --tags origin

  4. Tell collaborators to re-clone. Old clones still hold the keys, and a
     later merge from one would reintroduce them.

  5. In GitHub repo settings, enable Secret scanning and Push protection.

  6. GitHub keeps rewritten commits reachable for a while through the API.
     Ask GitHub Support to garbage-collect them, or accept that the keys
     stay retrievable until they do. Revocation remains the real fix.
NEXT

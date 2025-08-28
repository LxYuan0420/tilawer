#!/usr/bin/env bash
set -euo pipefail

# Mirror selected paths from a source repo to a destination repo,
# replaying commits (message, author, date) and pushing.
#
# - Finds the last mirrored source SHA in destination via a "Source-Commit:" trailer.
# - Replays only commits that touch the allowed paths, in order.
# - Copies file contents from each source commit, stages each file separately,
#   commits with the same message/author/date, and appends "Source-Commit: <sha>".
# - Pushes to the current branch on destination. Optional SSH key for push.
#
# Usage:
#   scripts/sync_mirror.sh \
#     --src ~/works/tilawer \
#     --dest ~/personal_works/tilawer-public \
#     [--paths til wer README.md] \
#     [--ssh-key ~/.ssh/id_ed25519_personal] \
#     [--dry-run]
#
# Notes:
# - Defaults to syncing paths: til, wer, README.md
# - Skips merge commits (replay simple linear commits only)
# - Requires git in PATH

SRC_REPO=""
DEST_REPO=""
SSH_KEY=""
DRY_RUN=0
PATHS=(til wer README.md)

die() { echo "Error: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

need git

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC_REPO=${2:-}; shift 2;;
    --dest) DEST_REPO=${2:-}; shift 2;;
    --paths)
      PATHS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        PATHS+=("$1"); shift
      done
      ;;
    --ssh-key) SSH_KEY=${2:-}; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help)
      sed -n '1,80p' "$0" | sed -n '1,50p'; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$SRC_REPO" && -n "$DEST_REPO" ]] || die "--src and --dest are required"
[[ -d "$SRC_REPO/.git" ]] || die "Not a git repo: $SRC_REPO"
[[ -d "$DEST_REPO/.git" ]] || die "Not a git repo: $DEST_REPO"

echo "Source:      $SRC_REPO"
echo "Destination: $DEST_REPO"
echo "Paths:       ${PATHS[*]}"
[[ $DRY_RUN -eq 1 ]] && echo "Mode:        DRY RUN"

# Find last mirrored source SHA in destination (by trailer line)
LAST_MIRRORED_SHA=$(git -C "$DEST_REPO" log -n 100 --pretty=format:%B \
  | sed -n 's/^Source-Commit: \([0-9a-f]\{7,\}\).*/\1/p' | head -n1 || true)

if [[ -n "$LAST_MIRRORED_SHA" ]]; then
  echo "Last mirrored source commit: $LAST_MIRRORED_SHA"
  RANGE="$LAST_MIRRORED_SHA..HEAD"
else
  echo "No previous Source-Commit trailer found in destination. Mirroring from repo start."
  RANGE="--all"
fi

# Collect candidate commits from source, in chronological order, touching PATHS
mapfile -t CANDIDATE_COMMITS < <(
  if [[ "$RANGE" == "--all" ]]; then
    git -C "$SRC_REPO" rev-list --reverse HEAD
  else
    git -C "$SRC_REPO" rev-list --reverse "$RANGE"
  fi
)

COMMITS_TO_REPLAY=()
for sha in "${CANDIDATE_COMMITS[@]}"; do
  # Skip merge commits (more than one parent)
  parents=$(git -C "$SRC_REPO" rev-list --parents -n 1 "$sha")
  parent_count=$(( $(wc -w <<<"$parents") - 1 ))
  if (( parent_count > 1 )); then
    continue
  fi
  # Check if commit touches allowed paths
  if git -C "$SRC_REPO" show --pretty=format: --name-only "$sha" -- "${PATHS[@]}" | grep -q .; then
    COMMITS_TO_REPLAY+=("$sha")
  fi
done

echo "Commits to replay: ${#COMMITS_TO_REPLAY[@]}"

dest_branch=$(git -C "$DEST_REPO" rev-parse --abbrev-ref HEAD)
echo "Destination branch: $dest_branch"

for sha in "${COMMITS_TO_REPLAY[@]}"; do
  echo "\nReplaying $sha..."

  # Gather metadata
  author=$(git -C "$SRC_REPO" show -s --format='%an <%ae>' "$sha")
  adate=$(git -C "$SRC_REPO" show -s --format='%aI' "$sha")
  tmpmsg=$(mktemp)
  git -C "$SRC_REPO" show -s --format=%B "$sha" > "$tmpmsg"
  echo -e "\nSource-Commit: $sha" >> "$tmpmsg"

  # File changes limited to PATHS
  # Use name-status for per-file operations; handle A/M/D/R
  while IFS=$'\t' read -r status path1 path2; do
    [[ -n "${status:-}" ]] || continue
    case "$status" in
      A|M)
        [[ -n "$path1" ]] || continue
        src_blob="$sha:$path1"
        dst_file="$DEST_REPO/$path1"
        mkdir -p "$(dirname "$dst_file")"
        if [[ $DRY_RUN -eq 0 ]]; then
          git -C "$SRC_REPO" show "$src_blob" > "$dst_file"
          git -C "$DEST_REPO" add -- "$path1"
        else
          echo "would write $path1"
        fi
        ;;
      D)
        [[ -n "$path1" ]] || continue
        if [[ $DRY_RUN -eq 0 ]]; then
          if [[ -e "$DEST_REPO/$path1" ]]; then
            git -C "$DEST_REPO" rm -f -- "$path1"
          fi
        else
          echo "would delete $path1"
        fi
        ;;
      R*)
        # rename: path1=old, path2=new
        [[ -n "$path1" && -n "$path2" ]] || continue
        # remove old if exists, then write new from blob
        if [[ $DRY_RUN -eq 0 ]]; then
          if [[ -e "$DEST_REPO/$path1" ]]; then
            git -C "$DEST_REPO" rm -f -- "$path1"
          fi
          mkdir -p "$(dirname "$DEST_REPO/$path2")"
          git -C "$SRC_REPO" show "$sha:$path2" > "$DEST_REPO/$path2"
          git -C "$DEST_REPO" add -- "$path2"
        else
          echo "would rename $path1 -> $path2"
        fi
        ;;
      *)
        # treat others conservatively as modify if path present
        if [[ -n "$path1" ]]; then
          if [[ $DRY_RUN -eq 0 && -f "$DEST_REPO/$path1" ]]; then
            git -C "$SRC_REPO" show "$sha:$path1" > "$DEST_REPO/$path1"
            git -C "$DEST_REPO" add -- "$path1"
          else
            echo "skip status=$status path=$path1"
          fi
        fi
        ;;
    esac
  done < <(git -C "$SRC_REPO" show --pretty=format: --name-status "$sha" -- "${PATHS[@]}")

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN: would commit and push"
    rm -f "$tmpmsg"
    continue
  fi

  # Commit only if there are staged changes
  if git -C "$DEST_REPO" diff --cached --quiet; then
    echo "No staged changes for $sha (paths filtered), skipping commit."
    rm -f "$tmpmsg"
    continue
  fi

  GIT_COMMITTER_DATE="$adate" git -C "$DEST_REPO" \
    commit --author "$author" --date "$adate" -F "$tmpmsg"
  rm -f "$tmpmsg"
  echo "Replayed commit $sha"
done

# Push destination
if [[ $DRY_RUN -eq 0 ]]; then
  if [[ -n "$SSH_KEY" ]]; then
    echo "Pushing with SSH key: $SSH_KEY"
    GIT_SSH_COMMAND="ssh -i $SSH_KEY" git -C "$DEST_REPO" push origin "$dest_branch"
  else
    git -C "$DEST_REPO" push origin "$dest_branch"
  fi
fi

echo "Done."


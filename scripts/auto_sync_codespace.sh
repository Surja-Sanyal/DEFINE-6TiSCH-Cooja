#!/usr/bin/env bash
cd /workspaces/DEFINE-6TiSCH-Cooja || exit 1
while true; do
  git fetch origin main >/dev/null 2>&1
  LOCAL=$(git rev-parse HEAD 2>/dev/null)
  REMOTE=$(git rev-parse origin/main 2>/dev/null)
  if [ -n "$LOCAL" ] && [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
    # Only auto-pull if working tree has no conflicting staged operations
    git pull --rebase --autostash origin main >/dev/null 2>&1 && \
      echo "[Auto-Sync] Pulled latest GitHub Actions results into Codespace at $(date)"
  fi
  sleep 30
done

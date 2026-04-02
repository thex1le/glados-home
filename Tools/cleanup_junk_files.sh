#!/bin/bash
# Cleans up 0-byte junk files in the home directory.
# These are created by an OS/driver bug that writes memory addresses as filenames.
#
# Usage:
#   # Dry run (show what would be deleted):
#   bash Tools/cleanup_junk_files.sh
#
#   # Actually delete:
#   bash Tools/cleanup_junk_files.sh --delete
#
# Can be run as a cron job to keep them from piling up:
#   */30 * * * * /home/textile/glados-home/Tools/cleanup_junk_files.sh --delete >> /tmp/junk_cleanup.log 2>&1

set -euo pipefail

TARGET_DIR="${HOME}"
MODE="${1:-}"
COUNT=0

# Approach: any 0-byte file whose name contains non-printable or high-byte
# characters is junk. Real files have purely printable ASCII names.
# This catches all the garbage filenames including ones starting with '.'

while IFS= read -r -d '' file; do
    # Only regular files
    [ -f "$file" ] || continue

    # Only 0-byte files
    [ ! -s "$file" ] || continue

    name="$(basename "$file")"

    # If the filename is 100% printable ASCII (space through tilde), keep it.
    # This preserves .bashrc, .profile, audio.py, .sudo_as_admin_successful, etc.
    if echo "$name" | LC_ALL=C grep -qP '^[\x20-\x7e]+$'; then
        continue
    fi

    # Filename contains non-ASCII or control characters — it's junk
    if [ "$MODE" = "--delete" ]; then
        rm -f -- "$file"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Deleted: $(ls -la -- "$file" 2>/dev/null || echo "$name")"
    else
        echo "Would delete: $name ($(stat -c '%y' -- "$file" 2>/dev/null || echo 'unknown date'))"
    fi
    COUNT=$((COUNT + 1))
done < <(find "$TARGET_DIR" -maxdepth 1 -type f -empty -print0 2>/dev/null)

if [ "$COUNT" -eq 0 ]; then
    echo "No junk files found."
else
    if [ "$MODE" = "--delete" ]; then
        echo "Cleaned up $COUNT junk files."
    else
        echo "Found $COUNT junk files. Run with --delete to remove them."
    fi
fi

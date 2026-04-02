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

# Known legitimate files/dirs that could be 0 bytes -- skip these
is_legitimate() {
    local name="$1"
    # Skip dotfiles, known directories, known scripts
    case "$name" in
        .*) return 0 ;;
        *.py|*.sh|*.txt|*.conf|*.wav|*.json|*.yaml|*.yml|*.md) return 0 ;;
        Bookshelf|Desktop|Documents|Downloads|Music|Pictures|Public|Templates|Videos) return 0 ;;
        glados-home|neopixel|servohat|tftdisplay|scan_results|RtspServer) return 0 ;;
    esac
    return 1
}

while IFS= read -r -d '' file; do
    name="$(basename "$file")"

    # Only look at regular files (not dirs, symlinks, pipes, etc.)
    [ -f "$file" ] || continue

    # Only 0-byte files
    [ ! -s "$file" ] || continue

    # Skip legitimate files
    if is_legitimate "$name"; then
        continue
    fi

    # These junk files are short names with non-printable or high-byte chars
    # Skip anything that looks like a normal filename (all printable ASCII, > 10 chars)
    if echo "$name" | grep -qP '^[\x20-\x7e]{10,}$'; then
        continue
    fi

    if [ "$MODE" = "--delete" ]; then
        rm -f -- "$file"
        echo "Deleted: $name"
    else
        echo "Would delete: $name"
    fi
    COUNT=$((COUNT + 1))
done < <(find "$TARGET_DIR" -maxdepth 1 -type f -empty -print0 2>/dev/null)

if [ "$COUNT" -eq 0 ]; then
    echo "No junk files found."
else
    if [ "$MODE" = "--delete" ]; then
        echo "Deleted $COUNT junk files."
    else
        echo "Found $COUNT junk files. Run with --delete to remove them."
    fi
fi

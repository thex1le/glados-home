"""Merge new config keys from the repo template into deployed config files.

Compares the template glog.conf (checked into git) against a deployed config
file and reports/merges differences:
  - NEW keys in template that are missing from deployed -> added with template defaults
  - NEW sections in template that are missing from deployed -> added entirely
  - REMOVED keys in template that exist in deployed -> warned but not removed
  - CHANGED default values -> reported but deployed values are preserved

Usage:
    # Show what would change (dry run, default)
    python3 Tools/config_merge.py glog.conf dont_commit_glados.conf

    # Apply changes
    python3 Tools/config_merge.py glog.conf dont_commit_glados.conf --apply

    # Merge into multiple deployed configs
    python3 Tools/config_merge.py glog.conf pi4.conf pi5.conf gpu.conf --apply
"""

import argparse
import sys
import os
import shutil
from configparser import ConfigParser
from datetime import datetime


def load_config(filepath: str) -> ConfigParser:
    """Load a config file, preserving comments isn't possible with ConfigParser
    but we handle the merge at the key level."""
    config = ConfigParser()
    config.read(filepath)
    return config


def _section_only_keys(config: ConfigParser, section: str) -> dict:
    """Get keys that are explicitly defined in a section, excluding DEFAULT inheritance.

    ConfigParser's .items(section) and dict(config[section]) both include
    DEFAULT keys merged in. This function returns only the keys that are
    actually written in the [section] block of the config file.

    Args:
        config: The ConfigParser instance.
        section: The section name.

    Returns:
        Dict of {key: value} for keys explicitly in this section.
    """
    # _sections is the internal dict of {section: {key: value}} without DEFAULT
    if section in config._sections:
        return dict(config._sections[section])
    return {}


def compare_configs(template: ConfigParser, deployed: ConfigParser) -> dict:
    """Compare template against deployed and return differences.

    Returns:
        Dict with keys:
            new_sections: sections in template but not deployed
            new_keys: dict of {section: {key: template_value}} for missing keys
            removed_keys: dict of {section: [keys]} in deployed but not template
            changed_defaults: dict of {section: {key: (template_val, deployed_val)}}
    """
    result = {
        "new_sections": [],
        "new_keys": {},
        "removed_keys": {},
        "changed_defaults": {},
    }

    # Check for new sections
    for section in template.sections():
        if not deployed.has_section(section):
            result["new_sections"].append(section)
            continue

        # Only compare keys explicitly in this section (not inherited from DEFAULT)
        template_section_keys = _section_only_keys(template, section)
        deployed_section_keys = _section_only_keys(deployed, section)

        for key, template_val in template_section_keys.items():
            if key == "__name__":
                continue
            if key not in deployed_section_keys:
                # Also check if deployed has it via DEFAULT — if so, skip
                if deployed.has_option(section, key):
                    continue
                if section not in result["new_keys"]:
                    result["new_keys"][section] = {}
                result["new_keys"][section][key] = template_val

    # Check for keys in deployed that aren't in template (potential removals)
    for section in deployed.sections():
        if not template.has_section(section):
            continue
        deployed_section_keys = _section_only_keys(deployed, section)
        template_section_keys = _section_only_keys(template, section)
        for key in deployed_section_keys:
            if key == "__name__":
                continue
            if key not in template_section_keys:
                if section not in result["removed_keys"]:
                    result["removed_keys"][section] = []
                result["removed_keys"][section].append(key)

    return result


def print_diff(filepath: str, diff: dict) -> int:
    """Print a human-readable diff. Returns count of actionable changes."""
    changes = 0
    print(f"\n--- {filepath} ---")

    if diff["new_sections"]:
        for section in diff["new_sections"]:
            print(f"  + NEW SECTION [{section}]")
            changes += 1

    if diff["new_keys"]:
        for section, keys in diff["new_keys"].items():
            for key, val in keys.items():
                print(f"  + [{section}] {key} = {val}")
                changes += 1

    if diff["removed_keys"]:
        for section, keys in diff["removed_keys"].items():
            for key in keys:
                print(f"  ? [{section}] {key}  (in deployed but not in template -- keeping)")

    if changes == 0:
        print("  Up to date.")

    return changes


def apply_merge(template: ConfigParser, deployed_path: str, diff: dict) -> None:
    """Apply the merge: read the deployed file as text, append new sections/keys."""
    # Backup first
    backup_path = f"{deployed_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(deployed_path, backup_path)
    print(f"  Backup saved to {backup_path}")

    # Read the deployed file as raw text so we preserve comments and formatting
    with open(deployed_path, 'r') as f:
        lines = f.readlines()

    additions = []

    # Add new keys to existing sections
    if diff["new_keys"]:
        for section, keys in diff["new_keys"].items():
            # Find the section in the file and append keys after it
            section_header = f"[{section}]"
            inserted = False
            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    # Find the last non-blank content line in this section
                    # (before the next section header or EOF)
                    insert_at = i + 1
                    last_content = i  # track last line with actual content
                    while insert_at < len(lines):
                        stripped = lines[insert_at].strip()
                        if stripped.startswith('[') and stripped.endswith(']'):
                            break
                        if stripped:
                            last_content = insert_at
                        insert_at += 1
                    # Insert new keys right after the last content line
                    insert_pos = last_content + 1
                    new_lines = []
                    for key, val in keys.items():
                        new_lines.append(f"{key} = {val}\n")
                    for nl in reversed(new_lines):
                        lines.insert(insert_pos, nl)
                    inserted = True
                    break
            if not inserted:
                # Section exists in ConfigParser but wasn't found as raw text (unusual)
                additions.append((section, keys))

    # Add entire new sections at the end (section-only keys, no DEFAULT inheritance)
    if diff["new_sections"]:
        for section in diff["new_sections"]:
            lines.append(f"\n[{section}]\n")
            section_keys = _section_only_keys(template, section)
            for key, val in section_keys.items():
                if key == "__name__":
                    continue
                lines.append(f"{key} = {val}\n")

    # Clean up formatting: ensure blank line between sections, no blanks within
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # If this is a section header, ensure blank line before it (unless first line)
        if stripped.startswith('[') and stripped.endswith(']'):
            if cleaned and cleaned[-1].strip():
                cleaned.append("\n")
            cleaned.append(line)
        # Skip consecutive blank lines within a section
        elif not stripped and cleaned and not cleaned[-1].strip():
            continue
        else:
            cleaned.append(line)
    # Ensure file ends with newline
    if cleaned and not cleaned[-1].endswith("\n"):
        cleaned.append("\n")
    lines = cleaned

    # Handle any sections we couldn't find in raw text
    for section, keys in additions:
        # Shouldn't happen normally, but just append
        for key, val in keys.items():
            lines.append(f"{key} = {val}\n")

    with open(deployed_path, 'w') as f:
        f.writelines(lines)

    print(f"  Merged changes into {deployed_path}")


def clean_config(filepath: str, dry_run: bool = False) -> int:
    """Clean up a config file: fix spacing, remove duplicate DEFAULT keys from sections.

    Rules applied:
        - Blank line between sections (before each [SECTION] header)
        - No blank lines between keys within a section
        - No duplicate DEFAULT keys repeated in individual sections
        - Comments preserved in place
        - Trailing whitespace removed
        - File ends with single newline

    Args:
        filepath: Path to the config file to clean.
        dry_run: If True, print what would change without modifying the file.

    Returns:
        Number of lines changed.
    """
    with open(filepath, 'r') as f:
        original_lines = f.readlines()

    # Parse to identify DEFAULT keys and their values
    config = ConfigParser()
    config.read(filepath)
    default_keys = config.defaults()  # dict of {key: value}

    cleaned = []
    current_section = None
    changes = 0

    for line in original_lines:
        stripped = line.strip()

        # Track current section
        if stripped.startswith('[') and stripped.endswith(']'):
            # Blank line before section header (unless first line)
            if cleaned and cleaned[-1].strip():
                cleaned.append("\n")
            # Remove any accumulated blank lines before the header
            while cleaned and not cleaned[-1].strip():
                if len(cleaned) >= 2 and not cleaned[-2].strip():
                    cleaned.pop()
                else:
                    break
            cleaned.append(line.rstrip() + "\n")
            current_section = stripped[1:-1]
            continue

        # Skip blank lines within a section
        if not stripped:
            # Skip consecutive blanks
            if cleaned and not cleaned[-1].strip():
                changes += 1
                continue
            cleaned.append("\n")
            continue

        # Check if this is a key=value line that duplicates a DEFAULT key
        if current_section and current_section != "DEFAULT" and '=' in stripped:
            # Parse key and value from the raw line
            eq_pos = stripped.index('=')
            key = stripped[:eq_pos].strip().lower()
            val = stripped[eq_pos + 1:].strip()

            if key in default_keys:
                # This key exists in DEFAULT — remove it from this section
                # since ConfigParser will inherit it automatically
                if dry_run:
                    print(f"  REMOVE: [{current_section}] {key} = {val}  (inherited from DEFAULT)")
                changes += 1
                continue

        # Comment or regular key line — keep it
        cleaned.append(line.rstrip() + "\n")

    # Remove trailing blank lines, ensure single newline at end
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    if cleaned:
        cleaned.append("\n")

    # Remove stray blank lines within sections (second pass)
    final = []
    for i, line in enumerate(cleaned):
        stripped = line.strip()
        # Skip blank line if the next line is NOT a section header and prev was content
        if not stripped:
            # Look ahead — keep blank only if next line is a section header
            if i + 1 < len(cleaned) and cleaned[i + 1].strip().startswith('['):
                final.append(line)
            # Skip blank lines within sections
            continue
        final.append(line)

    # Re-add blank lines between sections
    result = []
    for i, line in enumerate(final):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']') and result:
            result.append("\n")
        result.append(line)

    if not dry_run and changes > 0:
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"  Backup saved to {backup_path}")
        with open(filepath, 'w') as f:
            f.writelines(result)
        print(f"  Cleaned {filepath} ({changes} changes)")
    elif dry_run and changes > 0:
        print(f"  Would make {changes} changes to {filepath}")
    else:
        print(f"  {filepath}: already clean")

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge new config keys from template into deployed configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry run (show what would change)\n"
            "  python3 Tools/config_merge.py glog.conf dont_commit_glados.conf\n\n"
            "  # Apply changes\n"
            "  python3 Tools/config_merge.py glog.conf dont_commit_glados.conf --apply\n\n"
            "  # Multiple configs\n"
            "  python3 Tools/config_merge.py glog.conf pi4.conf pi5.conf --apply\n\n"
            "  # Clean up formatting of deployed configs\n"
            "  python3 Tools/config_merge.py glog.conf dont_commit_glados.conf --cleanup\n\n"
            "  # Dry run cleanup (show what would change)\n"
            "  python3 Tools/config_merge.py glog.conf dont_commit_glados.conf --cleanup --dry-run"
        ),
    )
    parser.add_argument("template", help="Template config file (e.g. glog.conf from repo)")
    parser.add_argument("deployed", nargs="+", help="Deployed config file(s) to update")
    parser.add_argument("--apply", action="store_true",
                        help="Apply merge changes (default: dry run)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up config formatting (fix spacing, remove DEFAULT dupes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what cleanup would change without modifying files")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        print(f"ERROR: Template file not found: {args.template}")
        sys.exit(1)

    template = load_config(args.template)

    print(f"Template: {args.template}")
    print(f"Sections: {template.sections()}")

    if args.cleanup:
        print("\n=== Config Cleanup ===")
        for deployed_path in args.deployed:
            if not os.path.exists(deployed_path):
                print(f"\nWARNING: File not found, skipping: {deployed_path}")
                continue
            print(f"\n--- {deployed_path} ---")
            clean_config(deployed_path, dry_run=args.dry_run)
        return

    total_changes = 0

    for deployed_path in args.deployed:
        if not os.path.exists(deployed_path):
            print(f"\nWARNING: Deployed file not found, skipping: {deployed_path}")
            continue

        deployed = load_config(deployed_path)
        diff = compare_configs(template, deployed)
        changes = print_diff(deployed_path, diff)
        total_changes += changes

        if args.apply and changes > 0:
            apply_merge(template, deployed_path, diff)

    if total_changes == 0:
        print("\nAll configs are up to date.")
    elif not args.apply:
        print(f"\n{total_changes} change(s) found. Run with --apply to merge.")


if __name__ == "__main__":
    main()

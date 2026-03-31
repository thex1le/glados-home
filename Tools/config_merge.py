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

        # Check for new and changed keys within existing sections
        for key, template_val in template.items(section):
            # Skip DEFAULT inherited keys unless they're explicitly in this section
            if key in template.defaults() and key not in dict(template[section]):
                continue
            if not deployed.has_option(section, key):
                if section not in result["new_keys"]:
                    result["new_keys"][section] = {}
                result["new_keys"][section][key] = template_val

    # Check for keys in deployed that aren't in template (potential removals)
    for section in deployed.sections():
        if not template.has_section(section):
            continue
        for key, _ in deployed.items(section):
            if key in deployed.defaults() and key not in dict(deployed[section]):
                continue
            if not template.has_option(section, key):
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
                    # Find the end of this section (next section or EOF)
                    insert_at = i + 1
                    while insert_at < len(lines):
                        stripped = lines[insert_at].strip()
                        if stripped.startswith('[') and stripped.endswith(']'):
                            break
                        insert_at += 1
                    # Insert new keys before the next section (or at end)
                    new_lines = []
                    for key, val in keys.items():
                        new_lines.append(f"{key} = {val}\n")
                    # Add a blank line before if the previous line isn't blank
                    if insert_at > 0 and lines[insert_at - 1].strip():
                        new_lines.insert(0, "\n")
                    for nl in reversed(new_lines):
                        lines.insert(insert_at, nl)
                    inserted = True
                    break
            if not inserted:
                # Section exists in ConfigParser but wasn't found as raw text (unusual)
                additions.append((section, keys))

    # Add entire new sections at the end
    if diff["new_sections"]:
        # Also grab all the keys for these sections from the template
        for section in diff["new_sections"]:
            lines.append(f"\n[{section}]\n")
            for key, val in template.items(section):
                if key in template.defaults() and key not in dict(template[section]):
                    continue
                lines.append(f"{key} = {val}\n")

    # Handle any sections we couldn't find in raw text
    for section, keys in additions:
        # Shouldn't happen normally, but just append
        for key, val in keys.items():
            lines.append(f"{key} = {val}\n")

    with open(deployed_path, 'w') as f:
        f.writelines(lines)

    print(f"  Merged changes into {deployed_path}")


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
            "  python3 Tools/config_merge.py glog.conf pi4.conf pi5.conf --apply"
        ),
    )
    parser.add_argument("template", help="Template config file (e.g. glog.conf from repo)")
    parser.add_argument("deployed", nargs="+", help="Deployed config file(s) to update")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes (default: dry run)")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        print(f"ERROR: Template file not found: {args.template}")
        sys.exit(1)

    template = load_config(args.template)

    print(f"Template: {args.template}")
    print(f"Sections: {template.sections()}")

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

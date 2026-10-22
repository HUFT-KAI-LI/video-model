#!/usr/bin/env bash
set -euo pipefail

scorers="${1:?scorer directory required}"
evidence="${2:?evidence directory required}"
unit=wan-said-vbench-prepare-paired100-20261022.service
mkdir -p "$evidence"
while [[ ! -s "$scorers/vbench_environment.json" ]]; do
    state="$(systemctl --user show "$unit" --property=ActiveState --value)"
    if [[ "$state" == failed || "$state" == inactive ]]; then
        journalctl --user -u "$unit" --no-pager --output=short-iso > "$evidence/vbench_prepare_failed.log"
        printf '%s\n' 'VBench preparation stopped without a verified environment' >&2
        exit 1
    fi
    sleep 30
done
install -m 644 "$scorers/vbench_environment.json" "$evidence/vbench_environment.json"
install -m 644 "$scorers/vbench_pip_freeze.txt" "$evidence/vbench_pip_freeze.log"
install -m 644 "$scorers/vbench_python" "$evidence/vbench_python.log"
for source in "$scorers"/*_source.json; do
    [[ -f "$source" ]] || continue
    install -m 644 "$source" "$evidence/$(basename "$source")"
done
journalctl --user -u "$unit" --no-pager --output=short-iso > "$evidence/vbench_prepare.log"
systemctl --user cat "$unit" > "$evidence/vbench_prepare_unit.log"

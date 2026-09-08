#!/usr/bin/env bash
# User scope supersedes the attachment: prepare only, no automatic experiments.
exec bash "$(dirname "$0")/prepare.sh" "$@"

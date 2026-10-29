#!/usr/bin/env bash
# BEGIN RUNNER CLEANUP: keep the old command as an alias for the unified run.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.bash" "$@"
# END RUNNER CLEANUP.

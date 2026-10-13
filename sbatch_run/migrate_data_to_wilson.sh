#!/usr/bin/env bash
# Run on Phoenix after stopping all jobs/processes that read or write this data.
# Copies and verifies data before switching paths. Never deletes either copy.
set -euo pipefail

source_data=/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/nkn_protein/data
destination_data=/storage/home/hcoda1/0/jlee3541/r-cwilson323-0/projects/nkn_protein/data
wilson_alias=/storage/home/hcoda1/0/jlee3541/r-cwilson323-0
wilson_project=/storage/project/r-cwilson323-0

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && $1 == --help ]]; then
        printf '%s\n' \
            'Usage: bash sbatch_run/migrate_data_to_wilson.sh' \
            'Stop all data-using jobs/processes before running this on Phoenix.' \
            "Source:      $source_data" \
            "Destination: $destination_data" \
            'Copies missing files, verifies checksums, and replaces source with a symlink.' \
            'Existing destination files are never overwritten; conflicts stop migration.' \
            'The original data is retained as a timestamped backup on Xie storage.' \
            'To copy current Wilson data back to Xie: bash sbatch_run/migrate_data_to_xie.sh'
        exit 0
    fi
    fail 'Only --help is accepted; the source and destination are fixed.'
fi

for command in rsync flock realpath find date mv ln mkdir; do
    command -v "$command" >/dev/null || fail "Required command not found: $command"
done

[[ -d $wilson_project && -d $wilson_alias ]] || fail 'Wilson project storage or its home alias is unavailable.'
project_real=$(realpath -e -- "$wilson_project")
alias_real=$(realpath -e -- "$wilson_alias")
[[ $alias_real == "$project_real" ]] || fail "Wilson alias resolves to $alias_real, not $project_real."
[[ ! -L $destination_data ]] || fail 'Destination data must be a real directory, not a symlink.'
destination_real=$(realpath -m -- "$destination_data")
[[ $destination_real == "$project_real/projects/nkn_protein/data" ]] || fail "Unexpected destination path: $destination_real"

source_parent=${source_data%/*}
[[ -d $source_parent ]] || fail "Source repository does not exist: $source_parent"
exec 9>"$source_parent/.migrate_data_to_wilson.lock"
flock -n 9 || fail 'Another migration helper is running for this repository.'

if [[ -L $source_data ]]; then
    [[ -d $destination_data ]] || fail 'Source is a symlink but the destination directory is missing.'
    source_real=$(realpath -e -- "$source_data")
    [[ $source_real == "$destination_real" ]] || fail "Source already points elsewhere: $source_real"
    printf 'Already configured: %s -> %s\n' "$source_data" "$destination_data"
    exit 0
fi
[[ -d $source_data ]] || fail "Source data directory is missing: $source_data"
source_real=$(realpath -e -- "$source_data")
[[ $source_real != "$destination_real" ]] || fail 'Source and destination are the same directory.'
[[ ! -e $destination_data || -d $destination_data ]] || fail 'Destination exists but is not a directory.'

# This helper deliberately avoids changing or dereferencing links inside data.
# External structure references also need checking before migration; see README.
first_link=$(find "$source_data" -type l -print -quit)
[[ -z $first_link ]] || fail "Source contains a symlink; review it before migrating: $first_link"
if [[ -d $destination_data ]]; then
    first_link=$(find "$destination_data" -type l -print -quit)
    [[ -z $first_link ]] || fail "Destination contains a symlink; review it before migrating: $first_link"
fi

printf 'Keep all data-using jobs/processes stopped until this command finishes.\n'
printf 'Copying %s\n      to %s\n' "$source_data" "$destination_data"
mkdir -p -- "$destination_data"
[[ -w $destination_data ]] || fail "Destination is not writable: $destination_data"
# Keep destination ownership, group, and default permissions (including setgid).
# Existing files remain untouched, allowing an interrupted copy to be retried.
rsync -a --no-owner --no-group --no-perms --ignore-existing --info=progress2 \
    -- "$source_data/" "$destination_data/"

printf 'Comparing every source file with the destination using checksums...\n'
differences=$(rsync -rnc --itemize-changes --out-format='%i %n' \
    -- "$source_data/" "$destination_data/")
if [[ -n $differences ]]; then
    printf '%s\n' "$differences" >&2
    fail 'Missing or conflicting destination files remain. Source is unchanged; resolve conflicts before retrying.'
fi

backup_data="${source_data}.before-wilson-$(date -u +%Y%m%dT%H%M%SZ)-$$"
[[ ! -e $backup_data && ! -L $backup_data ]] || fail "Backup path already exists: $backup_data"
restore_unlinked_source() {
    migration_status=$?
    if [[ -d $backup_data && ! -e $source_data && ! -L $source_data ]]; then
        printf 'Restoring the original data path after an interrupted switch.\n' >&2
        mv -T -- "$backup_data" "$source_data" || \
            printf 'Rollback failed; original data remains at %s\n' "$backup_data" >&2
    fi
    return "$migration_status"
}
trap restore_unlinked_source EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
mv -T -- "$source_data" "$backup_data"
if ! ln -sT -- "$destination_data" "$source_data"; then
    printf 'Symlink creation failed; restoring the original data directory.\n' >&2
    if ! mv -T -- "$backup_data" "$source_data"; then
        fail "Automatic rollback failed. Original data is safe at $backup_data; restore it manually."
    fi
    fail 'Migration was rolled back. The destination copy remains available.'
fi

[[ -L $source_data ]] || fail "Expected a data symlink. Original data remains at $backup_data."
linked_real=$(realpath -e -- "$source_data") || fail "Cannot resolve the new link. Original data remains at $backup_data."
[[ $linked_real == "$destination_real" ]] || fail "Unexpected new link target: $linked_real. Original data remains at $backup_data."
printf '\nData now resolves to: %s\n' "$linked_real"
printf 'Original data backup: %s\n' "$backup_data"
printf '%s\n' \
    'New data writes through the repository data/ path now use Wilson storage.' \
    'The backup still uses Xie quota. Remove it manually only after validating your jobs.'

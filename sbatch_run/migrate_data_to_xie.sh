#!/usr/bin/env bash
# Run on Phoenix after stopping all jobs/processes that read or write this data.
# Copies current Wilson data back to Xie. Never deletes Wilson data or old backups.
set -euo pipefail

source_data=/storage/home/hcoda1/0/jlee3541/r-cwilson323-0/projects/nkn_protein/data
destination_data=/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/nkn_protein/data
wilson_alias=/storage/home/hcoda1/0/jlee3541/r-cwilson323-0
wilson_project=/storage/project/r-cwilson323-0
xie_alias=/storage/home/hcoda1/0/jlee3541/r-yxie77-0
xie_project=/storage/project/r-yxie77-0

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && $1 == --help ]]; then
        printf '%s\n' \
            'Usage: bash sbatch_run/migrate_data_to_xie.sh' \
            'Stop all data-using jobs/processes before running this on Phoenix.' \
            "Source:      $source_data" \
            "Destination: $destination_data" \
            'Requires enough free Xie quota for a complete copy of current Wilson data.' \
            'Copies into a fresh staging directory, verifies checksums, then switches.' \
            'Wilson data, the previous link, and old Xie backups are retained.' \
            'Failed copies remain in the printed staging directory for inspection.'
        exit 0
    fi
    fail 'Only --help is accepted; the source and destination are fixed.'
fi

for command in rsync flock realpath find date mv mktemp chmod; do
    command -v "$command" >/dev/null || fail "Required command not found: $command"
done

[[ -d $xie_project && -d $xie_alias ]] || fail 'Xie project storage or its home alias is unavailable.'
xie_real=$(realpath -e -- "$xie_project")
alias_real=$(realpath -e -- "$xie_alias")
[[ $alias_real == "$xie_real" ]] || fail "Xie alias resolves to $alias_real, not $xie_real."
destination_parent=${destination_data%/*}
[[ -d $destination_parent ]] || fail "Xie repository does not exist: $destination_parent"
parent_real=$(realpath -e -- "$destination_parent")
[[ $parent_real == "$xie_real/projects/nkn_protein" ]] || fail "Unexpected Xie repository path: $parent_real"
destination_real="$parent_real/data"
exec 9>"$destination_parent/.migrate_data_to_wilson.lock"
flock -n 9 || fail 'Another migration helper is running for this repository.'

if [[ ! -L $destination_data && -d $destination_data ]]; then
    current_real=$(realpath -e -- "$destination_data")
    [[ $current_real == "$destination_real" ]] || fail "Unexpected active data path: $current_real"
    printf 'Already configured: data is a real directory on Xie at %s\n' "$current_real"
    exit 0
fi
[[ -L $destination_data ]] || fail 'Expected the Xie data path to be a symlink to Wilson.'
[[ -d $wilson_project && -d $wilson_alias ]] || fail 'Wilson project storage or its home alias is unavailable.'
wilson_real=$(realpath -e -- "$wilson_project")
alias_real=$(realpath -e -- "$wilson_alias")
[[ $alias_real == "$wilson_real" ]] || fail "Wilson alias resolves to $alias_real, not $wilson_real."
[[ ! -L $source_data && -d $source_data ]] || fail 'Wilson data must be an existing real directory.'
source_real=$(realpath -e -- "$source_data")
[[ $source_real == "$wilson_real/projects/nkn_protein/data" ]] || fail "Unexpected Wilson data path: $source_real"
current_real=$(realpath -e -- "$destination_data")
[[ $current_real == "$source_real" ]] || fail "Xie data points elsewhere: $current_real"
first_link=$(find "$source_data" -type l -print -quit)
[[ -z $first_link ]] || fail "Wilson data contains a symlink; review it before migrating: $first_link"
[[ -w $destination_parent ]] || fail "Xie repository is not writable: $destination_parent"

printf '%s\n' \
    'Keep all data-using jobs/processes stopped until this command finishes.' \
    'Xie must have enough free quota for a complete copy; old backups also use quota.'
staging_data=$(mktemp -d -- "${destination_data}.return-from-wilson.XXXXXXXX")
printf 'Staging directory (retained on failure): %s\n' "$staging_data"
# mktemp starts with mode 700; use the repository access mode for the new root.
chmod --reference="$destination_parent" -- "$staging_data"
printf 'Copying current Wilson data from %s\n' "$source_data"
rsync -a --no-owner --no-group --no-perms --info=progress2 \
    -- "$source_data/" "$staging_data/"
printf 'Comparing every source file with the copy using checksums...\n'
differences=$(rsync -rnc --delete --itemize-changes --out-format='%i %n' \
    -- "$source_data/" "$staging_data/")
if [[ -n $differences ]]; then
    printf '%s\n' "$differences" >&2
    fail "Verification failed. The active link is unchanged; the copy remains at $staging_data."
fi

backup_link="${destination_data}.wilson-link-$(date -u +%Y%m%dT%H%M%SZ)-$$"
[[ ! -e $backup_link && ! -L $backup_link ]] || fail "Backup link path already exists: $backup_link"
restore_unlinked_destination() {
    migration_status=$?
    if [[ -L $backup_link && ! -e $destination_data && ! -L $destination_data ]]; then
        printf 'Restoring the Wilson link after an interrupted switch.\n' >&2
        mv -T -- "$backup_link" "$destination_data" || \
            printf 'Rollback failed; restore the link from %s\n' "$backup_link" >&2
    fi
    return "$migration_status"
}
trap restore_unlinked_destination EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
# Recheck the active link immediately before the two-step switch.
[[ -L $destination_data ]] || fail 'The active data path changed during copying.'
[[ $(realpath -e -- "$destination_data") == "$source_real" ]] || fail 'The active link target changed during copying.'
mv -T -- "$destination_data" "$backup_link"
mv -T -- "$staging_data" "$destination_data"

[[ ! -L $destination_data && -d $destination_data ]] || fail "Expected a real Xie data directory; retained link: $backup_link"
current_real=$(realpath -e -- "$destination_data")
[[ $current_real == "$destination_real" ]] || fail "Unexpected final data path: $current_real"
printf '\nData now resolves to: %s\n' "$current_real"
printf 'Previous Wilson link: %s\n' "$backup_link"
printf 'Wilson data retained: %s\n' "$source_data"
printf '%s\n' \
    'New data writes through the repository data/ path now use Xie storage.' \
    'The Wilson copy and old Xie backups still use quota; they are not kept in sync.'

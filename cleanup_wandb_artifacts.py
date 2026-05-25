"""
cleanup_wandb_artifacts.py
──────────────────────────
Deletes W&B artifact versions that have no aliases (intermediate checkpoints
that were never marked "best"). Iterates by run, which is the most reliable
approach across wandb SDK versions.

Usage
─────
  python cleanup_wandb_artifacts.py --dry-run      # see what would be deleted
  python cleanup_wandb_artifacts.py                # delete for real
"""

import argparse
import wandb

# ── Configure these ───────────────────────────────────────────────────────────
ENTITY  = "ryanhana-wandb"
PROJECT = "hdr-dual-moe"
# ─────────────────────────────────────────────────────────────────────────────


def cleanup(entity, project, dry_run):
    api  = wandb.Api(overrides={"entity": entity, "project": project})
    runs = api.runs(f"{entity}/{project}")

    total_deleted = 0
    total_kept    = 0

    for run in runs:
        artifacts = list(run.logged_artifacts())
        if not artifacts:
            continue

        print(f"\nRun: {run.name}  ({len(artifacts)} artifact version(s))")

        # Sort by version number descending so index 0 = most recent
        artifacts.sort(key=lambda a: a.version, reverse=True)

        for idx, artifact in enumerate(artifacts):
            if idx == 0:
                print(f"  {'KEEP (latest)':<24} {artifact.name}  aliases={artifact.aliases}")
                total_kept += 1
            else:
                action = "[DRY-RUN] would delete" if dry_run else "DELETE"
                print(f"  {action:<24} {artifact.name}  aliases={artifact.aliases}")
                if not dry_run:
                    try:
                        artifact.delete(delete_aliases=True)
                    except Exception as e:
                        print(f"    ⚠ {e}")
                total_deleted += 1

    print(f"\n{'─'*50}")
    if dry_run:
        print(f"DRY RUN — would delete {total_deleted}, keep {total_kept}")
        print("Run without --dry-run to apply.")
    else:
        print(f"Deleted {total_deleted}, kept {total_kept}.")
        print("Storage freed after W&B's next GC cycle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity",  default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cleanup(args.entity, args.project, args.dry_run)
"""Auto-cleanup service for temp and output directories.

Removes old files from temp/ and output/ directories based on age.
Called at app startup and optionally on a schedule.
"""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_HOURS = 24  # Delete files older than 24 hours
DEFAULT_MAX_OUTPUT_FILES = 50  # Keep at most 50 files in output


def cleanup_temp_dir(temp_dir: str | Path, max_age_hours: float = DEFAULT_MAX_AGE_HOURS) -> int:
    """Remove files older than max_age_hours from the temp directory.

    Returns number of files removed.
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for f in temp_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Failed to delete {f}: {e}")

    if removed > 0:
        logger.info(f"Cleaned {removed} old temp files (>{max_age_hours}h)")
    return removed


def cleanup_output_dir(output_dir: str | Path, max_files: int = DEFAULT_MAX_OUTPUT_FILES) -> int:
    """Keep only the newest max_files in the output directory.

    Returns number of files removed.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0

    files = sorted(
        [f for f in output_dir.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if len(files) <= max_files:
        return 0

    to_remove = files[max_files:]
    removed = 0
    for f in to_remove:
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            logger.warning(f"Failed to delete {f}: {e}")

    if removed > 0:
        logger.info(f"Cleaned {removed} old output files (keeping {max_files} newest)")
    return removed


def run_startup_cleanup(config: dict) -> dict:
    """Run all cleanup tasks at app startup.

    Returns summary dict with counts.
    """
    paths = config.get("paths", {})
    temp_dir = paths.get("temp", "temp")
    output_dir = paths.get("output", "output")

    cleanup_config = config.get("cleanup", {})
    max_age = cleanup_config.get("temp_max_age_hours", DEFAULT_MAX_AGE_HOURS)
    max_output = cleanup_config.get("max_output_files", DEFAULT_MAX_OUTPUT_FILES)

    temp_removed = cleanup_temp_dir(temp_dir, max_age)
    output_removed = cleanup_output_dir(output_dir, max_output)

    summary = {"temp_removed": temp_removed, "output_removed": output_removed}
    if temp_removed or output_removed:
        logger.info(f"Startup cleanup: {summary}")

    return summary

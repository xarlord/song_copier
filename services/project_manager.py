"""Project / stem library manager — save and reload separation sessions."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECTS_DIR = Path("projects")


def _ensure_projects_dir() -> Path:
    """Create the projects directory if it doesn't exist."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def _sanitize_name(name: str) -> str:
    """Sanitize a project name for use as a filename."""
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip()
    return safe or "untitled_project"


def save_project(
    name: str,
    original_file: str,
    stems_dir: str,
    metadata: dict,
) -> str:
    """Save a project metadata JSON to the projects/ directory.

    Args:
        name: Human-readable project name.
        original_file: Path to the original audio file.
        stems_dir: Directory containing separated stem files.
        metadata: Additional metadata dict. May include:
            - bpm (float): from analysis service
            - key (str): from analysis service
            - duration (float): in seconds
            - stem_files (dict): mapping stem name -> file path
            - model (str): separation model used
            - format (str): output format

    Returns:
        Path to the saved project JSON file.
    """
    _ensure_projects_dir()
    safe_name = _sanitize_name(name)

    # Discover stem files from stems_dir if not explicitly provided
    stem_files = metadata.get("stem_files", {})
    if not stem_files and stems_dir:
        stems_path = Path(stems_dir)
        if stems_path.exists():
            for f in sorted(stems_path.iterdir()):
                if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                    stem_files[f.stem] = str(f)

    project_data = {
        "name": name,
        "safe_name": safe_name,
        "original_file": str(original_file),
        "original_filename": Path(original_file).name if original_file else "",
        "stems_dir": str(stems_dir),
        "date": datetime.now().isoformat(),
        "bpm": metadata.get("bpm"),
        "key": metadata.get("key"),
        "duration": metadata.get("duration"),
        "model": metadata.get("model", ""),
        "format": metadata.get("format", "wav"),
        "stem_files": stem_files,
        "stem_count": len(stem_files),
        "stems_list": list(stem_files.keys()),
    }

    # Preserve any extra metadata keys
    for k, v in metadata.items():
        if k not in project_data:
            project_data[k] = v

    path = PROJECTS_DIR / f"{safe_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved project: {safe_name} ({len(stem_files)} stems)")
    return str(path)


def list_projects() -> list[dict]:
    """List all saved projects with metadata.

    Returns:
        List of dicts, each with keys:
            name, original_filename, date, bpm, key, stem_count,
            safe_name, path
    """
    _ensure_projects_dir()
    projects = []
    for f in sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            projects.append({
                "name": data.get("name", f.stem),
                "safe_name": data.get("safe_name", f.stem),
                "original_filename": data.get("original_filename", ""),
                "date": data.get("date", ""),
                "bpm": data.get("bpm"),
                "key": data.get("key", ""),
                "stem_count": data.get("stem_count", 0),
                "duration": data.get("duration"),
                "path": str(f),
            })
        except Exception as e:
            logger.warning(f"Skipping broken project file {f}: {e}")
    return projects


def load_project(name: str) -> dict | None:
    """Load project metadata and verify stem files still exist.

    Args:
        name: Project name or safe_name.

    Returns:
        Project dict with an added 'stem_status' key mapping each stem
        to True/False based on file existence, or None if not found.
    """
    _ensure_projects_dir()

    # Try exact safe_name first, then search by name
    safe_name = _sanitize_name(name)
    path = PROJECTS_DIR / f"{safe_name}.json"

    if not path.exists():
        # Search by name field inside JSON files
        for f in PROJECTS_DIR.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("name") == name or data.get("safe_name") == name:
                    path = f
                    break
            except Exception:
                continue
        else:
            logger.warning(f"Project not found: {name}")
            return None

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.error(f"Failed to load project {name}: {e}")
        return None

    # Verify stem files
    stem_status = {}
    verified_stems = {}
    for stem_name, stem_path in data.get("stem_files", {}).items():
        exists = Path(stem_path).exists()
        stem_status[stem_name] = exists
        if exists:
            verified_stems[stem_name] = stem_path

    data["stem_status"] = stem_status
    data["verified_stems"] = verified_stems
    data["missing_stems"] = [
        s for s, ok in stem_status.items() if not ok
    ]

    return data


def delete_project(name: str) -> bool:
    """Delete a project (metadata only, not stem files).

    Args:
        name: Project name or safe_name.

    Returns:
        True if deleted, False if not found.
    """
    _ensure_projects_dir()

    safe_name = _sanitize_name(name)
    path = PROJECTS_DIR / f"{safe_name}.json"

    if not path.exists():
        # Try searching by name field
        for f in PROJECTS_DIR.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("name") == name:
                    path = f
                    break
            except Exception:
                continue
        else:
            logger.warning(f"Project not found for deletion: {name}")
            return False

    path.unlink(missing_ok=True)
    logger.info(f"Deleted project: {name}")
    return True


def search_projects(query: str) -> list[dict]:
    """Search projects by name, filename, or BPM range.

    Args:
        query: Search string. Supports:
            - Plain text: matches project name or original filename (case-insensitive)
            - BPM range: "bpm:120-140" or "bpm:128"

    Returns:
        List of matching project dicts (same format as list_projects).
    """
    all_projects = list_projects()
    if not query or not query.strip():
        return all_projects

    query = query.strip()

    # Check for BPM range query: "bpm:120-140" or "bpm:128"
    bpm_match = re.match(r"bpm:(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?", query, re.IGNORECASE)
    if bpm_match:
        bpm_low = float(bpm_match.group(1))
        bpm_high = float(bpm_match.group(2)) if bpm_match.group(2) else bpm_low
        results = []
        for p in all_projects:
            bpm = p.get("bpm")
            if bpm is not None and bpm_low <= bpm <= bpm_high:
                results.append(p)
        return results

    # Text search on name and original_filename
    query_lower = query.lower()
    results = []
    for p in all_projects:
        searchable = f"{p.get('name', '')} {p.get('original_filename', '')}".lower()
        if query_lower in searchable:
            results.append(p)

    return results

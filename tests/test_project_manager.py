"""Tests for services/project_manager.py — save, load, list, delete projects."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from services.project_manager import (
    save_project,
    load_project,
    list_projects,
    delete_project,
    search_projects,
    _sanitize_name,
    PROJECTS_DIR,
)


@pytest.fixture(autouse=True)
def use_tmp_projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_DIR to tmp_path for isolation."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr("services.project_manager.PROJECTS_DIR", projects_dir)
    return projects_dir


class TestSanitizeName:
    """Tests for _sanitize_name."""

    def test_normal_name(self):
        """Normal name should pass through."""
        assert _sanitize_name("my_project") == "my_project"

    def test_special_characters_removed(self):
        """Special characters should be removed."""
        result = _sanitize_name("my/project<>test")
        assert "/" not in result
        assert "<" not in result
        assert ">" not in result

    def test_empty_name_gets_default(self):
        """Empty string should become 'untitled_project'."""
        result = _sanitize_name("")
        assert result == "untitled_project"

    def test_only_special_chars_gets_default(self):
        """String of only special chars should become 'untitled_project'."""
        result = _sanitize_name("///:::!!!")
        assert result == "untitled_project"

    def test_spaces_preserved(self):
        """Spaces should be preserved."""
        result = _sanitize_name("my cool project")
        assert "my cool project" == result


class TestSaveProject:
    """Tests for save_project."""

    def test_save_creates_json(self, use_tmp_projects_dir, sample_project_dir):
        """save_project should create a JSON file."""
        path = save_project(
            "test_project",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {"bpm": 120.0, "key": "C major"},
        )
        assert Path(path).exists()
        assert Path(path).suffix == ".json"

    def test_save_contains_expected_fields(self, use_tmp_projects_dir, sample_project_dir):
        """Saved JSON should contain name, stem_files, etc."""
        path = save_project(
            "test_fields",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {"bpm": 128.0},
        )
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "test_fields"
        assert data["bpm"] == 128.0

    def test_save_discovers_stem_files(self, use_tmp_projects_dir, sample_project_dir):
        """When stem_files not provided, should discover from stems_dir."""
        path = save_project(
            "discovered",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {},
        )
        with open(path) as f:
            data = json.load(f)
        assert data["stem_count"] > 0


class TestLoadProject:
    """Tests for load_project."""

    def test_load_existing_project(self, use_tmp_projects_dir, sample_project_dir):
        """Load a saved project should return the data dict."""
        save_project(
            "load_test",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {"bpm": 120.0},
        )
        data = load_project("load_test")
        assert data is not None
        assert data["name"] == "load_test"

    def test_load_nonexistent_returns_none(self, use_tmp_projects_dir):
        """Loading non-existent project should return None."""
        data = load_project("nonexistent_project_xyz")
        assert data is None

    def test_load_has_stem_status(self, use_tmp_projects_dir, sample_project_dir):
        """Loaded project should have stem_status key."""
        save_project(
            "status_test",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {},
        )
        data = load_project("status_test")
        assert "stem_status" in data


class TestListProjects:
    """Tests for list_projects."""

    def test_empty_list(self, use_tmp_projects_dir):
        """Empty projects dir should return empty list."""
        result = list_projects()
        assert result == []

    def test_list_after_save(self, use_tmp_projects_dir, sample_project_dir):
        """List should include saved projects."""
        save_project(
            "list_test",
            str(sample_project_dir / "original.wav"),
            str(sample_project_dir),
            {},
        )
        result = list_projects()
        assert len(result) >= 1
        assert result[0]["name"] == "list_test"

    def test_list_sorted_by_date(self, use_tmp_projects_dir, sample_project_dir):
        """Most recent project should appear first."""
        import time
        save_project("older", "", "", {})
        time.sleep(0.1)
        save_project("newer", "", "", {})
        result = list_projects()
        assert result[0]["name"] == "newer"


class TestDeleteProject:
    """Tests for delete_project."""

    def test_delete_existing(self, use_tmp_projects_dir, sample_project_dir):
        """Delete should remove the project file."""
        save_project("to_delete", "", "", {})
        assert delete_project("to_delete") is True
        assert load_project("to_delete") is None

    def test_delete_nonexistent(self, use_tmp_projects_dir):
        """Deleting non-existent project should return False."""
        assert delete_project("nonexistent") is False


class TestSearchProjects:
    """Tests for search_projects."""

    def test_search_by_name(self, use_tmp_projects_dir, sample_project_dir):
        """Search by name should find matching projects."""
        save_project("rock_anthem", "", "", {"bpm": 140.0})
        save_project("jazz_ballad", "", "", {"bpm": 85.0})
        result = search_projects("rock")
        assert len(result) == 1
        assert result[0]["name"] == "rock_anthem"

    def test_search_empty_returns_all(self, use_tmp_projects_dir, sample_project_dir):
        """Empty query should return all projects."""
        save_project("proj1", "", "", {})
        save_project("proj2", "", "", {})
        result = search_projects("")
        assert len(result) == 2

    def test_search_bpm_range(self, use_tmp_projects_dir, sample_project_dir):
        """BPM range search should filter by tempo."""
        save_project("fast", "", "", {"bpm": 150.0})
        save_project("slow", "", "", {"bpm": 80.0})
        result = search_projects("bpm:100-160")
        assert len(result) == 1
        assert result[0]["name"] == "fast"

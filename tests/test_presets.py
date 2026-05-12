"""Tests for services/presets.py — save, load, list, delete presets."""

import pytest
from pathlib import Path

from services.presets import (
    save_preset,
    load_preset,
    list_presets,
    delete_preset,
    PRESETS_DIR,
)


@pytest.fixture(autouse=True)
def use_tmp_presets_dir(tmp_path, monkeypatch):
    """Redirect PRESETS_DIR to tmp_path for isolation."""
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    monkeypatch.setattr("services.presets.PRESETS_DIR", presets_dir)
    return presets_dir


class TestSavePreset:
    """Tests for save_preset."""

    def test_save_creates_file(self, use_tmp_presets_dir):
        """save_preset should create a YAML file."""
        path = save_preset("test_preset", "A test preset", {"vocals": {"volume": 0.8}})
        assert Path(path).exists()
        assert Path(path).suffix == ".yaml"

    def test_save_with_stems_data(self, use_tmp_presets_dir):
        """Preset should contain stems data."""
        stems = {
            "vocals": {"volume": 0.9, "effects": {"eq": {"low": 0, "mid": 0, "high": 0}}},
            "drums": {"volume": 0.7},
        }
        path = save_preset("stem_test", "Stem test", stems)
        loaded = load_preset("stem_test")
        assert loaded is not None
        assert "stems" in loaded
        assert loaded["stems"]["vocals"]["volume"] == 0.9

    def test_special_chars_sanitized(self, use_tmp_presets_dir):
        """Special characters in name should be sanitized."""
        path = save_preset("test/preset<>", "Special chars", {})
        assert Path(path).exists()

    def test_empty_name_gets_default(self, use_tmp_presets_dir):
        """Empty name should get default name."""
        path = save_preset("", "Empty name test", {})
        assert Path(path).exists()


class TestLoadPreset:
    """Tests for load_preset."""

    def test_load_existing(self, use_tmp_presets_dir):
        """Load a saved preset."""
        save_preset("load_test", "Loadable", {"vocals": {"volume": 0.8}})
        data = load_preset("load_test")
        assert data is not None
        assert data["name"] == "load_test"

    def test_load_nonexistent(self, use_tmp_presets_dir):
        """Loading non-existent preset should return None."""
        data = load_preset("nonexistent_preset")
        assert data is None

    def test_load_preserves_description(self, use_tmp_presets_dir):
        """Description should be preserved."""
        save_preset("desc_test", "My description", {})
        data = load_preset("desc_test")
        assert data["description"] == "My description"


class TestListPresets:
    """Tests for list_presets."""

    def test_empty_list(self, use_tmp_presets_dir):
        """Empty presets dir should return empty list."""
        result = list_presets()
        assert result == []

    def test_list_after_save(self, use_tmp_presets_dir):
        """List should include saved presets."""
        save_preset("list1", "First", {})
        save_preset("list2", "Second", {})
        result = list_presets()
        assert len(result) == 2

    def test_list_has_expected_keys(self, use_tmp_presets_dir):
        """Each preset in list should have name, description, custom, path."""
        save_preset("keys_test", "Test", {})
        result = list_presets()
        assert len(result) >= 1
        for key in ("name", "description", "custom", "path"):
            assert key in result[0]


class TestDeletePreset:
    """Tests for delete_preset."""

    def test_delete_existing(self, use_tmp_presets_dir):
        """Delete should remove the preset file."""
        save_preset("to_delete", "Delete me", {})
        assert delete_preset("to_delete") is True
        assert load_preset("to_delete") is None

    def test_delete_nonexistent(self, use_tmp_presets_dir):
        """Deleting non-existent preset should return False."""
        assert delete_preset("nonexistent") is False

    def test_delete_builtin_blocked(self, use_tmp_presets_dir):
        """Built-in preset (custom=False) cannot be deleted."""
        # Manually create a built-in preset
        import yaml
        presets_dir = use_tmp_presets_dir
        data = {"name": "builtin", "description": "Built-in", "custom": False, "stems": {}}
        with open(presets_dir / "builtin.yaml", "w") as f:
            yaml.dump(data, f)
        assert delete_preset("builtin") is False

    def test_delete_then_list(self, use_tmp_presets_dir):
        """After deletion, list should not contain the preset."""
        save_preset("temp", "Temp", {})
        delete_preset("temp")
        result = list_presets()
        names = [p["name"] for p in result]
        assert "temp" not in names

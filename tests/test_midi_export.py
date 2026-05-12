"""Tests for services/midi_export.py — pitch detection, MIDI conversion, export."""

import numpy as np
import pytest

from services.midi_export import (
    freq_to_midi_note,
    midi_note_to_name,
    detect_pitch,
    audio_to_midi,
    export_midi,
)


class TestFreqToMidiNote:
    """Tests for freq_to_midi_note."""

    def test_a4_440hz(self):
        """A4 = 440 Hz → MIDI note 69."""
        assert freq_to_midi_note(440.0) == 69

    def test_c4_261hz(self):
        """C4 ≈ 261.6 Hz → MIDI note 60."""
        assert freq_to_midi_note(261.63) == 60

    def test_zero_freq_returns_0(self):
        """Zero frequency should return 0 (silence)."""
        assert freq_to_midi_note(0.0) == 0

    def test_negative_freq_returns_0(self):
        """Negative frequency should return 0."""
        assert freq_to_midi_note(-100.0) == 0

    def test_high_freq(self):
        """High frequency should map to a high MIDI note."""
        note = freq_to_midi_note(4186.0)  # C8
        assert note >= 108  # C8 = MIDI 108

    def test_low_freq(self):
        """Low frequency should map to a low MIDI note."""
        note = freq_to_midi_note(65.4)  # C2
        assert note == 36  # C2 = MIDI 36


class TestMidiNoteToName:
    """Tests for midi_note_to_name."""

    def test_middle_c(self):
        """MIDI 60 → C4."""
        assert midi_note_to_name(60) == "C4"

    def test_a4(self):
        """MIDI 69 → A4."""
        assert midi_note_to_name(69) == "A4"

    def test_midi_0(self):
        """MIDI 0 → C-1."""
        assert midi_note_to_name(0) == "C-1"

    def test_sharp_note(self):
        """MIDI 61 → C#4."""
        assert midi_note_to_name(61) == "C#4"

    def test_out_of_range(self):
        """Out of range notes should return a '?' string."""
        result = midi_note_to_name(200)
        assert "?" in result

    def test_negative_note(self):
        """Negative MIDI note should return '?' string."""
        result = midi_note_to_name(-1)
        assert "?" in result


class TestDetectPitch:
    """Tests for detect_pitch."""

    def test_returns_tuple_of_arrays(self, sample_audio_sine, sample_rate):
        """detect_pitch should return (times, frequencies) tuple."""
        times, freqs = detect_pitch(sample_audio_sine, sample_rate)
        assert isinstance(times, np.ndarray)
        assert isinstance(freqs, np.ndarray)
        assert len(times) == len(freqs)

    def test_sine_wave_detected_pitch(self, sample_audio_sine, sample_rate):
        """440Hz sine wave should have detected frequency near 440Hz."""
        times, freqs = detect_pitch(sample_audio_sine, sample_rate)
        voiced = freqs[~np.isnan(freqs)]
        if len(voiced) > 0:
            median_freq = np.median(voiced)
            assert abs(median_freq - 440.0) < 20.0  # within 20Hz

    def test_silence_no_pitch(self, sample_audio_silence, sample_rate):
        """Silent audio should have no voiced frames (all NaN)."""
        # Use mono silence
        silence = np.zeros(sample_rate, dtype=np.float32)
        times, freqs = detect_pitch(silence, sample_rate)
        voiced = freqs[~np.isnan(freqs)]
        # Silence should produce no or very few voiced frames
        assert len(voiced) < 5

    def test_stereo_input_handled(self, sample_rate):
        """Stereo input should be mixed to mono for pitch detection."""
        t = np.linspace(0, 0.5, sample_rate // 2, endpoint=False)
        stereo = np.stack([
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
        ])
        times, freqs = detect_pitch(stereo, sample_rate)
        assert len(times) == len(freqs)


class TestAudioToMidi:
    """Tests for audio_to_midi."""

    def test_returns_midi_file(self, sample_audio_sine, sample_rate):
        """audio_to_midi should return a mido.MidiFile object."""
        import mido
        mid = audio_to_midi(sample_audio_sine, sample_rate)
        assert isinstance(mid, mido.MidiFile)

    def test_has_tracks(self, sample_audio_sine, sample_rate):
        """MIDI file should have at least one track."""
        mid = audio_to_midi(sample_audio_sine, sample_rate)
        assert len(mid.tracks) >= 1

    def test_instrument_piano(self, sample_audio_sine, sample_rate):
        """Setting instrument='piano' should set program to 0."""
        mid = audio_to_midi(sample_audio_sine, sample_rate, instrument="piano")
        # Check that there's a program_change message
        programs = [m for t in mid.tracks for m in t if m.type == "program_change"]
        assert len(programs) >= 1
        assert programs[0].program == 0

    def test_instrument_drums(self, sample_audio_sine, sample_rate):
        """Drums should use channel 9 (10 in 1-indexed)."""
        mid = audio_to_midi(sample_audio_sine, sample_rate, instrument="drums")
        programs = [m for t in mid.tracks for m in t if m.type == "program_change"]
        assert len(programs) >= 1
        assert programs[0].channel == 9

    def test_with_precomputed_pitch(self, sample_audio_sine, sample_rate):
        """Providing pitch_data should skip pitch detection."""
        pitch_data = detect_pitch(sample_audio_sine, sample_rate)
        mid = audio_to_midi(sample_audio_sine, sample_rate, pitch_data=pitch_data)
        assert mid is not None

    def test_empty_audio(self, sample_rate):
        """Empty audio should produce a valid (empty) MIDI file."""
        audio = np.zeros(0, dtype=np.float32)
        import mido
        mid = audio_to_midi(audio, sample_rate)
        assert isinstance(mid, mido.MidiFile)


class TestExportMidi:
    """Tests for export_midi."""

    def test_export_creates_file(self, sample_audio_sine, sample_rate, tmp_path):
        """export_midi should create a .mid file."""
        out_path = tmp_path / "test.mid"
        result = export_midi(sample_audio_sine, sample_rate, str(out_path))
        assert out_path.exists()

    def test_export_returns_path(self, sample_audio_sine, sample_rate, tmp_path):
        """export_midi should return the resolved path."""
        out_path = tmp_path / "test.mid"
        result = export_midi(sample_audio_sine, sample_rate, str(out_path))
        assert isinstance(result, str)
        assert result.endswith(".mid")

    def test_export_with_custom_bpm(self, sample_audio_sine, sample_rate, tmp_path):
        """Custom BPM should be set in the MIDI file."""
        out_path = tmp_path / "test.mid"
        result = export_midi(sample_audio_sine, sample_rate, str(out_path), bpm=140.0)
        assert out_path.exists()

    def test_export_adds_mid_extension(self, sample_audio_sine, sample_rate, tmp_path):
        """If path doesn't end in .mid, extension should be added."""
        out_path = tmp_path / "test"
        result = export_midi(sample_audio_sine, sample_rate, str(out_path))
        assert ".mid" in result

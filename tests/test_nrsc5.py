#!/usr/bin/env python3
"""Unit tests for nrsc5 process hooks."""

import io
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from nrsc5 import NRSC5Demodulator


def test_nrsc5_unavailable_reports_reason(monkeypatch):
    monkeypatch.setattr(
        NRSC5Demodulator,
        "_load_python_bindings",
        lambda self: (None, "", "", "bindings unavailable for test"),
    )

    demod = NRSC5Demodulator()

    assert not demod.available
    assert "bindings unavailable" in demod.unavailable_reason
    with pytest.raises(RuntimeError):
        demod.start(101_700_000)


def test_nrsc5_start_stop_python_backend(monkeypatch):
    class _FakeMode:
        FM = object()

    class _FakeNRSC5:
        libnrsc5 = object()

        def __init__(self, callback):
            self.callback = callback
            self.open_pipe_calls = 0
            self.set_mode_calls = 0
            self.start_calls = 0
            self.stop_calls = 0
            self.close_calls = 0

        def open_pipe(self):
            self.open_pipe_calls += 1

        def set_mode(self, _mode):
            self.set_mode_calls += 1

        def start(self):
            self.start_calls += 1

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

        def pipe_samples_cu8(self, _samples):
            return None

    class _FakeBindings:
        NRSC5 = _FakeNRSC5
        Mode = _FakeMode
        EventType = object()

    monkeypatch.setattr(
        NRSC5Demodulator,
        "_load_python_bindings",
        lambda self: (_FakeBindings, "/tmp/nrsc5.py", "/tmp/libnrsc5.so", ""),
    )

    demod = NRSC5Demodulator()
    demod.start(99_900_000)
    assert demod.is_running
    assert demod.backend == NRSC5Demodulator.BACKEND_PYTHON

    demod.stop()
    assert not demod.is_running


def test_nrsc5_python_backend_changes_program_without_restart(monkeypatch):
    class _FakeMode:
        FM = object()

    class _FakeNRSC5:
        libnrsc5 = object()
        instances = []

        def __init__(self, callback):
            self.callback = callback
            self.open_pipe_calls = 0
            self.set_mode_calls = 0
            self.start_calls = 0
            self.stop_calls = 0
            self.close_calls = 0
            _FakeNRSC5.instances.append(self)

        def open_pipe(self):
            self.open_pipe_calls += 1

        def set_mode(self, _mode):
            self.set_mode_calls += 1

        def start(self):
            self.start_calls += 1

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

        def pipe_samples_cu8(self, _samples):
            return None

    class _FakeBindings:
        NRSC5 = _FakeNRSC5
        Mode = _FakeMode
        EventType = object()

    monkeypatch.setattr(
        NRSC5Demodulator,
        "_load_python_bindings",
        lambda self: (_FakeBindings, "/tmp/nrsc5.py", "/tmp/libnrsc5.so", ""),
    )

    demod = NRSC5Demodulator(program=0)
    try:
        assert demod.backend == NRSC5Demodulator.BACKEND_PYTHON

        demod.start(99_900_000)
        assert demod.is_running
        first_decoder = demod._python_decoder

        demod.set_program(1)
        demod.start(99_900_000)
        assert demod.is_running
        second_decoder = demod._python_decoder

        assert first_decoder is second_decoder
        assert demod.active_program == 1
        assert len(_FakeNRSC5.instances) == 1
        assert first_decoder.start_calls == 1
    finally:
        demod.stop()


def test_nrsc5_python_audio_program_filter_locks_to_zero_based(monkeypatch):
    class _FakeMode:
        FM = object()

    class _FakeNRSC5:
        libnrsc5 = object()

        def __init__(self, callback):
            self.callback = callback

        def open_pipe(self):
            return None

        def set_mode(self, _mode):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

        def pipe_samples_cu8(self, _samples):
            return None

    class _FakeBindings:
        NRSC5 = _FakeNRSC5
        Mode = _FakeMode
        EventType = object()

    monkeypatch.setattr(
        NRSC5Demodulator,
        "_load_python_bindings",
        lambda self: (_FakeBindings, "/tmp/nrsc5.py", "/tmp/libnrsc5.so", ""),
    )

    demod = NRSC5Demodulator(program=0)
    raw = np.zeros((4096, 2), dtype=np.int16).tobytes()
    audio_evt_type = SimpleNamespace(name="AUDIO")
    try:
        demod.start(99_900_000)
        assert demod.backend == NRSC5Demodulator.BACKEND_PYTHON

        # Before seeing program 0, allow +1 fallback for one-based streams.
        with demod._lock:
            demod._python_seen_program_zero = False
            demod._audio_queue.clear()
        demod._handle_python_event(audio_evt_type, SimpleNamespace(program=1, data=raw))
        with demod._lock:
            assert len(demod._audio_queue) > 0
            demod._audio_queue.clear()

        # After program 0 is observed, reject +1 so HD1/HD2 don't interleave.
        with demod._lock:
            demod._python_seen_program_zero = True
        demod._handle_python_event(audio_evt_type, SimpleNamespace(program=1, data=raw))
        with demod._lock:
            assert len(demod._audio_queue) == 0

        demod._handle_python_event(audio_evt_type, SimpleNamespace(program=0, data=raw))
        with demod._lock:
            assert len(demod._audio_queue) > 0
    finally:
        demod.stop()


def test_nrsc5_iq_conversion_generates_cu8():
    demod = NRSC5Demodulator()
    t = np.arange(8192, dtype=np.float64)
    iq = (0.3 * np.exp(1j * (2 * np.pi * 0.01 * t))).astype(np.complex64)

    payload = demod._iq_to_cu8(iq, sample_rate_hz=480000)

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert (len(payload) % 4) == 0


def test_nrsc5_audio_pull_returns_exact_frames():
    demod = NRSC5Demodulator()
    t = np.arange(4410, dtype=np.float32) / 44100.0
    audio_44100 = np.column_stack((
        np.sin(2 * np.pi * 1000.0 * t),
        np.sin(2 * np.pi * 1200.0 * t),
    )).astype(np.float32)
    audio_48000 = demod._resample_audio(audio_44100)
    assert audio_48000.shape[0] > 100

    with demod._lock:
        demod._audio_queue.append(audio_48000)
        demod._last_audio_time_s = time.monotonic()

    pulled = demod.pull_audio(100)
    assert pulled is not None
    assert pulled.shape == (100, 2)


def test_nrsc5_metadata_parsing_selects_active_program():
    demod = NRSC5Demodulator(program=1)

    stream = io.StringIO(
        "Station name: WXYZ-HD\n"
        "Audio program 0: Main Channel, type: Rock, sound experience 0\n"
        "Audio program 1: Alt Channel, type: Alternative, sound experience 0\n"
        "Audio service 1: Alt Channel, type: Alternative, codec: 0, blend: 0, gain: 0 dB, delay: 0, latency: 0\n"
        "Title: Song A\n"
        "Artist: Artist B\n"
        "Album: Album C\n"
        "Genre: Classical\n"
        "Slogan: Listener-Supported\n"
        "Message: Public Radio for NC\n"
        "Alert: Category=[Weather] [12345] Storm Warning\n"
        "HERE Image: type=WEATHER, seq=1, n1=0, n2=0, time=2026-02-14T18:00:00Z, lat1=1.0, lon1=2.0, lat2=3.0, lon2=4.0, name=WeatherImage_0_0_rdhs.png, size=12345\n"
        "HERE Image: type=TRAFFIC, seq=2, n1=1, n2=9, time=2026-02-14T18:01:00Z, lat1=1.0, lon1=2.0, lat2=3.0, lon2=4.0, name=trafficMap_1_2_rdhs.png, size=22222\n"
    )
    demod._drain_output(stream)
    meta = demod.metadata_snapshot

    assert meta["station_name"] == "WXYZ-HD"
    assert meta["program_name"] == "Alt Channel"
    assert meta["program_number"] == 1
    assert meta["service_name"] == "Alt Channel"
    assert meta["service_number"] == 1
    assert meta["title"] == "Song A"
    assert meta["artist"] == "Artist B"
    assert meta["album"] == "Album C"
    assert meta["genre"] == "Classical"
    assert meta["station_slogan"] == "Listener-Supported"
    assert meta["station_message"] == "Public Radio for NC"
    assert meta["emergency_alert"] == "Category=[Weather] [12345] Storm Warning"
    assert meta["here_weather_time_utc"] == "2026-02-14T18:00:00Z"
    assert meta["here_weather_name"] == "WeatherImage_0_0_rdhs.png"
    assert meta["here_traffic_time_utc"] == "2026-02-14T18:01:00Z"
    assert meta["here_traffic_name"] == "trafficMap_1_2_rdhs.png"


def test_nrsc5_metadata_parses_prefixed_log_lines():
    demod = NRSC5Demodulator(program=0)
    demod._drain_output(io.StringIO(
        "[I] Station name: PREFIX-HD\n"
        "2026-02-14T13:00:00Z INFO Title: Prefixed Song\n"
        "2026-02-14T13:00:01Z INFO Artist: Prefixed Artist\n"
    ))
    meta = demod.metadata_snapshot
    assert meta["station_name"] == "PREFIX-HD"
    assert meta["title"] == "Prefixed Song"
    assert meta["artist"] == "Prefixed Artist"


def test_nrsc5_metadata_strips_terminal_title_escape_sequences():
    demod = NRSC5Demodulator(program=0)
    demod._drain_output(io.StringIO(
        "\x1b]0;nrsc5\x07Station name: CLEAN-HD\n"
        "\x1b]0;nrsc5\x07Title: Clean Song\n"
    ))
    meta = demod.metadata_snapshot
    assert meta["station_name"] == "CLEAN-HD"
    assert meta["title"] == "Clean Song"
    assert "\x1b" not in demod.last_output_line


def test_nrsc5_metadata_alert_ended_clears_alert():
    demod = NRSC5Demodulator(program=0)
    demod._drain_output(io.StringIO(
        "Alert: Category=[Weather] [12345] Storm Warning\n"
        "Alert ended\n"
    ))
    meta = demod.metadata_snapshot
    assert meta["emergency_alert"] == ""


def test_nrsc5_metadata_ignores_public_as_service_name_and_keeps_type():
    demod = NRSC5Demodulator(program=0)
    demod._drain_output(io.StringIO(
        "Audio program 0: public, type: Classical, sound experience 0\n"
        "Audio service 0: public, type: Classical, codec: 0, blend: 0, gain: 0 dB, delay: 0, latency: 0\n"
    ))
    meta = demod.metadata_snapshot
    assert meta["program_name"] == ""
    assert meta["service_name"] == ""
    assert meta["genre"] == "Classical"


def test_nrsc5_stop_clears_metadata():
    demod = NRSC5Demodulator()
    demod._drain_output(io.StringIO("Station name: TEST\nTitle: Example\n"))
    assert demod.metadata_snapshot["station_name"] == "TEST"

    demod.stop()
    meta = demod.metadata_snapshot
    assert meta["station_name"] == ""
    assert meta["title"] == ""


def test_nrsc5_stats_snapshot_tracks_python_events():
    demod = NRSC5Demodulator(program=0)

    demod._handle_python_event(
        SimpleNamespace(name="SYNC"),
        SimpleNamespace(freq_offset=1.25, psmi=7),
    )
    demod._handle_python_event(
        SimpleNamespace(name="MER"),
        SimpleNamespace(lower=12.5, upper=11.8),
    )
    demod._handle_python_event(
        SimpleNamespace(name="BER"),
        SimpleNamespace(cber=3.2e-4),
    )
    demod._handle_python_event(
        SimpleNamespace(name="AGC"),
        SimpleNamespace(gain_db=-2.1, peak_dbfs=-0.4, is_final=True),
    )
    demod._handle_python_event(
        SimpleNamespace(name="LOST_SYNC"),
        SimpleNamespace(),
    )

    stats = demod.stats_snapshot
    assert stats["sync"] is False
    assert stats["sync_count"] == 1
    assert stats["lost_sync_count"] == 1
    assert stats["last_sync_freq_offset_hz"] == pytest.approx(1.25)
    assert stats["last_sync_psmi"] == 7
    assert stats["mer_lower_db"] == pytest.approx(12.5)
    assert stats["mer_upper_db"] == pytest.approx(11.8)
    assert stats["ber_cber"] == pytest.approx(3.2e-4)
    assert stats["agc_gain_db"] == pytest.approx(-2.1)
    assert stats["agc_peak_dbfs"] == pytest.approx(-0.4)
    assert stats["agc_is_final"] is True
    assert stats["updated_at_s"] > 0.0


def test_nrsc5_stop_clears_stats():
    demod = NRSC5Demodulator(program=0)
    demod._handle_python_event(
        SimpleNamespace(name="SYNC"),
        SimpleNamespace(freq_offset=0.5, psmi=2),
    )
    assert demod.stats_snapshot["sync_count"] == 1

    demod.stop()
    stats = demod.stats_snapshot
    assert stats["sync"] is False
    assert stats["sync_count"] == 0
    assert stats["lost_sync_count"] == 0
    assert stats["last_sync_freq_offset_hz"] is None
    assert stats["last_sync_psmi"] is None
    assert stats["mer_lower_db"] is None
    assert stats["mer_upper_db"] is None
    assert stats["ber_cber"] is None
    assert stats["agc_gain_db"] is None
    assert stats["agc_peak_dbfs"] is None
    assert stats["agc_is_final"] is None

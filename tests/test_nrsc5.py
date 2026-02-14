#!/usr/bin/env python3
"""Unit tests for nrsc5 process hooks."""

import io
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from nrsc5 import NRSC5Demodulator


def test_nrsc5_unavailable_reports_reason(monkeypatch):
    monkeypatch.delenv("PJFM_NRSC5_COMMAND", raising=False)
    monkeypatch.setenv("PJFM_NRSC5_BIN", "nrsc5-definitely-not-installed")

    demod = NRSC5Demodulator()

    assert not demod.available
    assert "not found" in demod.unavailable_reason
    with pytest.raises(RuntimeError):
        demod.start(101_700_000)


def test_nrsc5_command_template_formats_frequency(monkeypatch):
    monkeypatch.setenv(
        "PJFM_NRSC5_COMMAND",
        "{nrsc5} --freq={freq_mhz} --hz={freq_hz} --program={program}",
    )
    monkeypatch.setenv("PJFM_NRSC5_BIN", "nrsc5")

    demod = NRSC5Demodulator(program=2)
    cmd = demod._build_command(101_700_000)

    assert cmd[1] == "--freq=101.700"
    assert cmd[2] == "--hz=101700000"
    assert cmd[3] == "--program=2"


def test_nrsc5_start_stop_with_template_process(monkeypatch):
    monkeypatch.setenv(
        "PJFM_NRSC5_COMMAND",
        f"{sys.executable} -c 'import time; time.sleep(10)'",
    )

    demod = NRSC5Demodulator()
    demod.start(99_900_000)
    assert demod.is_running

    demod.stop()
    assert not demod.is_running


def test_nrsc5_default_command_uses_stdin_iq(monkeypatch):
    monkeypatch.delenv("PJFM_NRSC5_COMMAND", raising=False)
    monkeypatch.delenv("PJFM_NRSC5_ARGS", raising=False)
    monkeypatch.delenv("PJFM_NRSC5_PROGRAM", raising=False)

    demod = NRSC5Demodulator(program=3, binary_name="python3")
    cmd = demod._build_command(101_700_000)

    assert "-q" not in cmd
    assert "-r" in cmd
    r_index = cmd.index("-r")
    assert cmd[r_index + 1] == "-"
    assert "-o" in cmd
    o_index = cmd.index("-o")
    assert cmd[o_index + 1] == "-"
    assert cmd[-1] == "3"
    # stdin mode command should not append frequency argument.
    assert "101.700" not in cmd


def test_nrsc5_set_program_updates_runtime_command(monkeypatch):
    monkeypatch.delenv("PJFM_NRSC5_COMMAND", raising=False)
    monkeypatch.delenv("PJFM_NRSC5_ARGS", raising=False)
    monkeypatch.delenv("PJFM_NRSC5_PROGRAM", raising=False)

    demod = NRSC5Demodulator(program=0, binary_name="python3")
    demod.set_program(2)
    cmd = demod._build_command(101_700_000)

    assert cmd[-1] == "2"


def test_nrsc5_start_restarts_when_program_changes(monkeypatch):
    monkeypatch.setenv(
        "PJFM_NRSC5_COMMAND",
        f"{sys.executable} -c 'import time; time.sleep(10)'",
    )

    demod = NRSC5Demodulator(program=0)
    try:
        demod.start(99_900_000)
        assert demod.is_running
        with demod._lock:
            first_proc = demod._process

        # Same program/frequency should keep the running process.
        demod.start(99_900_000)
        with demod._lock:
            second_proc = demod._process
        assert second_proc is first_proc

        demod.set_program(1)
        demod.start(99_900_000)
        assert demod.is_running
        with demod._lock:
            third_proc = demod._process

        assert third_proc is not first_proc
        assert demod.active_program == 1
    finally:
        demod.stop()


def test_nrsc5_iq_conversion_generates_cu8():
    demod = NRSC5Demodulator(binary_name="python3")
    t = np.arange(8192, dtype=np.float64)
    iq = (0.3 * np.exp(1j * (2 * np.pi * 0.01 * t))).astype(np.complex64)

    payload = demod._iq_to_cu8(iq, sample_rate_hz=480000)

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert (len(payload) % 4) == 0


def test_nrsc5_audio_pull_returns_exact_frames():
    demod = NRSC5Demodulator(binary_name="python3")
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
    demod = NRSC5Demodulator(binary_name="python3", program=1)

    stream = io.StringIO(
        "Station name: WXYZ-HD\n"
        "Audio program 0: Main Channel, type: Rock, sound experience 0\n"
        "Audio program 1: Alt Channel, type: Alternative, sound experience 0\n"
        "Audio service 1: Alt Channel, type: Alternative, codec: 0, blend: 0, gain: 0 dB, delay: 0, latency: 0\n"
        "Title: Song A\n"
        "Artist: Artist B\n"
        "Album: Album C\n"
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


def test_nrsc5_metadata_parses_prefixed_log_lines():
    demod = NRSC5Demodulator(binary_name="python3", program=0)
    demod._drain_output(io.StringIO(
        "[I] Station name: PREFIX-HD\n"
        "2026-02-14T13:00:00Z INFO Title: Prefixed Song\n"
        "2026-02-14T13:00:01Z INFO Artist: Prefixed Artist\n"
    ))
    meta = demod.metadata_snapshot
    assert meta["station_name"] == "PREFIX-HD"
    assert meta["title"] == "Prefixed Song"
    assert meta["artist"] == "Prefixed Artist"


def test_nrsc5_stop_clears_metadata():
    demod = NRSC5Demodulator(binary_name="python3")
    demod._drain_output(io.StringIO("Station name: TEST\nTitle: Example\n"))
    assert demod.metadata_snapshot["station_name"] == "TEST"

    demod.stop()
    meta = demod.metadata_snapshot
    assert meta["station_name"] == ""
    assert meta["title"] == ""

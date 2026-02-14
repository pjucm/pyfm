#!/usr/bin/env python3
"""Tests for adaptive rate-adjust clamping logic."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pjfm import FMRadio


class _FakeHDDecoder:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _FakeHDMetadataDecoder:
    def __init__(self, metadata):
        self.metadata_snapshot = dict(metadata)
        self.audio_active = True
        self.iq_bytes_in_total = 0
        self.audio_bytes_out_total = 0
        self.last_output_line = ""
        self.last_error = ""
        self.available = True
        self.program = 0


def test_rate_adjust_clamp_fm_mode():
    assert FMRadio._clamp_rate_adjust(1.02, weather_mode=False) == FMRadio.RATE_ADJ_MAX
    assert FMRadio._clamp_rate_adjust(0.98, weather_mode=False) == FMRadio.RATE_ADJ_MIN
    assert FMRadio._clamp_rate_adjust(1.0, weather_mode=False) == 1.0


def test_rate_adjust_clamp_weather_mode_tighter():
    assert FMRadio._clamp_rate_adjust(1.02, weather_mode=True) == FMRadio.NBFM_RATE_ADJ_MAX
    assert FMRadio._clamp_rate_adjust(0.98, weather_mode=True) == FMRadio.NBFM_RATE_ADJ_MIN
    assert FMRadio._clamp_rate_adjust(1.0004, weather_mode=True) == 1.0004


def test_normalize_pll_kernel_mode():
    assert FMRadio._normalize_pll_kernel_mode("python") == "python"
    assert FMRadio._normalize_pll_kernel_mode("AUTO") == "auto"
    assert FMRadio._normalize_pll_kernel_mode("numba") == "numba"
    assert FMRadio._normalize_pll_kernel_mode("invalid") == "python"


def test_iq_loss_flush_suppressed_during_startup_grace():
    now_s = 10.0
    stream_start_s = 9.0  # within default 2.0 s startup grace
    should_flush = FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=0.0,
        stream_start_s=stream_start_s,
    )
    assert not should_flush


def test_iq_loss_flush_requires_threshold_and_cooldown():
    now_s = 10.0
    stream_start_s = 0.0  # disable startup grace for this case

    assert not FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD - 1,
        now_s=now_s,
        last_flush_s=0.0,
        stream_start_s=stream_start_s,
    )
    assert not FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=now_s - (FMRadio.IQ_LOSS_FLUSH_COOLDOWN_S * 0.5),
        stream_start_s=stream_start_s,
    )
    assert FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=now_s - (FMRadio.IQ_LOSS_FLUSH_COOLDOWN_S + 0.1),
        stream_start_s=stream_start_s,
    )


def test_hd_program_helpers_cycle_hd1_hd2_hd3():
    assert FMRadio._normalize_hd_program("bad") == 0
    assert FMRadio._next_hd_program(0) == 1
    assert FMRadio._next_hd_program(1) == 2
    assert FMRadio._next_hd_program(2) == 0
    assert FMRadio._hd_program_label(0) == "HD1"
    assert FMRadio._hd_program_label(1) == "HD2"
    assert FMRadio._hd_program_label(2) == "HD3"


def test_snap_hd_decoder_off_stops_running_decoder():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDDecoder()
    radio.hd_enabled = True

    FMRadio._snap_hd_decoder_off(radio)

    assert radio.hd_enabled is False
    assert radio.hd_decoder.stop_calls == 1


def test_snap_hd_decoder_off_handles_missing_decoder():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = None
    radio.hd_enabled = True

    FMRadio._snap_hd_decoder_off(radio)

    assert radio.hd_enabled is False


def test_hd_metadata_summaries_include_station_and_track():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "WXYZ-HD",
        "program_name": "Alt Rock",
        "service_name": "",
        "sig_service_name": "",
        "title": "Song A",
        "artist": "Artist B",
        "album": "Album C",
    })
    radio.hd_enabled = True

    assert radio.hd_station_summary == "WXYZ-HD / Alt Rock"
    assert radio.hd_now_playing_summary == "Artist B - Song A (Album C)"


def test_hd_status_detail_falls_back_to_last_output_line():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({})
    radio.hd_decoder.last_output_line = "Audio service 1: Test"
    radio.hd_enabled = True
    radio.weather_mode = False

    detail = radio.hd_status_detail
    assert "Audio service 1: Test" in detail


def test_hd_status_detail_suppresses_station_repeat_when_metadata_present():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "WXYZ-HD",
        "program_name": "Alt Rock",
        "service_name": "",
        "sig_service_name": "",
        "title": "",
        "artist": "",
        "album": "",
    })
    radio.hd_enabled = True
    radio.weather_mode = False

    assert radio.hd_status_detail == ""

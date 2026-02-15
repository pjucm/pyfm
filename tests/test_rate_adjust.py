#!/usr/bin/env python3
"""Tests for adaptive rate-adjust clamping logic."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pjfm import FMRadio


class _FakeHDDecoder:
    def __init__(self):
        self.stop_calls = 0
        self.program = 2
        self.set_program_calls = []

    def stop(self):
        self.stop_calls += 1

    def set_program(self, program):
        self.program = int(program)
        self.set_program_calls.append(self.program)


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


class _FakeRDSDecoder:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _FakeRecorder:
    def __init__(self, output_dir="/tmp/recordings"):
        self.output_dir = output_dir
        self.is_recording = False
        self.start_calls = []

    def start(self, output_path=None):
        self.is_recording = True
        self.start_calls.append(output_path)
        return output_path

    def stop(self):
        self.is_recording = False
        return None


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
    assert radio.hd_decoder.program == 0
    assert radio.hd_decoder.set_program_calls == [0]


def test_snap_hd_decoder_off_handles_missing_decoder():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = None
    radio.hd_enabled = True

    FMRadio._snap_hd_decoder_off(radio)

    assert radio.hd_enabled is False


def test_snap_hd_decoder_off_resets_program_when_already_off():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDDecoder()
    radio.hd_enabled = False

    FMRadio._snap_hd_decoder_off(radio)

    assert radio.hd_enabled is False
    assert radio.hd_decoder.stop_calls == 0
    assert radio.hd_decoder.program == 0
    assert radio.hd_decoder.set_program_calls == [0]


def test_suspend_rds_for_hd_clears_decoder_state():
    radio = FMRadio.__new__(FMRadio)
    radio.weather_mode = False
    radio.hd_enabled = True
    radio.rds_enabled = True
    radio.rds_data = {"station_name": "WXYZ"}
    radio.rds_decoder = _FakeRDSDecoder()

    FMRadio._suspend_rds_for_hd(radio)

    assert radio.rds_enabled is False
    assert radio.rds_data == {}
    assert radio.rds_decoder.reset_calls == 1


def test_suspend_rds_for_hd_noop_when_not_active():
    radio = FMRadio.__new__(FMRadio)
    radio.weather_mode = False
    radio.hd_enabled = False
    radio.rds_enabled = True
    radio.rds_data = {"station_name": "WXYZ"}
    radio.rds_decoder = _FakeRDSDecoder()

    FMRadio._suspend_rds_for_hd(radio)

    assert radio.rds_enabled is True
    assert radio.rds_data == {"station_name": "WXYZ"}
    assert radio.rds_decoder.reset_calls == 0

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
        "genre": "Classical",
    })
    radio.hd_enabled = True

    assert radio.hd_station_summary == "WXYZ-HD (Classical)"
    assert radio.hd_now_playing_summary == "Artist B - Song A (Album C)"


def test_normalize_broadcast_text_decodes_html_entities():
    assert FMRadio._normalize_broadcast_text("Song &quot;A&quot;") == 'Song "A"'
    assert FMRadio._normalize_broadcast_text("AT&amp;T") == "AT&T"


def test_hd_now_playing_summary_decodes_html_quote_entities():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "WXYZ-HD",
        "program_name": "Alt Rock",
        "service_name": "",
        "sig_service_name": "",
        "title": "Song &quot;A&quot;",
        "artist": "Artist &quot;B&quot;",
        "album": "Album &quot;C&quot;",
        "genre": "Classical",
    })
    radio.hd_enabled = True

    assert radio.hd_now_playing_summary == 'Artist "B" - Song "A" (Album "C")'


def test_hd_metadata_extended_summaries():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "genre": "Classical",
        "station_slogan": "Listener-Supported",
        "station_message": "Public Radio for NC",
        "emergency_alert": "",
        "here_weather_time_utc": "2026-02-14T18:00:00Z",
        "here_weather_name": "WeatherImage_0_0_rdhs.png",
    })
    radio.hd_enabled = True

    assert radio.hd_genre_summary == "Classical"
    assert radio.hd_info_summary == "HD1 | Public Radio for NC"
    assert radio.hd_weather_summary == "2026-02-14T18:00:00Z  WeatherImage_0_0_rdhs.png"


def test_hd_info_summary_prefers_active_alert():
    radio = FMRadio.__new__(FMRadio)
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_slogan": "Listener-Supported",
        "station_message": "Public Radio for NC",
        "emergency_alert": "Category=[Weather] [12345] Storm Warning",
    })
    radio.hd_enabled = True

    assert radio.hd_info_summary == "HD1 | Alert: Category=[Weather] [12345] Storm Warning"


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


def test_build_recording_output_path_hd_mode_uses_station_track_timestamp(monkeypatch):
    radio = FMRadio.__new__(FMRadio)
    radio.recorder = _FakeRecorder(output_dir="/tmp/recordings")
    radio.weather_mode = False
    radio.hd_enabled = True
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "WDAV",
        "title": "Overture in F",
        "artist": "Genevieve Soly",
    })

    monkeypatch.setattr("pjfm.time.strftime", lambda _fmt: "260215101530")
    path = FMRadio._build_recording_output_path(radio)

    assert path == "/tmp/recordings/WDAV - Genevieve Soly - Overture in F - 260215101530.opus"


def test_build_recording_output_path_sanitizes_filename_components(monkeypatch):
    radio = FMRadio.__new__(FMRadio)
    radio.recorder = _FakeRecorder(output_dir="/tmp/recordings")
    radio.weather_mode = False
    radio.hd_enabled = True
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "W/DA:V*?",
        "title": "Song <One>",
        "artist": "A|B",
    })

    monkeypatch.setattr("pjfm.time.strftime", lambda _fmt: "260215101530")
    path = FMRadio._build_recording_output_path(radio)

    assert path == "/tmp/recordings/W DA V - A B - Song One - 260215101530.opus"


def test_toggle_recording_passes_hd_output_path(monkeypatch):
    radio = FMRadio.__new__(FMRadio)
    radio.recorder = _FakeRecorder(output_dir="/tmp/recordings")
    radio.weather_mode = False
    radio.hd_enabled = True
    radio.hd_decoder = _FakeHDMetadataDecoder({
        "station_name": "WDAV",
        "title": "Overture in F",
        "artist": "Genevieve Soly",
    })
    radio.error_message = None

    monkeypatch.setattr("pjfm.time.strftime", lambda _fmt: "260215101530")
    out = FMRadio.toggle_recording(radio)

    assert out == "/tmp/recordings/WDAV - Genevieve Soly - Overture in F - 260215101530.opus"
    assert radio.recorder.start_calls == [out]


def test_toggle_recording_non_hd_uses_recorder_default_path():
    radio = FMRadio.__new__(FMRadio)
    radio.recorder = _FakeRecorder(output_dir="/tmp/recordings")
    radio.weather_mode = False
    radio.hd_enabled = False
    radio.hd_decoder = _FakeHDMetadataDecoder({})
    radio.error_message = None

    out = FMRadio.toggle_recording(radio)

    assert out is None
    assert radio.recorder.start_calls == [None]

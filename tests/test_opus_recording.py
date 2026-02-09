#!/usr/bin/env python3
"""
Unit tests for Opus recording support.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from opus import OpusRecorder, build_recording_status_text


class _FakeStdin:
    def __init__(self):
        self.buffer = io.BytesIO()
        self.closed = False

    def write(self, data):
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.buffer.write(data)
        return len(data)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.returncode = None
        self.terminated = False
        self.wait_calls = 0
        self.pid = 12345

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_opus_recorder_start_write_stop(monkeypatch, tmp_path):
    popen_calls = {}
    fake_process = _FakeProcess()

    def _fake_popen(cmd, stdin, stdout, stderr):
        popen_calls['cmd'] = cmd
        popen_calls['stdin'] = stdin
        popen_calls['stdout'] = stdout
        popen_calls['stderr'] = stderr
        return fake_process

    monkeypatch.setattr("opus.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("opus.subprocess.Popen", _fake_popen)

    recorder = OpusRecorder(
        sample_rate=48000,
        channels=2,
        bitrate_kbps=128,
        output_dir=tmp_path,
    )

    output_path = recorder.start()
    assert recorder.is_recording is True
    assert output_path.endswith(".opus")
    assert Path(output_path).parent == tmp_path

    # Verify ffmpeg invocation requests stereo 48kHz 128kbps Opus.
    cmd = popen_calls['cmd']
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "48000"
    assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "128k"
    assert "libopus" in cmd

    # Write a small stereo frame and ensure bytes are forwarded to ffmpeg stdin.
    frame = np.ones((256, 2), dtype=np.float32) * 0.25
    recorder.write(frame)
    assert fake_process.stdin.buffer.getbuffer().nbytes > 0

    stopped_path = recorder.stop()
    assert stopped_path == output_path
    assert recorder.is_recording is False
    assert fake_process.stdin.closed is True
    assert fake_process.wait_calls >= 1


def test_build_recording_status_text_active_is_red():
    text = build_recording_status_text(
        is_recording=True,
        elapsed_seconds=12.3,
        output_path="/tmp/example.opus",
    )
    assert "REC" in text.plain
    assert any("red" in str(span.style) for span in text.spans)


def test_opus_write_ignores_closed_stdin_during_stop_race(monkeypatch, tmp_path):
    fake_process = _FakeProcess()

    def _fake_popen(cmd, stdin, stdout, stderr):
        return fake_process

    monkeypatch.setattr("opus.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("opus.subprocess.Popen", _fake_popen)

    recorder = OpusRecorder(
        sample_rate=48000,
        channels=2,
        bitrate_kbps=128,
        output_dir=tmp_path,
    )
    recorder.start()

    # Simulate race: stdin gets closed by stop() before recorder state is cleared.
    fake_process.stdin.close()

    frame = np.zeros((64, 2), dtype=np.float32)
    recorder.write(frame)  # Must not raise.

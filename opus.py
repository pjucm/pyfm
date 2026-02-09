#!/usr/bin/env python3
"""
Opus recording support for pjfm.

Uses ffmpeg + libopus to encode float32 PCM frames into .opus files.
"""

import os
import shutil
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
from rich.text import Text


class OpusRecorder:
    """Record stereo audio frames to an Opus file via ffmpeg."""

    def __init__(self, sample_rate=48000, channels=2, bitrate_kbps=128, output_dir=None):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.bitrate_kbps = int(bitrate_kbps)
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recordings",
        )

        self._process = None
        self._output_path = None
        self._started_at = None
        self._lock = threading.Lock()

    @property
    def is_recording(self):
        with self._lock:
            return self._process is not None

    @property
    def output_path(self):
        with self._lock:
            return self._output_path

    @property
    def elapsed_seconds(self):
        with self._lock:
            started_at = self._started_at
        if started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - started_at)

    def _default_output_path(self):
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(self.output_dir, f"pjfm-{stamp}.opus")

    def start(self, output_path=None):
        """Start recording to a new Opus file."""
        with self._lock:
            if self._process is not None:
                return self._output_path

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found; install ffmpeg to use recording")

        out_path = output_path or self._default_output_path()
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-f", "f32le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", f"{self.bitrate_kbps}k",
            "-vbr", "on",
            "-application", "audio",
            "-compression_level", "10",
            "-y",
            out_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._lock:
            self._process = proc
            self._output_path = out_path
            self._started_at = time.monotonic()
        return out_path

    def write(self, audio_frame):
        """Write one decoded audio frame to the active recording."""
        frame = np.asarray(audio_frame, dtype=np.float32)
        if frame.ndim == 1:
            frame = np.column_stack((frame, frame))
        elif frame.ndim == 2 and frame.shape[1] == 1:
            frame = np.repeat(frame, 2, axis=1)
        elif frame.ndim != 2 or frame.shape[1] != 2:
            raise ValueError(f"unsupported audio frame shape for recording: {frame.shape}")

        with self._lock:
            proc = self._process
        if proc is None:
            return

        stdin = proc.stdin
        if stdin is None or getattr(stdin, "closed", False):
            return

        payload = frame.tobytes()
        try:
            stdin.write(payload)
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            # If stop() already detached this process, suppress benign race errors.
            with self._lock:
                still_active = self._process is proc
            if not still_active:
                return
            self.stop()
            raise RuntimeError(f"recording pipeline failed: {exc}") from exc

    def stop(self):
        """Stop recording and finalize the current Opus file."""
        with self._lock:
            proc = self._process
            out_path = self._output_path
            if proc is None:
                return out_path
            # Detach recorder state first so audio thread stops writing immediately.
            self._process = None
            self._output_path = None
            self._started_at = None

        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (OSError, ValueError):
            pass

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

        return out_path


def build_recording_status_text(is_recording, elapsed_seconds=0.0, output_path=None):
    """Build rich text for recording status."""
    if not is_recording:
        return Text("OFF", style="dim")

    text = Text()
    text.append("REC", style="red bold")
    text.append(f" {elapsed_seconds:5.1f}s", style="red")
    if output_path:
        text.append("  ", style="")
        text.append(os.path.basename(output_path), style="red")
    return text

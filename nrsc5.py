#!/usr/bin/env python3
"""
Optional HD Radio demodulation hooks for pjfm via the external `nrsc5` CLI.
"""

from collections import deque
import os
import re
import shlex
import shutil
import subprocess
import threading
import time

import numpy as np


class _StreamingLinearResampler:
    """Phase-continuous streaming linear resampler."""

    def __init__(self):
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def process(self, ratio, samples):
        x = np.asarray(samples, dtype=np.float64)
        n_in = len(x)
        if n_in <= 0 or ratio <= 0.0:
            return np.array([], dtype=np.float64)

        step = 1.0 / float(ratio)
        phase = float(self.phase)
        if phase < 0.0 or phase >= n_in:
            phase = phase % n_in

        n_out = int(np.floor((n_in - 1e-12 - phase) / step)) + 1
        if n_out <= 0:
            self.phase = phase - n_in
            return np.array([], dtype=np.float64)

        positions = phase + step * np.arange(n_out, dtype=np.float64)
        self.phase = positions[-1] + step - n_in
        idx = np.arange(n_in, dtype=np.float64)
        return np.interp(positions, idx, x)


class NRSC5Demodulator:
    """
    Manage an optional nrsc5 subprocess for HD Radio decoding.

    By default this launches stdin IQ mode:
        nrsc5 -r - -o - -t raw <program>

    and accepts IQ pushes from pjfm, converting to the CU8 stream format
    expected by nrsc5's file input mode.

    Environment overrides:
      - PJFM_NRSC5_BIN: binary name/path (default: nrsc5)
      - PJFM_NRSC5_PROGRAM: HD subprogram number (default: 0)
      - PJFM_NRSC5_ARGS: extra args before frequency/program
      - PJFM_NRSC5_COMMAND: full command template (highest priority)

    `PJFM_NRSC5_COMMAND` tokens support format keys:
      - {freq_mhz}
      - {freq_hz}
      - {program}
      - {nrsc5}   (resolved binary path)
    """

    TARGET_IQ_RATE_HZ = 1_488_375.0
    TARGET_AUDIO_IN_RATE_HZ = 44_100.0
    TARGET_AUDIO_OUT_RATE_HZ = 48_000.0
    DEFAULT_ARGS = ("-r", "-", "-o", "-", "-t", "raw")
    IQ_QUEUE_MAX_BLOCKS_DEFAULT = 8
    AUDIO_QUEUE_MAX_BLOCKS_DEFAULT = 64
    IQ_TARGET_RMS_DEFAULT = 0.18
    IQ_GAIN_MIN = 0.2
    IQ_GAIN_MAX = 120.0
    _RE_STATION_NAME = re.compile(r"Station name:\s*(?P<name>.+)$")
    _RE_AUDIO_PROGRAM = re.compile(r"Audio program\s+(?P<num>\d+):\s*(?P<name>[^,]+)")
    _RE_AUDIO_SERVICE = re.compile(r"Audio service\s+(?P<num>\d+):\s*(?P<name>[^,]+)")
    _RE_SIG_SERVICE = re.compile(
        r"SIG Service:\s*type=(?P<type>\S+)\s+number=(?P<num>\d+)\s+name=(?P<name>.+)$"
    )
    _RE_TITLE = re.compile(r"Title:\s*(?P<value>.+)$")
    _RE_ARTIST = re.compile(r"Artist:\s*(?P<value>.+)$")
    _RE_ALBUM = re.compile(r"Album:\s*(?P<value>.+)$")
    _RE_GENRE = re.compile(r"Genre:\s*(?P<value>.+)$")
    _RE_STATION_SLOGAN = re.compile(r"Slogan:\s*(?P<value>.+)$")
    _RE_STATION_MESSAGE = re.compile(r"Message:\s*(?P<value>.+)$")
    _RE_ALERT = re.compile(r"Alert:\s*(?P<value>.+)$")
    _RE_ALERT_ENDED = re.compile(r"Alert ended$")
    _RE_HERE_IMAGE = re.compile(
        r"HERE Image:\s*type=(?P<type>[A-Z]+),.*?time=(?P<time>[^,]+),.*?name=(?P<name>.+?),\s*size=",
        re.IGNORECASE,
    )

    def __init__(self, program=0, binary_name=None, extra_args=None):
        self.binary_name = (
            binary_name
            or os.environ.get("PJFM_NRSC5_BIN", "nrsc5").strip()
            or "nrsc5"
        )
        self._binary_path = shutil.which(self.binary_name)

        program_override = os.environ.get("PJFM_NRSC5_PROGRAM", "").strip()
        if program_override:
            try:
                self.program = int(program_override)
            except ValueError:
                self.program = int(program)
        else:
            self.program = int(program)

        if extra_args is not None:
            self.extra_args = [str(arg) for arg in extra_args]
        else:
            arg_string = os.environ.get("PJFM_NRSC5_ARGS", "").strip()
            if arg_string:
                self.extra_args = shlex.split(arg_string)
            else:
                self.extra_args = list(self.DEFAULT_ARGS)

        cmd_template = os.environ.get("PJFM_NRSC5_COMMAND", "").strip()
        self.command_template = cmd_template or None

        self._lock = threading.Lock()
        self._iq_cond = threading.Condition(self._lock)
        self._process = None
        self._frequency_hz = None
        self._active_program = None
        self._stdin_iq_mode = False
        self._audio_stdout_mode = False
        self._uses_frequency_arg = True

        self._log_stderr_thread = None
        self._log_stdout_thread = None
        self._writer_thread = None
        self._audio_thread = None
        self._writer_stop = False

        self._iq_queue = deque()
        self._audio_queue = deque()
        self._audio_buffer = np.zeros((0, 2), dtype=np.float32)

        try:
            iq_blocks = int(os.environ.get("PJFM_NRSC5_QUEUE_BLOCKS", ""))
        except ValueError:
            iq_blocks = self.IQ_QUEUE_MAX_BLOCKS_DEFAULT
        self._iq_queue_max_blocks = max(2, iq_blocks or self.IQ_QUEUE_MAX_BLOCKS_DEFAULT)

        try:
            audio_blocks = int(os.environ.get("PJFM_NRSC5_AUDIO_QUEUE_BLOCKS", ""))
        except ValueError:
            audio_blocks = self.AUDIO_QUEUE_MAX_BLOCKS_DEFAULT
        self._audio_queue_max_blocks = max(4, audio_blocks or self.AUDIO_QUEUE_MAX_BLOCKS_DEFAULT)

        self._source_sample_rate_hz = None
        self._resampler_i = _StreamingLinearResampler()
        self._resampler_q = _StreamingLinearResampler()
        self._audio_resampler_l = _StreamingLinearResampler()
        self._audio_resampler_r = _StreamingLinearResampler()
        self._last_audio_time_s = 0.0
        self._iq_rms_ema = 0.1
        self._iq_gain = 1.0
        try:
            self._iq_target_rms = float(
                os.environ.get("PJFM_NRSC5_TARGET_RMS", str(self.IQ_TARGET_RMS_DEFAULT))
            )
        except ValueError:
            self._iq_target_rms = self.IQ_TARGET_RMS_DEFAULT
        self._iq_target_rms = max(0.01, min(0.8, self._iq_target_rms))

        self._iq_bytes_in_total = 0
        self._audio_bytes_out_total = 0

        self.last_error = ""
        self.last_output_line = ""
        self._metadata = {}
        self._reset_metadata_locked()

    @property
    def available(self):
        """True if nrsc5 launch prerequisites are satisfied."""
        return bool(self.command_template or self._binary_path)

    @property
    def unavailable_reason(self):
        """Human-readable reason when hooks are unavailable."""
        if self.available:
            return ""
        return f"{self.binary_name} not found in PATH"

    @property
    def is_running(self):
        """True if the nrsc5 subprocess is currently alive."""
        return self.poll()

    @property
    def frequency_hz(self):
        """Current tuned frequency for nrsc5, or None when stopped."""
        with self._lock:
            return self._frequency_hz

    @property
    def active_program(self):
        """Program index currently running in nrsc5, or None when stopped."""
        with self._lock:
            return self._active_program

    @property
    def stdin_iq_mode(self):
        """True when nrsc5 was launched in '-r -' IQ stdin mode."""
        with self._lock:
            return self._stdin_iq_mode

    @property
    def audio_active(self):
        """True when HD audio frames were received recently."""
        with self._lock:
            has_buffer = self._audio_buffer.shape[0] > 0 or bool(self._audio_queue)
            last_audio = self._last_audio_time_s
        if has_buffer:
            return True
        if last_audio <= 0.0:
            return False
        return (time.monotonic() - last_audio) < 1.0

    @property
    def iq_bytes_in_total(self):
        with self._lock:
            return int(self._iq_bytes_in_total)

    @property
    def audio_bytes_out_total(self):
        with self._lock:
            return int(self._audio_bytes_out_total)

    @property
    def metadata_snapshot(self):
        """Latest parsed metadata from nrsc5 output lines."""
        with self._lock:
            return dict(self._metadata)

    def _reset_metadata_locked(self):
        """Reset cached metadata state. Caller must hold `_lock`."""
        self._metadata = {
            "station_name": "",
            "program_name": "",
            "program_number": None,
            "service_name": "",
            "service_number": None,
            "sig_service_name": "",
            "title": "",
            "artist": "",
            "album": "",
            "genre": "",
            "station_slogan": "",
            "station_message": "",
            "emergency_alert": "",
            "here_weather_time_utc": "",
            "here_weather_name": "",
            "here_traffic_time_utc": "",
            "here_traffic_name": "",
            "updated_at_s": 0.0,
        }

    def _active_program_index_locked(self):
        prog = self._active_program
        if prog is None:
            prog = self.program
        try:
            return int(prog)
        except (TypeError, ValueError):
            return None

    def _program_match_score_locked(self, number):
        if number is None:
            return -1
        try:
            num = int(number)
        except (TypeError, ValueError):
            return -1
        active = self._active_program_index_locked()
        if active is None:
            return 0
        if num == active:
            return 3
        # Some nrsc5 builds log program numbers as 1-based service IDs.
        if num == (active + 1):
            return 2
        return 0

    def _update_text_metadata_locked(self, key, value):
        text = str(value).strip()
        if not text or self._metadata.get(key) == text:
            return False
        self._metadata[key] = text
        self._metadata["updated_at_s"] = time.monotonic()
        return True

    def _clear_text_metadata_locked(self, key):
        if not self._metadata.get(key):
            return False
        self._metadata[key] = ""
        self._metadata["updated_at_s"] = time.monotonic()
        return True

    def _update_program_metadata_locked(self, name_key, number_key, number, name):
        text = str(name).strip()
        if not text:
            return False
        try:
            num = int(number)
        except (TypeError, ValueError):
            num = None

        current_name = self._metadata.get(name_key, "")
        current_num = self._metadata.get(number_key)
        current_score = self._program_match_score_locked(current_num)
        new_score = self._program_match_score_locked(num)

        replace = False
        if not current_name:
            replace = True
        elif num is not None and current_num == num:
            replace = True
        elif new_score > current_score:
            replace = True

        if not replace:
            return False
        changed = (current_name != text) or (current_num != num)
        if not changed:
            return False
        self._metadata[name_key] = text
        self._metadata[number_key] = num
        self._metadata["updated_at_s"] = time.monotonic()
        return True

    def _ingest_output_line_locked(self, line):
        station_match = self._RE_STATION_NAME.search(line)
        if station_match:
            self._update_text_metadata_locked("station_name", station_match.group("name"))
            return

        title_match = self._RE_TITLE.search(line)
        if title_match:
            self._update_text_metadata_locked("title", title_match.group("value"))
            return

        artist_match = self._RE_ARTIST.search(line)
        if artist_match:
            self._update_text_metadata_locked("artist", artist_match.group("value"))
            return

        album_match = self._RE_ALBUM.search(line)
        if album_match:
            self._update_text_metadata_locked("album", album_match.group("value"))
            return

        genre_match = self._RE_GENRE.search(line)
        if genre_match:
            self._update_text_metadata_locked("genre", genre_match.group("value"))
            return

        slogan_match = self._RE_STATION_SLOGAN.search(line)
        if slogan_match:
            self._update_text_metadata_locked("station_slogan", slogan_match.group("value"))
            return

        message_match = self._RE_STATION_MESSAGE.search(line)
        if message_match:
            self._update_text_metadata_locked("station_message", message_match.group("value"))
            return

        alert_match = self._RE_ALERT.search(line)
        if alert_match:
            self._update_text_metadata_locked("emergency_alert", alert_match.group("value"))
            return

        if self._RE_ALERT_ENDED.search(line):
            self._clear_text_metadata_locked("emergency_alert")
            return

        here_match = self._RE_HERE_IMAGE.search(line)
        if here_match:
            image_type = here_match.group("type").strip().upper()
            image_time = here_match.group("time").strip()
            image_name = here_match.group("name").strip()
            if image_type == "WEATHER":
                self._update_text_metadata_locked("here_weather_time_utc", image_time)
                self._update_text_metadata_locked("here_weather_name", image_name)
            elif image_type == "TRAFFIC":
                self._update_text_metadata_locked("here_traffic_time_utc", image_time)
                self._update_text_metadata_locked("here_traffic_name", image_name)
            return

        prog_match = self._RE_AUDIO_PROGRAM.search(line)
        if prog_match:
            self._update_program_metadata_locked(
                "program_name",
                "program_number",
                prog_match.group("num"),
                prog_match.group("name"),
            )
            return

        svc_match = self._RE_AUDIO_SERVICE.search(line)
        if svc_match:
            self._update_program_metadata_locked(
                "service_name",
                "service_number",
                svc_match.group("num"),
                svc_match.group("name"),
            )
            return

        sig_match = self._RE_SIG_SERVICE.search(line)
        if not sig_match:
            return
        self._update_text_metadata_locked("sig_service_name", sig_match.group("name"))
        if sig_match.group("type").strip().lower().startswith("audio"):
            self._update_program_metadata_locked(
                "service_name",
                "service_number",
                sig_match.group("num"),
                sig_match.group("name"),
            )

    @staticmethod
    def _args_use_stdin_iq(args):
        for i, token in enumerate(args):
            if token == "-r" and i + 1 < len(args) and args[i + 1] == "-":
                return True
        return False

    @staticmethod
    def _args_use_audio_stdout(args):
        for i, token in enumerate(args):
            if token == "-o" and i + 1 < len(args) and args[i + 1] == "-":
                return True
        return False

    def set_program(self, program):
        """Set the HD subprogram index for subsequent starts/restarts."""
        new_program = int(program)
        if new_program < 0:
            raise ValueError("program must be >= 0")
        with self._lock:
            self.program = new_program

    def _build_runtime_command(self, frequency_hz, program=None):
        freq_hz = int(round(float(frequency_hz)))
        freq_mhz = float(freq_hz) / 1e6
        program = int(self.program if program is None else program)

        if self.command_template:
            template_parts = shlex.split(self.command_template)
            cmd = [
                token.format(
                    freq_hz=freq_hz,
                    freq_mhz=f"{freq_mhz:.3f}",
                    program=program,
                    nrsc5=self._binary_path or self.binary_name,
                )
                for token in template_parts
            ]
            stdin_iq_mode = self._args_use_stdin_iq(cmd)
            uses_frequency = (
                ("{freq_hz}" in self.command_template) or
                ("{freq_mhz}" in self.command_template)
            )
            audio_stdout_mode = self._args_use_audio_stdout(cmd)
            return cmd, stdin_iq_mode, uses_frequency, audio_stdout_mode

        if not self._binary_path:
            raise RuntimeError(self.unavailable_reason)

        args = [
            token.format(
                freq_hz=freq_hz,
                freq_mhz=f"{freq_mhz:.3f}",
                program=program,
                nrsc5=self._binary_path,
            )
            for token in self.extra_args
        ]
        stdin_iq_mode = self._args_use_stdin_iq(args)
        audio_stdout_mode = self._args_use_audio_stdout(args)
        if stdin_iq_mode:
            cmd = [self._binary_path] + args + [str(program)]
            return cmd, True, False, audio_stdout_mode

        cmd = [self._binary_path] + args + [f"{freq_mhz:.3f}", str(program)]
        return cmd, False, True, audio_stdout_mode

    def _build_command(self, frequency_hz):
        """
        Return the command list (tests and diagnostics helper).

        Runtime mode metadata is resolved by `_build_runtime_command()`.
        """
        cmd, _, _, _ = self._build_runtime_command(frequency_hz)
        return cmd

    def _drain_output(self, stream):
        if stream is None:
            return
        try:
            for raw_line in stream:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                else:
                    line = raw_line.strip()
                if not line:
                    continue
                with self._lock:
                    self.last_output_line = line[-240:]
                    self._ingest_output_line_locked(line)
        except Exception:
            pass

    def _reset_iq_resampler_state(self):
        self._source_sample_rate_hz = None
        self._resampler_i.reset()
        self._resampler_q.reset()
        self._iq_rms_ema = 0.1
        self._iq_gain = 1.0

    def _reset_audio_resampler_state(self):
        self._audio_resampler_l.reset()
        self._audio_resampler_r.reset()

    def _iq_to_cu8(self, iq_block, sample_rate_hz):
        fs = float(sample_rate_hz)
        if fs <= 0.0:
            return b""

        samples = np.asarray(iq_block, dtype=np.complex64)
        if samples.size < 2:
            return b""

        # Remove block DC offset and apply gentle RMS AGC so CU8 quantization
        # uses useful dynamic range for nrsc5's demod.
        mean_i = float(np.mean(samples.real))
        mean_q = float(np.mean(samples.imag))
        centered_i = samples.real - mean_i
        centered_q = samples.imag - mean_q
        rms = float(np.sqrt(np.mean(centered_i * centered_i + centered_q * centered_q) + 1e-12))
        self._iq_rms_ema = (0.9 * self._iq_rms_ema) + (0.1 * rms)
        if self._iq_rms_ema > 1e-7:
            self._iq_gain = self._iq_target_rms / self._iq_rms_ema
        else:
            self._iq_gain = 1.0
        self._iq_gain = max(self.IQ_GAIN_MIN, min(self.IQ_GAIN_MAX, self._iq_gain))
        scaled_i = centered_i * self._iq_gain
        scaled_q = centered_q * self._iq_gain

        if (self._source_sample_rate_hz is None or
                abs(self._source_sample_rate_hz - fs) > 1e-3):
            self._source_sample_rate_hz = fs
            self._resampler_i.reset()
            self._resampler_q.reset()

        ratio = self.TARGET_IQ_RATE_HZ / fs
        i_res = self._resampler_i.process(ratio, scaled_i)
        q_res = self._resampler_q.process(ratio, scaled_q)
        n_out = min(len(i_res), len(q_res))
        if n_out < 2:
            return b""

        if n_out & 1:
            n_out -= 1
        if n_out <= 0:
            return b""

        i_u8 = np.clip(np.rint((i_res[:n_out] * 127.5) + 127.5), 0, 255).astype(np.uint8)
        q_u8 = np.clip(np.rint((q_res[:n_out] * 127.5) + 127.5), 0, 255).astype(np.uint8)

        cu8 = np.empty(n_out * 2, dtype=np.uint8)
        cu8[0::2] = i_u8
        cu8[1::2] = q_u8

        remainder = cu8.size & 0x3
        if remainder:
            cu8 = cu8[:-remainder]
        if cu8.size == 0:
            return b""
        return cu8.tobytes()

    def _resample_audio(self, audio_44100):
        frames = np.asarray(audio_44100, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 2 or frames.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)

        ratio = self.TARGET_AUDIO_OUT_RATE_HZ / self.TARGET_AUDIO_IN_RATE_HZ
        left = self._audio_resampler_l.process(ratio, frames[:, 0])
        right = self._audio_resampler_r.process(ratio, frames[:, 1])
        n_out = min(len(left), len(right))
        if n_out <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        out = np.empty((n_out, 2), dtype=np.float32)
        out[:, 0] = left[:n_out].astype(np.float32)
        out[:, 1] = right[:n_out].astype(np.float32)
        return out

    def _audio_reader_loop(self, proc):
        stdout = proc.stdout
        if stdout is None:
            return

        remainder = b""
        while True:
            with self._lock:
                if self._process is not proc or self._writer_stop:
                    return
            try:
                chunk = stdout.read(32768)
            except Exception:
                return
            if not chunk:
                return

            blob = remainder + chunk
            frame_bytes = (len(blob) // 4) * 4
            remainder = blob[frame_bytes:]
            if frame_bytes <= 0:
                continue

            pcm16 = np.frombuffer(blob[:frame_bytes], dtype="<i2")
            if pcm16.size < 2:
                continue
            audio = pcm16.reshape(-1, 2).astype(np.float32) / 32768.0
            out = self._resample_audio(audio)
            if out.shape[0] == 0:
                continue

            with self._lock:
                if self._process is not proc or self._writer_stop:
                    return
                self._audio_bytes_out_total += int(frame_bytes)
                if len(self._audio_queue) >= self._audio_queue_max_blocks:
                    self._audio_queue.popleft()
                self._audio_queue.append(out)
                self._last_audio_time_s = time.monotonic()

    def _writer_loop(self, proc):
        while True:
            with self._iq_cond:
                while (not self._writer_stop and
                       self._process is proc and
                       proc.poll() is None and
                       not self._iq_queue):
                    self._iq_cond.wait(timeout=0.2)

                if (self._writer_stop or self._process is not proc or
                        proc.poll() is not None):
                    return

                iq_block, sample_rate_hz = self._iq_queue.popleft()

            payload = self._iq_to_cu8(iq_block, sample_rate_hz)
            if not payload:
                continue

            stdin = proc.stdin
            if stdin is None or getattr(stdin, "closed", False):
                return

            try:
                stdin.write(payload)
                stdin.flush()
                with self._lock:
                    if self._process is proc:
                        self._iq_bytes_in_total += int(len(payload))
            except (BrokenPipeError, OSError, ValueError) as exc:
                with self._lock:
                    if self._process is proc and not self.last_error:
                        self.last_error = f"nrsc5 IQ pipe write failed: {exc}"
                return

    def start(self, frequency_hz):
        """Start nrsc5 (or retune/restart if already running)."""
        freq_hz = int(round(float(frequency_hz)))
        with self._lock:
            proc = self._process
            current_freq = self._frequency_hz
            active_program = self._active_program
            desired_program = int(self.program)
            uses_frequency = self._uses_frequency_arg

        if (proc is not None and proc.poll() is None and
                active_program == desired_program and
                ((not uses_frequency) or current_freq == freq_hz)):
            return

        self.stop()

        if not self.available:
            raise RuntimeError(f"HD Radio unavailable: {self.unavailable_reason}")

        try:
            cmd, stdin_iq_mode, uses_frequency, audio_stdout_mode = self._build_runtime_command(
                freq_hz,
                program=desired_program,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build nrsc5 command: {exc}") from exc

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if stdin_iq_mode else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            self.last_error = f"Failed to start nrsc5: {exc}"
            raise RuntimeError(self.last_error) from exc

        with self._lock:
            self._process = proc
            self._frequency_hz = freq_hz
            self._active_program = desired_program
            self._stdin_iq_mode = stdin_iq_mode
            self._audio_stdout_mode = audio_stdout_mode
            self._uses_frequency_arg = uses_frequency
            self._writer_stop = False
            self._iq_queue.clear()
            self._audio_queue.clear()
            self._audio_buffer = np.zeros((0, 2), dtype=np.float32)
            self._reset_iq_resampler_state()
            self._reset_audio_resampler_state()
            self._last_audio_time_s = 0.0
            self._iq_bytes_in_total = 0
            self._audio_bytes_out_total = 0
            self.last_error = ""
            self.last_output_line = ""
            self._reset_metadata_locked()

            self._log_stderr_thread = threading.Thread(
                target=self._drain_output,
                args=(proc.stderr,),
                daemon=True,
            )
            self._log_stderr_thread.start()

            if audio_stdout_mode:
                self._audio_thread = threading.Thread(
                    target=self._audio_reader_loop,
                    args=(proc,),
                    daemon=True,
                )
                self._audio_thread.start()
                self._log_stdout_thread = None
            else:
                self._audio_thread = None
                self._log_stdout_thread = threading.Thread(
                    target=self._drain_output,
                    args=(proc.stdout,),
                    daemon=True,
                )
                self._log_stdout_thread.start()

            if stdin_iq_mode:
                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    args=(proc,),
                    daemon=True,
                )
                self._writer_thread.start()
            else:
                self._writer_thread = None

    def push_iq(self, iq_block, sample_rate_hz):
        """Queue IQ samples for nrsc5 when running in '-r -' mode."""
        with self._iq_cond:
            proc = self._process
            if (proc is None or proc.poll() is not None or
                    not self._stdin_iq_mode):
                return

            block = np.asarray(iq_block, dtype=np.complex64)
            if block.size == 0:
                return

            if len(self._iq_queue) >= self._iq_queue_max_blocks:
                self._iq_queue.popleft()
            self._iq_queue.append((block.copy(), float(sample_rate_hz)))
            self._iq_cond.notify()

    def pull_audio(self, frame_count):
        """Return exactly `frame_count` stereo frames when available, else None."""
        count = int(frame_count)
        if count <= 0:
            return None

        with self._lock:
            chunks = []
            total = 0

            if self._audio_buffer.shape[0] > 0:
                chunks.append(self._audio_buffer)
                total += self._audio_buffer.shape[0]

            while total < count and self._audio_queue:
                block = self._audio_queue.popleft()
                chunks.append(block)
                total += block.shape[0]

            if total < count:
                if chunks:
                    if len(chunks) == 1:
                        self._audio_buffer = chunks[0]
                    else:
                        self._audio_buffer = np.concatenate(chunks, axis=0)
                return None

            if len(chunks) == 1:
                merged = chunks[0]
            else:
                merged = np.concatenate(chunks, axis=0)

            out = merged[:count]
            self._audio_buffer = merged[count:]
            return out

    def stop(self):
        """Stop nrsc5 if active."""
        with self._iq_cond:
            proc = self._process
            log_stderr_thread = self._log_stderr_thread
            log_stdout_thread = self._log_stdout_thread
            writer_thread = self._writer_thread
            audio_thread = self._audio_thread
            self._process = None
            self._log_stderr_thread = None
            self._log_stdout_thread = None
            self._writer_thread = None
            self._audio_thread = None
            self._frequency_hz = None
            self._active_program = None
            self._stdin_iq_mode = False
            self._audio_stdout_mode = False
            self._uses_frequency_arg = True
            self._writer_stop = True
            self._iq_queue.clear()
            self._audio_queue.clear()
            self._audio_buffer = np.zeros((0, 2), dtype=np.float32)
            self._reset_iq_resampler_state()
            self._reset_audio_resampler_state()
            self._reset_metadata_locked()
            self._last_audio_time_s = 0.0
            self._iq_cond.notify_all()

        if proc is None:
            return

        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception:
            pass

        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

        for thread in (log_stderr_thread, log_stdout_thread, writer_thread, audio_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

    def poll(self):
        """
        Refresh process state.

        Returns:
            True when running, False when stopped/exited.
        """
        with self._lock:
            proc = self._process
        if proc is None:
            return False

        rc = proc.poll()
        if rc is None:
            return True

        with self._iq_cond:
            if self._process is proc:
                log_stderr_thread = self._log_stderr_thread
                log_stdout_thread = self._log_stdout_thread
                writer_thread = self._writer_thread
                audio_thread = self._audio_thread
                self._process = None
                self._log_stderr_thread = None
                self._log_stdout_thread = None
                self._writer_thread = None
                self._audio_thread = None
                self._frequency_hz = None
                self._active_program = None
                self._stdin_iq_mode = False
                self._audio_stdout_mode = False
                self._uses_frequency_arg = True
                self._writer_stop = True
                self._iq_queue.clear()
                self._audio_queue.clear()
                self._audio_buffer = np.zeros((0, 2), dtype=np.float32)
                self._reset_iq_resampler_state()
                self._reset_audio_resampler_state()
                self._reset_metadata_locked()
                self._last_audio_time_s = 0.0
                self._iq_cond.notify_all()
                if rc != 0:
                    detail = f" ({self.last_output_line})" if self.last_output_line else ""
                    self.last_error = f"nrsc5 exited with code {rc}{detail}"
            else:
                log_stderr_thread = None
                log_stdout_thread = None
                writer_thread = None
                audio_thread = None

        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

        for thread in (log_stderr_thread, log_stdout_thread, writer_thread, audio_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.2)

        return False

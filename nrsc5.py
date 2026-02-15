#!/usr/bin/env python3
"""
Optional HD Radio demodulation hooks for pjfm.

Supports NRSC5 Python bindings only.
"""

from collections import deque
import ctypes
import importlib.util
import os
from pathlib import Path
import platform
import re
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
    Manage NRSC5 HD Radio decode via Python bindings.

    The decoder accepts IQ pushes from pjfm, converts them to CU8 at NRSC5's
    expected input rate, and feeds them into libnrsc5 through `open_pipe()`.

    Environment overrides:
      - PJFM_NRSC5_PY_BINDINGS: Python binding module path (nrsc5.py)
      - PJFM_NRSC5_LIB: explicit libnrsc5 shared library path
      - PJFM_NRSC5_PROGRAM: HD subprogram number (default: 0)
    """

    TARGET_IQ_RATE_HZ = 1_488_375.0
    TARGET_AUDIO_IN_RATE_HZ = 44_100.0
    TARGET_AUDIO_OUT_RATE_HZ = 48_000.0
    IQ_QUEUE_MAX_BLOCKS_DEFAULT = 8
    AUDIO_QUEUE_MAX_BLOCKS_DEFAULT = 64
    IQ_TARGET_RMS_DEFAULT = 0.18
    IQ_GAIN_MIN = 0.2
    IQ_GAIN_MAX = 120.0
    _RE_STATION_NAME = re.compile(r"Station name:\s*(?P<name>.+)$")
    _RE_AUDIO_PROGRAM = re.compile(r"Audio program\s+(?P<num>\d+):\s*(?P<name>[^,]+)")
    _RE_AUDIO_SERVICE = re.compile(r"Audio service\s+(?P<num>\d+):\s*(?P<name>[^,]+)")
    _RE_AUDIO_PROGRAM_DESC = re.compile(
        r"Audio program\s+(?P<num>\d+):\s*(?P<access>[^,]+),\s*type:\s*(?P<type>[^,]+)"
    )
    _RE_AUDIO_SERVICE_DESC = re.compile(
        r"Audio service\s+(?P<num>\d+):\s*(?P<access>[^,]+),\s*type:\s*(?P<type>[^,]+)"
    )
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
    _RE_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
    _RE_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _RE_ESC_SINGLE = re.compile(r"\x1b[@-_]")
    _RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    BACKEND_PYTHON = "python"

    def __init__(self, program=0):
        program_override = os.environ.get("PJFM_NRSC5_PROGRAM", "").strip()
        if program_override:
            try:
                self.program = int(program_override)
            except ValueError:
                self.program = int(program)
        else:
            self.program = int(program)

        self._python_bindings = None
        self._python_binding_path = ""
        self._python_library_path = ""
        self._python_binding_error = ""
        self._python_decoder = None
        self._python_lost_device = False
        self._python_sync = False
        self._python_seen_program_zero = False

        self._lock = threading.Lock()
        self._iq_cond = threading.Condition(self._lock)
        self._process = None
        self._frequency_hz = None
        self._active_program = None
        self._stdin_iq_mode = False
        self._audio_stdout_mode = False
        self._uses_frequency_arg = True

        (
            self._python_bindings,
            self._python_binding_path,
            self._python_library_path,
            self._python_binding_error,
        ) = self._load_python_bindings()
        self._backend = self.BACKEND_PYTHON

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
        self._stats = {}
        self._reset_metadata_locked()
        self._reset_stats_locked()

    @property
    def available(self):
        """True if NRSC5 Python bindings are available."""
        return self._python_bindings is not None

    @property
    def unavailable_reason(self):
        """Human-readable reason when hooks are unavailable."""
        if self.available:
            return ""
        if self._python_binding_error:
            return self._python_binding_error
        return "NRSC5 Python bindings unavailable"

    @property
    def is_running(self):
        """True if the nrsc5 decoder backend is currently active."""
        return self.poll()

    @property
    def backend(self):
        """Selected NRSC5 backend."""
        return self._backend

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

    @property
    def stats_snapshot(self):
        """Latest decoder/sync diagnostics from NRSC5 events."""
        with self._lock:
            return dict(self._stats)

    def _reset_metadata_locked(self):
        """Reset cached metadata state. Caller must hold `_lock`."""
        self._metadata = {
            "station_name": "",
            "program_name": "",
            "program_number": None,
            "service_name": "",
            "service_number": None,
            "genre_program_number": None,
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

    def _reset_stats_locked(self):
        """Reset cached decoder diagnostics. Caller must hold `_lock`."""
        self._stats = {
            "sync": False,
            "sync_count": 0,
            "lost_sync_count": 0,
            "last_sync_freq_offset_hz": None,
            "last_sync_psmi": None,
            "mer_lower_db": None,
            "mer_upper_db": None,
            "ber_cber": None,
            "agc_gain_db": None,
            "agc_peak_dbfs": None,
            "agc_is_final": None,
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
        if self._python_seen_program_zero:
            return 0
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
        if text.lower() in {"public", "restricted"}:
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

        prog_desc_match = self._RE_AUDIO_PROGRAM_DESC.search(line)
        if prog_desc_match:
            self._update_program_metadata_locked(
                "program_name",
                "program_number",
                prog_desc_match.group("num"),
                prog_desc_match.group("access"),
            )
            self._update_program_metadata_locked(
                "genre",
                "genre_program_number",
                prog_desc_match.group("num"),
                prog_desc_match.group("type"),
            )
            return

        svc_desc_match = self._RE_AUDIO_SERVICE_DESC.search(line)
        if svc_desc_match:
            self._update_program_metadata_locked(
                "service_name",
                "service_number",
                svc_desc_match.group("num"),
                svc_desc_match.group("access"),
            )
            self._update_program_metadata_locked(
                "genre",
                "genre_program_number",
                svc_desc_match.group("num"),
                svc_desc_match.group("type"),
            )
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
    def _lib_names_for_platform():
        system = platform.system()
        if system == "Windows":
            return ("libnrsc5.dll", "nrsc5.dll")
        if system == "Darwin":
            return ("libnrsc5.dylib",)
        return ("libnrsc5.so",)

    def _candidate_binding_module_paths(self):
        candidates = []
        env_path = os.environ.get("PJFM_NRSC5_PY_BINDINGS", "").strip()
        if env_path:
            path = Path(env_path).expanduser()
            if path.is_dir():
                path = path / "nrsc5.py"
            candidates.append(path)

        this_dir = Path(__file__).resolve().parent
        candidates.append(this_dir / "support" / "nrsc5.py")
        candidates.append(this_dir.parent / "nrsc5" / "support" / "nrsc5.py")

        seen = set()
        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                continue
            if str(resolved) in seen or not resolved.is_file():
                continue
            seen.add(str(resolved))
            yield resolved

    def _candidate_binding_library_paths(self, binding_path):
        names = self._lib_names_for_platform()
        explicit = os.environ.get("PJFM_NRSC5_LIB", "").strip()
        if explicit:
            path = Path(explicit).expanduser()
            if path.is_file():
                yield path.resolve()

        explicit_dir = os.environ.get("PJFM_NRSC5_LIB_DIR", "").strip()
        if explicit_dir:
            base = Path(explicit_dir).expanduser()
            for name in names:
                path = base / name
                if path.is_file():
                    yield path.resolve()

        base = binding_path.parent.parent
        for name in names:
            for candidate in (
                base / "build" / "src" / name,
                base / "src" / name,
                Path.home() / ".local" / "lib" / name,
                Path("/usr/local/lib") / name,
                Path("/opt/homebrew/lib") / name,
                Path("/usr/lib") / name,
            ):
                if candidate.is_file():
                    yield candidate.resolve()

    @staticmethod
    def _import_binding_module_from_path(binding_path):
        module_name = f"_pjfm_nrsc5_bindings_{abs(hash(str(binding_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(binding_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Python bindings spec from {binding_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_python_bindings(self):
        errors = []
        for binding_path in self._candidate_binding_module_paths():
            try:
                module = self._import_binding_module_from_path(binding_path)
            except Exception as exc:
                errors.append(f"{binding_path}: {exc}")
                continue

            if not all(hasattr(module, attr) for attr in ("NRSC5", "EventType", "Mode")):
                errors.append(f"{binding_path}: missing required NRSC5 binding attributes")
                continue

            tried = set()
            candidates = [None]
            candidates.extend(list(self._candidate_binding_library_paths(binding_path)))
            for lib_path in candidates:
                key = "" if lib_path is None else str(lib_path)
                if key in tried:
                    continue
                tried.add(key)
                try:
                    module.NRSC5.libnrsc5 = None
                    if lib_path is not None:
                        module.NRSC5.libnrsc5 = ctypes.cdll.LoadLibrary(str(lib_path))
                    _probe = module.NRSC5(lambda *_args: None)
                    del _probe
                    return (
                        module,
                        str(binding_path),
                        "" if lib_path is None else str(lib_path),
                        "",
                    )
                except Exception as exc:
                    source = "default loader" if lib_path is None else str(lib_path)
                    errors.append(f"{binding_path} via {source}: {exc}")

        reason = "NRSC5 Python bindings were not found"
        if errors:
            reason = f"{reason} ({errors[-1]})"
        return None, "", "", reason

    @staticmethod
    def _format_program_type(value):
        if value is None:
            return ""
        if hasattr(value, "name"):
            text = str(value.name)
        else:
            text = str(value)
        text = text.strip().replace("_", " ")
        return text.title() if text else ""

    def _program_matches_active_locked(self, number):
        if number is None:
            return True
        try:
            num = int(number)
        except (TypeError, ValueError):
            return True
        active = self._active_program_index_locked()
        if active is None:
            return True
        if num == active:
            return True
        if num != (active + 1):
            return False
        if self._python_seen_program_zero:
            return False
        return True

    def _set_last_output_line_locked(self, text):
        line = self._sanitize_output_line(text)
        if line:
            self.last_output_line = line[-240:]

    def _observe_program_number_locked(self, number):
        try:
            num = int(number)
        except (TypeError, ValueError):
            return
        if num == 0:
            self._python_seen_program_zero = True

    def _handle_python_event(self, evt_type, evt):
        if evt is None:
            return

        event_name = getattr(evt_type, "name", str(evt_type))
        now = time.monotonic()

        if event_name == "LOST_DEVICE":
            with self._lock:
                self.last_error = "nrsc5 lost input device"
                self._set_last_output_line_locked("Lost input device")
                self._python_lost_device = True
                self._stats["sync"] = False
                self._stats["updated_at_s"] = now
            return

        if event_name == "SYNC":
            with self._lock:
                freq_offset = getattr(evt, "freq_offset", 0.0)
                psmi = getattr(evt, "psmi", None)
                try:
                    freq_offset_val = float(freq_offset)
                except (TypeError, ValueError):
                    freq_offset_val = None
                try:
                    psmi_val = int(psmi)
                except (TypeError, ValueError):
                    psmi_val = None
                display_offset = 0.0 if freq_offset_val is None else freq_offset_val
                self._set_last_output_line_locked(f"SYNC lock (offset {display_offset:.1f} Hz)")
                self._python_sync = True
                self._stats["sync"] = True
                self._stats["sync_count"] = int(self._stats.get("sync_count", 0)) + 1
                self._stats["last_sync_freq_offset_hz"] = freq_offset_val
                self._stats["last_sync_psmi"] = psmi_val
                self._stats["updated_at_s"] = now
            return

        if event_name == "LOST_SYNC":
            with self._lock:
                self._set_last_output_line_locked("SYNC lost")
                self._python_sync = False
                self._stats["sync"] = False
                self._stats["lost_sync_count"] = int(self._stats.get("lost_sync_count", 0)) + 1
                self._stats["updated_at_s"] = now
            return

        if event_name == "MER":
            with self._lock:
                lower = getattr(evt, "lower", None)
                upper = getattr(evt, "upper", None)
                try:
                    lower_val = float(lower)
                except (TypeError, ValueError):
                    lower_val = None
                try:
                    upper_val = float(upper)
                except (TypeError, ValueError):
                    upper_val = None
                self._stats["mer_lower_db"] = lower_val
                self._stats["mer_upper_db"] = upper_val
                self._stats["updated_at_s"] = now
            return

        if event_name == "BER":
            with self._lock:
                cber = getattr(evt, "cber", None)
                try:
                    cber_val = float(cber)
                except (TypeError, ValueError):
                    cber_val = None
                self._stats["ber_cber"] = cber_val
                self._stats["updated_at_s"] = now
            return

        if event_name == "AGC":
            with self._lock:
                gain_db = getattr(evt, "gain_db", None)
                peak_dbfs = getattr(evt, "peak_dbfs", None)
                is_final = getattr(evt, "is_final", None)
                try:
                    gain_db_val = float(gain_db)
                except (TypeError, ValueError):
                    gain_db_val = None
                try:
                    peak_dbfs_val = float(peak_dbfs)
                except (TypeError, ValueError):
                    peak_dbfs_val = None
                is_final_val = None if is_final is None else bool(is_final)
                self._stats["agc_gain_db"] = gain_db_val
                self._stats["agc_peak_dbfs"] = peak_dbfs_val
                self._stats["agc_is_final"] = is_final_val
                self._stats["updated_at_s"] = now
            return

        if event_name == "STATION_NAME":
            with self._lock:
                self._update_text_metadata_locked("station_name", getattr(evt, "name", ""))
                self._set_last_output_line_locked(f"Station name: {getattr(evt, 'name', '')}")
            return

        if event_name == "STATION_SLOGAN":
            with self._lock:
                self._update_text_metadata_locked("station_slogan", getattr(evt, "slogan", ""))
            return

        if event_name == "STATION_MESSAGE":
            with self._lock:
                self._update_text_metadata_locked("station_message", getattr(evt, "message", ""))
            return

        if event_name == "EMERGENCY_ALERT":
            with self._lock:
                self._update_text_metadata_locked("emergency_alert", getattr(evt, "message", ""))
                self._set_last_output_line_locked(f"Alert: {getattr(evt, 'message', '')}")
            return

        if event_name == "HERE_IMAGE":
            image_type = getattr(getattr(evt, "image_type", None), "name", "")
            image_name = str(getattr(evt, "name", "")).strip()
            image_time = ""
            time_utc = getattr(evt, "time_utc", None)
            if time_utc is not None:
                try:
                    image_time = time_utc.isoformat().replace("+00:00", "Z")
                except Exception:
                    image_time = str(time_utc)
            with self._lock:
                if image_type == "WEATHER":
                    self._update_text_metadata_locked("here_weather_time_utc", image_time)
                    self._update_text_metadata_locked("here_weather_name", image_name)
                elif image_type == "TRAFFIC":
                    self._update_text_metadata_locked("here_traffic_time_utc", image_time)
                    self._update_text_metadata_locked("here_traffic_name", image_name)
                if image_name:
                    self._set_last_output_line_locked(f"HERE {image_type}: {image_name}")
            return

        if event_name in {"AUDIO_SERVICE", "AUDIO_SERVICE_DESCRIPTOR"}:
            number = getattr(evt, "program", None)
            genre = self._format_program_type(getattr(evt, "type", None))
            with self._lock:
                self._observe_program_number_locked(number)
                if genre:
                    self._update_program_metadata_locked(
                        "genre",
                        "genre_program_number",
                        number,
                        genre,
                    )
            return

        if event_name == "SIG":
            services = evt if isinstance(evt, (list, tuple)) else []
            with self._lock:
                for service in services:
                    if not getattr(service, "audio_component", None):
                        continue
                    number = getattr(service, "number", None)
                    self._observe_program_number_locked(number)
                    name = str(getattr(service, "name", "")).strip()
                    if not name:
                        continue
                    self._update_program_metadata_locked("program_name", "program_number", number, name)
                    self._update_program_metadata_locked("service_name", "service_number", number, name)
                    self._update_text_metadata_locked("sig_service_name", name)
            return

        if event_name == "SIS":
            with self._lock:
                self._update_text_metadata_locked("station_name", getattr(evt, "name", ""))
                self._update_text_metadata_locked("station_slogan", getattr(evt, "slogan", ""))
                self._update_text_metadata_locked("station_message", getattr(evt, "message", ""))
                alert_text = getattr(evt, "alert", "")
                if alert_text:
                    self._update_text_metadata_locked("emergency_alert", alert_text)
                for svc in getattr(evt, "audio_services", []) or []:
                    program = getattr(svc, "program", None)
                    self._observe_program_number_locked(program)
                    genre = self._format_program_type(getattr(svc, "type", None))
                    if genre:
                        self._update_program_metadata_locked(
                            "genre",
                            "genre_program_number",
                            program,
                            genre,
                        )
            return

        if event_name == "ID3":
            program = getattr(evt, "program", None)
            with self._lock:
                self._observe_program_number_locked(program)
                if not self._program_matches_active_locked(program):
                    return
                self._update_text_metadata_locked("title", getattr(evt, "title", ""))
                self._update_text_metadata_locked("artist", getattr(evt, "artist", ""))
                self._update_text_metadata_locked("album", getattr(evt, "album", ""))
                self._update_text_metadata_locked("genre", getattr(evt, "genre", ""))
                title = getattr(evt, "title", "") or ""
                artist = getattr(evt, "artist", "") or ""
                if title and artist:
                    self._set_last_output_line_locked(f"{title} - {artist}")
                elif title:
                    self._set_last_output_line_locked(f"Title: {title}")
            return

        if event_name == "AUDIO":
            program = getattr(evt, "program", None)
            raw = getattr(evt, "data", None)
            if not raw:
                return

            with self._lock:
                self._observe_program_number_locked(program)
                if not self._program_matches_active_locked(program):
                    return

            pcm16 = np.frombuffer(raw, dtype="<i2")
            if pcm16.size < 2:
                return
            if pcm16.size & 1:
                pcm16 = pcm16[:-1]
                if pcm16.size < 2:
                    return
            frames = pcm16.reshape(-1, 2).astype(np.float32) / 32768.0
            out = self._resample_audio(frames)
            if out.shape[0] == 0:
                return

            with self._lock:
                if self._python_decoder is None:
                    return
                if len(self._audio_queue) >= self._audio_queue_max_blocks:
                    self._audio_queue.popleft()
                self._audio_queue.append(out)
                self._audio_bytes_out_total += int(len(raw))
                self._last_audio_time_s = now
            return

    def _python_writer_loop(self, decoder):
        while True:
            with self._iq_cond:
                while (
                    not self._writer_stop and
                    self._python_decoder is decoder and
                    not self._iq_queue
                ):
                    self._iq_cond.wait(timeout=0.2)

                if self._writer_stop or self._python_decoder is not decoder:
                    return

                iq_block, sample_rate_hz = self._iq_queue.popleft()

            payload = self._iq_to_cu8(iq_block, sample_rate_hz)
            if not payload:
                continue
            try:
                decoder.pipe_samples_cu8(payload)
                with self._lock:
                    if self._python_decoder is decoder:
                        self._iq_bytes_in_total += int(len(payload))
            except Exception as exc:
                with self._lock:
                    if self._python_decoder is decoder and not self.last_error:
                        self.last_error = f"nrsc5 pipe write failed: {exc}"
                return

    def set_program(self, program):
        """Set the HD subprogram index for subsequent starts/restarts."""
        new_program = int(program)
        if new_program < 0:
            raise ValueError("program must be >= 0")
        with self._lock:
            self.program = new_program
            if self._python_decoder is not None:
                self._active_program = new_program

    @classmethod
    def _sanitize_output_line(cls, line):
        text = str(line or "")
        if not text:
            return ""
        text = text.replace("\r", "").replace("\n", "")
        text = cls._RE_OSC.sub("", text)
        text = cls._RE_CSI.sub("", text)
        text = cls._RE_ESC_SINGLE.sub("", text)
        text = cls._RE_CONTROL.sub("", text)
        return text.strip()

    def _drain_output(self, stream):
        if stream is None:
            return
        try:
            for raw_line in stream:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="ignore")
                else:
                    line = raw_line
                line = self._sanitize_output_line(line)
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

    def start(self, frequency_hz):
        """Start NRSC5 decoder (or retune/program-switch if already running)."""
        freq_hz = int(round(float(frequency_hz)))
        with self._lock:
            desired_program = int(self.program)
        with self._lock:
            running_decoder = self._python_decoder
            running = (
                running_decoder is not None and
                not self._writer_stop and
                not self._python_lost_device
            )
            if running:
                self._frequency_hz = freq_hz
                self._active_program = desired_program
                return

        self.stop()
        if not self.available:
            raise RuntimeError(f"HD Radio unavailable: {self.unavailable_reason}")

        bindings = self._python_bindings
        decoder = None
        try:
            decoder = bindings.NRSC5(self._handle_python_event)
            decoder.open_pipe()
            decoder.set_mode(bindings.Mode.FM)
            decoder.start()
        except Exception as exc:
            if decoder is not None:
                try:
                    decoder.close()
                except Exception:
                    pass
            self.last_error = f"Failed to start nrsc5 Python backend: {exc}"
            raise RuntimeError(self.last_error) from exc

        with self._lock:
            self._python_decoder = decoder
            self._process = None
            self._frequency_hz = freq_hz
            self._active_program = desired_program
            self._stdin_iq_mode = True
            self._audio_stdout_mode = False
            self._uses_frequency_arg = False
            self._writer_stop = False
            self._python_lost_device = False
            self._python_sync = False
            self._python_seen_program_zero = False
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
            self._reset_stats_locked()
            self._log_stderr_thread = None
            self._log_stdout_thread = None
            self._audio_thread = None
            self._writer_thread = threading.Thread(
                target=self._python_writer_loop,
                args=(decoder,),
                daemon=True,
            )
            self._writer_thread.start()

    def push_iq(self, iq_block, sample_rate_hz):
        """Queue IQ samples for the running NRSC5 decoder."""
        with self._iq_cond:
            if self._python_decoder is None or self._writer_stop:
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
        """Stop NRSC5 if active."""
        with self._iq_cond:
            decoder = self._python_decoder
            writer_thread = self._writer_thread
            self._python_decoder = None
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
            self._reset_stats_locked()
            self._last_audio_time_s = 0.0
            self._iq_cond.notify_all()

        if decoder is None:
            return

        try:
            decoder.stop()
        except Exception:
            pass
        try:
            decoder.close()
        except Exception:
            pass

        if writer_thread and writer_thread.is_alive():
            writer_thread.join(timeout=0.5)

    def poll(self):
        """
        Refresh process state.

        Returns:
            True when running, False when stopped/exited.
        """
        with self._lock:
            decoder = self._python_decoder
            writer_thread = self._writer_thread
            lost_device = self._python_lost_device
            writer_stop = self._writer_stop
            has_error = bool(self.last_error)
        if decoder is None:
            return False
        if lost_device or writer_stop:
            self.stop()
            return False
        if writer_thread and not writer_thread.is_alive():
            if not has_error:
                self.last_error = "nrsc5 IQ writer stopped"
            self.stop()
            return False
        return True

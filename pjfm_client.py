#!/usr/bin/env python3
"""
pjfm_client - UDP audio client for pjfm server

Receives audio streamed by pjfm --server and plays it locally.
Maintains a configurable jitter buffer (default 1 second).

Usage:
    ./pjfm_client.py <server-host> [--port PORT] [--buffer SECONDS]

Controls:
    Ctrl-C  Quit
"""

import argparse
import json
import os
import select
import socket
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    from rich.align import Align
    from rich import box as rich_box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _rich_available = True
except ImportError:
    _rich_available = False

# Must match pjfm.py
_MAGIC = b'PJ'
_VERSION = 1
_HEADER_FMT = '>2sBBIII'   # magic(2) ver(1) ch(1) seq(4) rate(4) frames(4)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# How often the client sends keepalive/reconnect packets (seconds)
_HEARTBEAT_INTERVAL_S = 3.0
# How long without a packet before we reset playback state and wait for refill
_RECONNECT_TIMEOUT_S = 5.0


class _RingBuffer:
    """Thread-safe stereo float32 ring buffer."""

    def __init__(self, sample_rate, channels, capacity_s):
        self.sample_rate = sample_rate
        self.channels = channels
        self.capacity = int(sample_rate * capacity_s)
        self._buf = np.zeros((self.capacity, channels), dtype=np.float32)
        self._wp = 0
        self._rp = 0
        self._lock = threading.Lock()

    def write(self, frames):
        """Append frames (shape N×channels).  Drops oldest on overflow."""
        with self._lock:
            n = len(frames)
            if n > self.capacity:
                frames = frames[-self.capacity:]
                n = self.capacity

            used = (self._wp - self._rp) % self.capacity
            space = self.capacity - used - 1
            if n > space:
                # Advance read pointer to make room (drops oldest audio)
                self._rp = (self._rp + (n - space)) % self.capacity

            end = self._wp + n
            if end <= self.capacity:
                self._buf[self._wp:end] = frames
            else:
                first = self.capacity - self._wp
                self._buf[self._wp:] = frames[:first]
                self._buf[:n - first] = frames[first:]
            self._wp = end % self.capacity

    def read(self, n):
        """Read n frames. Returns silence for any underrun portion."""
        with self._lock:
            available = (self._wp - self._rp) % self.capacity
            actual = min(n, available)
            out = np.zeros((n, self.channels), dtype=np.float32)
            if actual > 0:
                end = self._rp + actual
                if end <= self.capacity:
                    out[:actual] = self._buf[self._rp:end]
                else:
                    first = self.capacity - self._rp
                    out[:first] = self._buf[self._rp:]
                    out[first:actual] = self._buf[:actual - first]
                self._rp = end % self.capacity
            return out

    @property
    def level_s(self):
        with self._lock:
            return ((self._wp - self._rp) % self.capacity) / self.sample_rate

    @property
    def level_ms(self):
        return self.level_s * 1000.0


class TCPControlClient:
    """Connects to pjfm server control port; receives status, sends commands."""

    RECONNECT_INTERVAL_S = 5.0

    def __init__(self, host, port=14551):
        self._addr = (host, port)
        self._sock = None
        self._lock = threading.Lock()
        self.status = {}             # latest parsed status dict from server
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def send_command(self, cmd_dict):
        """Send a JSON command to the server (best-effort)."""
        with self._lock:
            sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall((json.dumps(cmd_dict) + "\n").encode())
        except OSError:
            pass

    def _connect_loop(self):
        while self._running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(5.0)
                sock.connect(self._addr)
                sock.settimeout(None)
                with self._lock:
                    self._sock = sock
                self._recv_loop(sock)
            except OSError:
                pass
            finally:
                with self._lock:
                    if self._sock is sock:
                        self._sock = None
                try:
                    sock.close()
                except Exception:
                    pass
            if self._running:
                time.sleep(self.RECONNECT_INTERVAL_S)

    def _recv_loop(self, sock):
        buf = b""
        while self._running:
            try:
                data = sock.recv(4096)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    try:
                        self.status = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        pass


def _signal_bar(dbm, width=24):
    """Colored signal strength bar mapped to [-100, -20] dBm."""
    normalized = (dbm - (-100.0)) / 80.0
    normalized = max(0.0, min(1.0, normalized))
    filled = int(normalized * width)
    t = Text()
    style = "green" if dbm > -60 else ("yellow" if dbm > -75 else "red")
    t.append("█" * filled, style=style)
    t.append("░" * (width - filled), style="dim")
    return t


def _build_client_display(control, audio, server_addr, buffer_s, freq_input=None):
    """Build the Rich Panel shown in the client interactive UI."""
    s = control.status if control else {}

    table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    table.add_column("Label", style="cyan", width=12, justify="right")
    table.add_column("Value")

    # ── Server connection ──────────────────────────────────────
    addr_str = f"{server_addr[0]}:{server_addr[1]}"
    if s:
        addr_style = "white"
    else:
        addr_str += "  (connecting…)"
        addr_style = "dim"
    table.add_row("Server:", Text(addr_str, style=addr_style))
    table.add_row("", "")

    if s:
        # Frequency
        freq = s.get("freq_mhz", 0.0)
        table.add_row("Frequency:", Text(f"{freq:.3f} MHz", style="green bold"))

        # Signal
        dbm = s.get("signal_dbm", -100.0)
        sig_text = _signal_bar(dbm)
        sig_text.append(f"  {dbm:.1f} dBm", style="green bold")
        table.add_row("Signal:", sig_text)

        # Mode
        mode = s.get("mode", "")
        blend = s.get("stereo_blend_pct")
        mode_text = Text()
        if mode == "Stereo":
            if blend is not None and blend < 99:
                mode_text.append(f"Stereo ({blend}%)", style="yellow bold")
            else:
                mode_text.append("Stereo", style="green bold")
        elif mode == "Mono":
            mode_text.append("Mono", style="yellow")
        elif mode == "NBFM":
            mode_text.append("NBFM", style="cyan bold")
        elif mode == "HD":
            mode_text.append("HD", style="green bold")
        else:
            mode_text.append(mode, style="white")
        table.add_row("Mode:", mode_text)

        # HD status (only when not OFF)
        hd_status = s.get("hd_status", "OFF")
        if hd_status != "OFF":
            hd_text = Text()
            hd_label = s.get("hd_label")
            hd_ber = s.get("hd_ber_pct")
            if hd_status == "ON":
                hd_text.append("ON", style="green bold")
                if hd_label:
                    hd_text.append(f"  {hd_label}", style="cyan")
                if hd_ber is not None:
                    hd_text.append(f"  BER {hd_ber:.2f}%", style="dim")
            elif hd_status == "synced":
                hd_text.append("synced", style="green")
                if hd_ber is not None:
                    hd_text.append(f"  BER {hd_ber:.2e}", style="dim")
            elif hd_status == "seeking":
                hd_text.append("seeking…", style="yellow")
            elif hd_status == "error":
                hd_text.append("error", style="red bold")
            else:
                hd_text.append(hd_status, style="dim")
            table.add_row("HD:", hd_text)

        # Metadata
        station = s.get("station")
        ps_name = s.get("ps_name")
        radio_text = s.get("radio_text")
        artist = s.get("artist")
        title = s.get("title")
        if station:
            table.add_row("Station:", Text(station, style="cyan bold"))
        if ps_name:
            table.add_row("Name:", Text(ps_name, style="green bold"))
        if artist and title:
            table.add_row("Now Playing:",
                          Text(f"{artist}  —  {title}", style="green"))
        elif title:
            table.add_row("Title:", Text(title, style="green"))
        elif radio_text:
            table.add_row("Text:", Text(radio_text[:72], style="green"))

        table.add_row("", "")

    # ── Local audio stats ──────────────────────────────────────
    buf_ms = audio._buf.level_ms if audio._buf else 0.0
    capacity_ms = buffer_s * 2500.0
    filled = int((buf_ms / capacity_ms) * 24) if capacity_ms > 0 else 0
    filled = max(0, min(24, filled))
    buf_text = Text()
    buf_text.append("█" * filled, style="green")
    buf_text.append("░" * (24 - filled), style="dim")
    buf_text.append(f"  {buf_ms:.0f} ms", style="yellow")
    if audio._buf is None:
        buf_text.append("  waiting for stream", style="dim")
    elif audio._playing:
        buf_text.append("  playing", style="green bold")
    else:
        buf_text.append("  buffering", style="yellow")
    table.add_row("Buffer:", buf_text)

    if audio._buf:
        stream_text = Text(
            f"{audio._buf.sample_rate} Hz · {audio._buf.channels} ch"
            f"  (target {buffer_s:.1f} s)",
            style="white",
        )
        table.add_row("Stream:", stream_text)

    pkt_text = Text()
    pkt_text.append(f"rx {audio._received}", style="green")
    pkt_text.append("  dropped ", style="dim")
    if audio._dropped > 0:
        pkt_text.append(str(audio._dropped), style="red bold")
    else:
        pkt_text.append("0", style="dim")
    table.add_row("Packets:", pkt_text)

    table.add_row("", "")

    # ── Goto prompt (active while user is typing a frequency) ──
    if freq_input is not None:
        prompt = Text()
        prompt.append(freq_input or "", style="green bold")
        prompt.append("_", style="green bold")   # cursor
        prompt.append("  MHz  ", style="dim")
        prompt.append("Enter", style="yellow")
        prompt.append(" to tune  ", style="dim")
        prompt.append("Esc", style="yellow")
        prompt.append(" to cancel", style="dim")
        table.add_row("Goto:", prompt)
        table.add_row("", "")

    # ── Key hints ──────────────────────────────────────────────
    hints = Text()
    hints.append("← →", style="yellow")
    hints.append(" Tune  ", style="dim")
    hints.append("↑ ↓", style="yellow")
    hints.append(" Volume  ", style="dim")
    hints.append("g", style="yellow")
    hints.append(" Goto  ", style="dim")
    hints.append("h", style="yellow")
    hints.append(" HD  ", style="dim")
    hints.append("q", style="yellow")
    hints.append(" Quit", style="dim")
    table.add_row("", Align.center(hints))

    return Panel(
        Align.center(table),
        title="[bold cyan]📻 pjfm client[/]",
        border_style="cyan",
        box=rich_box.ROUNDED,
        padding=(1, 2),
    )


class UDPAudioClient:
    """Receives pjfm UDP audio stream and plays it via sounddevice."""

    def __init__(self, server_host, server_port=14550, buffer_s=1.0):
        self.server_addr = (server_host, server_port)
        self.buffer_s = buffer_s

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)

        self._buf = None          # created on first packet
        self._stream = None
        self._playing = False
        self._running = False

        self._last_seq = None
        self._received = 0
        self._dropped = 0
        self._last_packet_time = 0.0

    def start(self):
        self._running = True
        self._last_packet_time = time.monotonic()
        self._sock.sendto(b'CONNECT', self.server_addr)
        print(f"Connecting to {self.server_addr[0]}:{self.server_addr[1]} …")

        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()

    def stop(self):
        self._running = False
        try:
            self._sock.sendto(b'DISCONNECT', self.server_addr)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def run(self, control=None):
        """Block until Ctrl-C.  Uses Rich UI on a TTY, plain print otherwise."""
        self.start()
        is_tty = sys.stdout.isatty() and sys.stdin.isatty()

        if not is_tty or not _rich_available:
            # Fallback: plain periodic print, no raw mode
            try:
                while True:
                    time.sleep(5.0)
                    s = control.status if control else {}
                    if s:
                        buf_ms = self._buf.level_ms if self._buf else 0
                        freq = s.get("freq_mhz", "?")
                        dbm = s.get("signal_dbm", "?")
                        mode = s.get("mode", "")
                        print(f"{freq} MHz  {dbm} dBm  {mode}  [buf {buf_ms:.0f}ms]")
                    elif self._buf:
                        state = "playing" if self._playing else "buffering"
                        print(f"[{state}]  buffer {self._buf.level_ms:.0f} ms  "
                              f"rx {self._received}  dropped {self._dropped}")
                    else:
                        print("Waiting for server …")
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()
            return

        # Rich interactive mode
        console = Console()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            input_buf = ''
            freq_input = None   # None = normal mode; str = collecting frequency
            with Live(
                _build_client_display(control, self, self.server_addr, self.buffer_s),
                console=console,
                refresh_per_second=5,
                screen=True,
            ) as live:
                while True:
                    live.update(
                        _build_client_display(control, self, self.server_addr,
                                              self.buffer_s, freq_input=freq_input)
                    )

                    # Non-blocking drain of all available stdin bytes
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        try:
                            chunk = os.read(fd, 32)
                            if chunk:
                                input_buf += chunk.decode('utf-8', errors='ignore')
                        except (BlockingIOError, IOError):
                            pass

                    # Process accumulated input
                    while input_buf:
                        ch = input_buf[0]

                        # ── Frequency-input mode ───────────────────────
                        if freq_input is not None:
                            if ch in ('\r', '\n'):
                                # Commit: parse and send
                                try:
                                    freq_mhz = float(freq_input)
                                    if control:
                                        control.send_command(
                                            {"cmd": "tune_to", "freq_mhz": freq_mhz}
                                        )
                                except ValueError:
                                    pass
                                freq_input = None
                                input_buf = input_buf[1:]
                            elif ch in ('\x1b',):
                                # Escape — check for multi-byte sequence
                                if input_buf.startswith('\x1b[') or input_buf.startswith('\x1bO'):
                                    if len(input_buf) < 3:
                                        break
                                    input_buf = input_buf[3:]  # discard arrow etc.
                                else:
                                    freq_input = None          # lone Esc = cancel
                                    input_buf = input_buf[1:]
                            elif ch in ('\x08', '\x7f'):
                                freq_input = freq_input[:-1]
                                input_buf = input_buf[1:]
                            elif ch.isdigit() or (ch == '.' and '.' not in freq_input):
                                freq_input += ch
                                input_buf = input_buf[1:]
                            else:
                                input_buf = input_buf[1:]   # ignore other chars
                            continue

                        # ── Normal mode ────────────────────────────────
                        if ch in ('q', 'Q', '\x03'):
                            input_buf = ''
                            raise KeyboardInterrupt
                        elif ch == 'g':
                            freq_input = ''
                            input_buf = input_buf[1:]
                        elif ch == 'h':
                            if control:
                                control.send_command({"cmd": "hd_cycle"})
                            input_buf = input_buf[1:]
                        elif input_buf.startswith('\x1b[C') or input_buf.startswith('\x1bOC'):
                            if control:
                                control.send_command({"cmd": "tune", "dir": "up"})
                            input_buf = input_buf[3:]
                        elif input_buf.startswith('\x1b[D') or input_buf.startswith('\x1bOD'):
                            if control:
                                control.send_command({"cmd": "tune", "dir": "down"})
                            input_buf = input_buf[3:]
                        elif input_buf.startswith('\x1b[A') or input_buf.startswith('\x1bOA'):
                            if control:
                                control.send_command({"cmd": "volume", "dir": "up"})
                            input_buf = input_buf[3:]
                        elif input_buf.startswith('\x1b[B') or input_buf.startswith('\x1bOB'):
                            if control:
                                control.send_command({"cmd": "volume", "dir": "down"})
                            input_buf = input_buf[3:]
                        elif input_buf.startswith('\x1b[') or input_buf.startswith('\x1bO'):
                            if len(input_buf) < 3:
                                break   # wait for rest of sequence
                            input_buf = input_buf[3:]
                        elif ch == '\x1b':
                            if len(input_buf) == 1:
                                break
                            input_buf = input_buf[1:]
                        else:
                            input_buf = input_buf[1:]
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.stop()

    # ------------------------------------------------------------------ #

    def _heartbeat_loop(self):
        while self._running:
            try:
                # Always send CONNECT (not HEARTBEAT) so we auto-register
                # with a restarted server without any manual intervention.
                self._sock.sendto(b'CONNECT', self.server_addr)
            except OSError:
                pass

            # Watchdog: if the server has gone silent, reset playback so
            # we wait for the buffer to refill before resuming audio.
            if (self._buf is not None and
                    time.monotonic() - self._last_packet_time > _RECONNECT_TIMEOUT_S):
                if self._playing:
                    print("\nServer silent — waiting for reconnect …")
                    self._playing = False
                # Clear buffer so we get a clean refill on reconnect
                self._buf = _RingBuffer(
                    self._buf.sample_rate, self._buf.channels,
                    capacity_s=self.buffer_s * 2.5)
                self._last_seq = None

            time.sleep(_HEARTBEAT_INTERVAL_S)

    def _receive_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_packet(data)

    def _handle_packet(self, data):
        if len(data) < _HEADER_SIZE:
            return

        magic, version, channels, seq, sample_rate, num_frames = \
            struct.unpack_from(_HEADER_FMT, data)

        if magic != _MAGIC or version != _VERSION:
            return

        expected_bytes = num_frames * channels * 4
        payload = data[_HEADER_SIZE:]
        if len(payload) < expected_bytes:
            return

        # Initialise buffer and sounddevice stream on first packet
        if self._buf is None:
            self._buf = _RingBuffer(sample_rate, channels,
                                    capacity_s=self.buffer_s * 2.5)
            self._start_stream(sample_rate, channels)
            print(f"Stream: {sample_rate} Hz, {channels}ch  "
                  f"(buffer target {self.buffer_s:.1f} s)")

        self._last_packet_time = time.monotonic()

        # Sequence-number gap detection.
        # A gap >= 0x80000000 means the sender reset its counter (e.g. FM→HD
        # transition); treat it as a re-sync rather than billions of drops.
        if self._last_seq is not None:
            expected_seq = (self._last_seq + 1) & 0xFFFFFFFF
            if seq != expected_seq:
                gap = (seq - expected_seq) & 0xFFFFFFFF
                if gap < 0x80000000:
                    self._dropped += gap
                # else: sequence reset — re-sync silently
        self._last_seq = seq
        self._received += 1

        audio = np.frombuffer(payload[:expected_bytes],
                              dtype=np.float32).reshape(num_frames, channels)
        self._buf.write(audio)

        # Start playback once the buffer has reached its target level
        if not self._playing and self._buf.level_s >= self.buffer_s:
            self._playing = True
            print(f"Buffer ready ({self._buf.level_ms:.0f} ms) — starting playback")

    def _start_stream(self, sample_rate, channels):
        def callback(outdata, frames, time_info, status):
            if self._playing:
                outdata[:] = self._buf.read(frames)
            else:
                outdata[:] = 0

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            blocksize=1024,
            callback=callback,
            latency='low',
        )
        self._stream.start()


def main():
    parser = argparse.ArgumentParser(
        description="pjfm_client — UDP audio client for pjfm --server"
    )
    parser.add_argument("server", help="Server hostname or IP address")
    parser.add_argument(
        "--port", type=int, default=14550,
        help="UDP port (default: 14550)"
    )
    parser.add_argument(
        "--buffer", type=float, default=1.0, metavar="SECONDS",
        help="Jitter buffer size in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--control-port", type=int, default=None, metavar="PORT",
        help="TCP control port (default: audio-port + 1)"
    )
    args = parser.parse_args()

    ctrl_port = args.control_port if args.control_port else args.port + 1
    control = TCPControlClient(args.server, ctrl_port)
    control.start()

    client = UDPAudioClient(args.server, server_port=args.port, buffer_s=args.buffer)
    try:
        client.run(control=control)
    finally:
        control.stop()
    print("Goodbye.")


if __name__ == "__main__":
    main()

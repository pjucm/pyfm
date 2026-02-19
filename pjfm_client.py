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
        self.status_line = ""        # latest status string from server
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
                    self.status_line = line


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
        """Block until Ctrl-C.  If control is a TCPControlClient, display its
        status line and process arrow-key commands in raw terminal mode."""
        self.start()
        is_tty = sys.stdout.isatty() and sys.stdin.isatty()

        if not is_tty or control is None:
            # Fallback: plain periodic print, no raw mode
            try:
                while True:
                    time.sleep(5.0)
                    if control and control.status_line:
                        buf_ms = self._buf.level_ms if self._buf else 0
                        print(f"{control.status_line}  [buf {buf_ms:.0f}ms]")
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

        # Interactive raw-terminal mode
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            last_display = 0.0
            while True:
                now = time.monotonic()
                # Refresh display ~5 times per second
                if now - last_display >= 0.2:
                    buf_ms = self._buf.level_ms if self._buf else 0
                    status = control.status_line or "Connecting to control port…"
                    line = f"\r{status}  [buf {buf_ms:.0f}ms]"
                    # Pad to clear previous longer lines
                    line = f"{line:<120}"
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    last_display = now

                # Poll stdin for keypresses (0.1 s timeout)
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                ch = sys.stdin.read(1)
                if ch in ('q', 'Q', '\x03'):   # q or Ctrl-C
                    break
                if ch == '\x1b':
                    # Escape sequence — read up to 2 more bytes non-blocking
                    seq = ch
                    for _ in range(2):
                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            seq += sys.stdin.read(1)
                        else:
                            break
                    if seq in ('\x1b[C', '\x1bOC'):
                        control.send_command({"cmd": "tune", "dir": "up"})
                    elif seq in ('\x1b[D', '\x1bOD'):
                        control.send_command({"cmd": "tune", "dir": "down"})
                    elif seq in ('\x1b[A', '\x1bOA'):
                        control.send_command({"cmd": "volume", "dir": "up"})
                    elif seq in ('\x1b[B', '\x1bOB'):
                        control.send_command({"cmd": "volume", "dir": "down"})
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\n")
            sys.stdout.flush()
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

        # Sequence-number gap detection
        if self._last_seq is not None:
            expected_seq = (self._last_seq + 1) & 0xFFFFFFFF
            if seq != expected_seq:
                gap = (seq - expected_seq) & 0xFFFFFFFF
                self._dropped += gap
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

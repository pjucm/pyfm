#!/usr/bin/env python3
"""
pjfm Web Application - FM Radio Receiver for SignalHound BB60D

A web-based interface for the pjfm FM radio application.
Uses Flask + Flask-SocketIO for real-time audio streaming and control.
"""

import sys
import os
import threading
import time
import json
import base64
import struct
import logging
import numpy as np

try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False
    print("Warning: opuslib not available, using uncompressed audio")

# Suppress HTTP access logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from bb60d import BB60D, get_api_version
from demodulator import FMStereoDecoder
from rds_decoder import RDSDecoder, pi_to_callsign

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pjfm-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


def dbm_to_s_meter(dbm):
    """Convert dBm to S-meter reading (VHF/UHF standard)."""
    S9_DBM = -93.0
    DB_PER_S = 6.0
    if dbm >= S9_DBM:
        db_over = dbm - S9_DBM
        return (9, db_over)
    else:
        s_units = 9 + (dbm - S9_DBM) / DB_PER_S
        s_units = max(0, min(9, s_units))
        return (s_units, 0)


def format_s_meter(dbm):
    """Format S-meter reading as string."""
    s_units, db_over = dbm_to_s_meter(dbm)
    if db_over > 0:
        return f"S9+{db_over:.0f}dB"
    elif s_units >= 1:
        return f"S{s_units:.0f}"
    else:
        return "S0"


class SpectrumAnalyzer:
    """Audio spectrum analyzer for web display."""

    NUM_BANDS = 16

    def __init__(self, sample_rate=48000, fft_size=2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

        # Logarithmic frequency bands from ~60Hz to ~16kHz
        min_freq = 60
        max_freq = 16000
        self.band_edges = np.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            self.NUM_BANDS + 1
        )

        freq_per_bin = sample_rate / fft_size
        self.band_bins = (self.band_edges / freq_per_bin).astype(int)
        self.band_bins = np.clip(self.band_bins, 0, fft_size // 2)

        self.levels = np.zeros(self.NUM_BANDS)
        self.peaks = np.zeros(self.NUM_BANDS)

        self.attack = 0.7
        self.decay = 0.15
        self.peak_decay = 0.02

        self.audio_buffer = np.zeros(fft_size)
        self.buffer_pos = 0
        self.window = np.hanning(fft_size)

    def update(self, audio_samples):
        """Update spectrum with new audio samples."""
        if audio_samples.ndim == 2:
            audio_samples = audio_samples.mean(axis=1)

        samples_to_add = len(audio_samples)
        if samples_to_add == 0:
            return

        if self.buffer_pos + samples_to_add <= self.fft_size:
            self.audio_buffer[self.buffer_pos:self.buffer_pos + samples_to_add] = audio_samples
            self.buffer_pos += samples_to_add
        else:
            if samples_to_add >= self.fft_size:
                self.audio_buffer[:] = audio_samples[-self.fft_size:]
                self.buffer_pos = self.fft_size
            else:
                shift = samples_to_add
                self.audio_buffer[:-shift] = self.audio_buffer[shift:]
                self.audio_buffer[-shift:] = audio_samples
                self.buffer_pos = self.fft_size

        if self.buffer_pos >= self.fft_size:
            self._compute_spectrum()
            self.buffer_pos = self.fft_size // 2

    def _compute_spectrum(self):
        """Compute FFT and update band levels."""
        windowed = self.audio_buffer * self.window
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result) / self.fft_size

        new_levels = np.zeros(self.NUM_BANDS)
        for i in range(self.NUM_BANDS):
            start_bin = self.band_bins[i]
            end_bin = self.band_bins[i + 1]
            if end_bin > start_bin:
                band_power = np.mean(magnitude[start_bin:end_bin] ** 2)
                if band_power > 1e-12:
                    db = 10 * np.log10(band_power)
                    new_levels[i] = np.clip((db + 70) / 60, 0, 1)

        for i in range(self.NUM_BANDS):
            if new_levels[i] > self.levels[i]:
                self.levels[i] += (new_levels[i] - self.levels[i]) * self.attack
            else:
                self.levels[i] += (new_levels[i] - self.levels[i]) * self.decay

            if self.levels[i] > self.peaks[i]:
                self.peaks[i] = self.levels[i]
            else:
                self.peaks[i] = max(0, self.peaks[i] - self.peak_decay)

    def reset(self):
        """Reset analyzer state."""
        self.levels = np.zeros(self.NUM_BANDS)
        self.peaks = np.zeros(self.NUM_BANDS)
        self.audio_buffer = np.zeros(self.fft_size)
        self.buffer_pos = 0


class WebRadio:
    """
    FM Radio controller for web interface.

    Streams audio to connected WebSocket clients.
    """

    IQ_SAMPLE_RATE = 250000
    AUDIO_SAMPLE_RATE = 48000
    IQ_BLOCK_SIZE = 8192

    def __init__(self, initial_freq=89.9e6):
        self.device = BB60D()
        self.device.frequency = initial_freq

        self.stereo_decoder = None
        self.rds_decoder = None
        self.rds_enabled = True
        self.rds_data = {}

        self.squelch_enabled = True
        self.squelch_threshold = -95.0

        self.running = False
        self.audio_thread = None
        self.error_message = None
        self.signal_dbm = -140.0

        self.tuning_lock = threading.Lock()
        self.is_tuning = False

        # Connected clients
        self.clients = set()
        self.clients_lock = threading.Lock()

        # Volume (applied server-side before streaming)
        self.volume = 1.0

        # Bass/treble boost
        self._bass_boost = True
        self._treble_boost = True

        # Spectrum analyzer
        self.spectrum_analyzer = SpectrumAnalyzer(
            sample_rate=self.AUDIO_SAMPLE_RATE,
            fft_size=2048
        )

        # Opus encoder for compressed audio streaming
        self.opus_encoder = None
        self.opus_frame_size = 960  # 20ms at 48kHz
        self.opus_buffer = np.array([], dtype=np.int16)

    def start(self):
        """Start the radio."""
        try:
            self.device.open()
            self.device.configure_iq_streaming(self.device.frequency, self.IQ_SAMPLE_RATE)

            actual_rate = self.device.iq_sample_rate
            self.stereo_decoder = FMStereoDecoder(
                iq_sample_rate=actual_rate,
                audio_sample_rate=self.AUDIO_SAMPLE_RATE,
                deviation=75000,
                deemphasis=75e-6
            )
            self.stereo_decoder.bass_boost_enabled = self._bass_boost
            self.stereo_decoder.treble_boost_enabled = self._treble_boost

            self.rds_decoder = RDSDecoder(sample_rate=actual_rate)

            # Initialize Opus encoder
            if OPUS_AVAILABLE:
                self.opus_encoder = opuslib.Encoder(
                    self.AUDIO_SAMPLE_RATE,
                    2,  # stereo
                    opuslib.APPLICATION_AUDIO
                )
                self.opus_encoder.bitrate = 128000  # 128 kbps
                self.opus_buffer = np.array([], dtype=np.int16)
                print("Opus encoder initialized at 128 kbps")
            else:
                self.opus_encoder = None

            self.running = True
            self._loop_counter = 0  # For periodic state updates
            self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self.audio_thread.start()

        except Exception as e:
            self.error_message = str(e)
            raise

    def stop(self):
        """Stop the radio."""
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        self.device.close()

    def _audio_loop(self):
        """Background thread for IQ capture, demodulation, and audio streaming."""
        while self.running:
            try:
                if self.is_tuning:
                    time.sleep(0.01)
                    continue

                with self.tuning_lock:
                    iq = self.device.fetch_iq(self.IQ_BLOCK_SIZE)

                if self.is_tuning:
                    continue

                # Measure signal power
                iq_subset = iq[::16]
                mean_power = np.mean(iq_subset.real**2 + iq_subset.imag**2)
                if mean_power > 0:
                    self.signal_dbm = 10 * np.log10(mean_power)
                else:
                    self.signal_dbm = -140.0

                squelched = self.squelch_enabled and self.signal_dbm < self.squelch_threshold

                # Demodulate FM
                audio = self.stereo_decoder.demodulate(iq)

                # Auto-enable RDS when pilot detected
                if self.stereo_decoder:
                    pilot_present = self.stereo_decoder.pilot_detected and self.signal_dbm >= self.squelch_threshold
                    if pilot_present and not self.rds_enabled:
                        self.rds_enabled = True
                        if self.rds_decoder:
                            self.rds_decoder.reset()

                # Update spectrum analyzer
                self.spectrum_analyzer.update(audio)

                # Process RDS
                if self.rds_enabled and self.rds_decoder and self.stereo_decoder.last_baseband is not None:
                    self.rds_data = self.rds_decoder.process(
                        self.stereo_decoder.last_baseband,
                        use_coherent=True
                    )

                # Apply squelch
                if squelched:
                    audio = np.zeros_like(audio)

                # Apply volume
                audio = audio * self.volume

                # Stream audio and state to connected clients
                self._stream_audio(audio)

                # Send state updates every 4 iterations (~10 Hz at 26ms per iteration)
                self._loop_counter += 1
                if self._loop_counter >= 4:
                    self._loop_counter = 0
                    self._broadcast_state()

            except Exception as e:
                if not self.is_tuning:
                    self.error_message = str(e)
                time.sleep(0.01)

    def _stream_audio(self, audio):
        """Stream audio to all connected WebSocket clients."""
        with self.clients_lock:
            if not self.clients:
                return

        # Convert to 16-bit PCM (interleaved stereo)
        audio_int16 = (audio * 32767).astype(np.int16)

        if self.opus_encoder is not None:
            # Buffer audio for Opus encoding (needs fixed frame sizes)
            self.opus_buffer = np.concatenate([self.opus_buffer, audio_int16.flatten()])

            # Encode complete frames (960 samples per channel = 1920 int16 values for stereo)
            frame_samples = self.opus_frame_size * 2  # stereo
            while len(self.opus_buffer) >= frame_samples:
                frame = self.opus_buffer[:frame_samples]
                self.opus_buffer = self.opus_buffer[frame_samples:]

                # Encode to Opus
                opus_data = self.opus_encoder.encode(frame.tobytes(), self.opus_frame_size)

                # Send as binary via base64 (for JSON compatibility)
                opus_b64 = base64.b64encode(opus_data).decode('ascii')
                socketio.emit('audio', {'data': opus_b64, 'codec': 'opus', 'frame_size': self.opus_frame_size})
        else:
            # Fallback: uncompressed PCM
            audio_bytes = audio_int16.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
            socketio.emit('audio', {'data': audio_b64, 'codec': 'pcm', 'channels': 2, 'sample_rate': self.AUDIO_SAMPLE_RATE})

    def _broadcast_state(self):
        """Broadcast radio state to all connected clients."""
        with self.clients_lock:
            if not self.clients:
                return
        try:
            state = self.get_state()
            socketio.emit('state', state)
        except Exception:
            pass

    def get_state(self):
        """Get current radio state as a dictionary."""
        rds_snapshot = dict(self.rds_data) if self.rds_data else {}
        pi_hex = rds_snapshot.get('pi_hex')
        callsign = pi_to_callsign(pi_hex) if pi_hex else None

        # Convert numpy types to native Python types for JSON serialization
        return {
            'frequency_mhz': float(self.device.frequency / 1e6),
            'signal_dbm': float(self.signal_dbm),
            's_meter': format_s_meter(self.signal_dbm),
            'squelch_enabled': bool(self.squelch_enabled),
            'squelch_threshold': float(self.squelch_threshold),
            'is_squelched': bool(self.squelch_enabled and self.signal_dbm < self.squelch_threshold),
            'pilot_detected': bool(self.stereo_decoder.pilot_detected) if self.stereo_decoder else False,
            'snr_db': float(self.stereo_decoder.snr_db) if self.stereo_decoder else 0.0,
            'stereo_blend': float(self.stereo_decoder.stereo_blend_factor) if self.stereo_decoder else 0.0,
            'rds_enabled': bool(self.rds_enabled),
            'rds': {
                'station_name': str(rds_snapshot.get('station_name', '') or ''),
                'program_type': str(rds_snapshot.get('program_type', '') or ''),
                'radio_text': str(rds_snapshot.get('radio_text', '') or ''),
                'pi_hex': str(pi_hex) if pi_hex else None,
                'callsign': str(callsign) if callsign else None,
                'synced': bool(rds_snapshot.get('synced', False)),
            },
            'volume': float(self.volume),
            'bass_boost': bool(self._bass_boost),
            'treble_boost': bool(self._treble_boost),
            'error': str(self.error_message) if self.error_message else None,
            'spectrum': {
                'levels': [float(x) for x in self.spectrum_analyzer.levels],
                'peaks': [float(x) for x in self.spectrum_analyzer.peaks],
            },
        }

    def tune_to(self, freq_hz):
        """Tune to a specific frequency."""
        if not (88.0e6 <= freq_hz <= 108.0e6):
            return False

        self.is_tuning = True
        self.error_message = None
        with self.tuning_lock:
            self.device.frequency = freq_hz
            self.device.configure_iq_streaming(freq_hz, self.IQ_SAMPLE_RATE)
            if self.stereo_decoder:
                self.stereo_decoder.reset()
            if self.rds_decoder:
                self.rds_decoder.reset()
                self.rds_data = {}
        self.is_tuning = False
        return True

    def tune_up(self):
        """Tune up by 100 kHz."""
        new_freq = min(108.0e6, self.device.frequency + 100e3)
        return self.tune_to(new_freq)

    def tune_down(self):
        """Tune down by 100 kHz."""
        new_freq = max(88.0e6, self.device.frequency - 100e3)
        return self.tune_to(new_freq)

    def set_volume(self, vol):
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, vol))

    def toggle_squelch(self):
        """Toggle squelch on/off."""
        self.squelch_enabled = not self.squelch_enabled

    def toggle_rds(self):
        """Toggle RDS on/off."""
        self.rds_enabled = not self.rds_enabled
        if self.rds_decoder:
            self.rds_decoder.reset()
        self.rds_data = {}

    def toggle_bass_boost(self):
        """Toggle bass boost."""
        self._bass_boost = not self._bass_boost
        if self.stereo_decoder:
            self.stereo_decoder.bass_boost_enabled = self._bass_boost
        print(f"Bass boost: {self._bass_boost}")

    def toggle_treble_boost(self):
        """Toggle treble boost."""
        self._treble_boost = not self._treble_boost
        if self.stereo_decoder:
            self.stereo_decoder.treble_boost_enabled = self._treble_boost
        print(f"Treble boost: {self._treble_boost}")

    def add_client(self, sid):
        """Register a client connection."""
        with self.clients_lock:
            self.clients.add(sid)

    def remove_client(self, sid):
        """Unregister a client connection."""
        with self.clients_lock:
            self.clients.discard(sid)


# Global radio instance
radio = None


def get_radio():
    """Get or create the radio instance."""
    global radio
    if radio is None:
        radio = WebRadio()
        radio.start()
    return radio


# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return app.send_static_file('favicon.ico')


@app.route('/api/state')
def api_state():
    """Get current radio state."""
    r = get_radio()
    return jsonify(r.get_state())


@app.route('/api/tune', methods=['POST'])
def api_tune():
    """Tune to a frequency."""
    r = get_radio()
    data = request.get_json()
    freq_mhz = data.get('frequency')
    if freq_mhz is not None:
        success = r.tune_to(freq_mhz * 1e6)
        return jsonify({'success': success, 'frequency_mhz': r.device.frequency / 1e6})
    return jsonify({'success': False, 'error': 'No frequency provided'}), 400


@app.route('/api/tune/up', methods=['POST'])
def api_tune_up():
    """Tune up by 100 kHz."""
    r = get_radio()
    r.tune_up()
    return jsonify({'frequency_mhz': r.device.frequency / 1e6})


@app.route('/api/tune/down', methods=['POST'])
def api_tune_down():
    """Tune down by 100 kHz."""
    r = get_radio()
    r.tune_down()
    return jsonify({'frequency_mhz': r.device.frequency / 1e6})


@app.route('/api/volume', methods=['POST'])
def api_volume():
    """Set volume."""
    r = get_radio()
    data = request.get_json()
    vol = data.get('volume')
    if vol is not None:
        r.set_volume(vol)
        return jsonify({'volume': r.volume})
    return jsonify({'error': 'No volume provided'}), 400


@app.route('/api/squelch/toggle', methods=['POST'])
def api_squelch_toggle():
    """Toggle squelch."""
    r = get_radio()
    r.toggle_squelch()
    return jsonify({'squelch_enabled': r.squelch_enabled})


@app.route('/api/rds/toggle', methods=['POST'])
def api_rds_toggle():
    """Toggle RDS."""
    r = get_radio()
    r.toggle_rds()
    return jsonify({'rds_enabled': r.rds_enabled})


@app.route('/api/bass/toggle', methods=['POST'])
def api_bass_toggle():
    """Toggle bass boost."""
    r = get_radio()
    r.toggle_bass_boost()
    return jsonify({'bass_boost': r._bass_boost})


@app.route('/api/treble/toggle', methods=['POST'])
def api_treble_toggle():
    """Toggle treble boost."""
    r = get_radio()
    r.toggle_treble_boost()
    return jsonify({'treble_boost': r._treble_boost})


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    r = get_radio()
    r.add_client(request.sid)
    emit('state', r.get_state())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    global radio
    if radio:
        radio.remove_client(request.sid)


@socketio.on('tune')
def handle_tune(data):
    """Handle tune command via WebSocket."""
    r = get_radio()
    freq_mhz = data.get('frequency')
    if freq_mhz:
        r.tune_to(freq_mhz * 1e6)


@socketio.on('tune_up')
def handle_tune_up():
    """Handle tune up command."""
    r = get_radio()
    r.tune_up()


@socketio.on('tune_down')
def handle_tune_down():
    """Handle tune down command."""
    r = get_radio()
    r.tune_down()


@socketio.on('volume')
def handle_volume(data):
    """Handle volume change."""
    r = get_radio()
    vol = data.get('volume')
    if vol is not None:
        r.set_volume(vol)


def main():
    """Run the web application."""
    import argparse

    parser = argparse.ArgumentParser(description="pjfm Web Interface")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"Starting pjfm web interface on http://{args.host}:{args.port}")
    print(f"BB API Version: {get_api_version()}")

    try:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        global radio
        if radio:
            radio.stop()


if __name__ == '__main__':
    main()

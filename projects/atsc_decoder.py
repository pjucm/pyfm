#!/usr/bin/env python3
"""
ATSC 8VSB Digital TV Decoder

Receives IQ samples from the BB60D, demodulates 8VSB, and pipes MPEG-2
transport stream to mpv for live playback.

Technical Background:
- ATSC 8VSB: 6 MHz channel, 10.762238 Msps symbol rate, 8-level VSB modulation
- Error Correction: Rate-2/3 trellis code + RS(207,187) outer code
- Output: 19.39 Mbps MPEG-2 transport stream (188-byte packets)
- BB60D Sample Rate: 20 MHz (40 MHz / 2) provides ~1.86 samples/symbol

Usage:
    python atsc_decoder.py --channel 24
    python atsc_decoder.py --freq 533
"""

import sys
import argparse
import time
import subprocess
import threading
import numpy as np
from scipy import signal
from collections import deque

# Try to import numba for JIT acceleration of Viterbi decoder
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available, Viterbi decoder will be slow")

# Try to import reedsolo for Reed-Solomon decoding
try:
    from reedsolo import RSCodec, ReedSolomonError
    REEDSOLO_AVAILABLE = True
except ImportError:
    REEDSOLO_AVAILABLE = False
    print("Warning: reedsolo not available, RS decoding disabled")


# ATSC Constants
ATSC_SYMBOL_RATE = 10.762238e6  # Symbols per second
ATSC_CHANNEL_BW = 6e6  # 6 MHz channel bandwidth
ATSC_PILOT_OFFSET = 310e3  # Pilot at 310 kHz above lower band edge
ATSC_SAMPLE_RATE = 20e6  # BB60D sample rate (40 MHz / 2)
SAMPLES_PER_SYMBOL = ATSC_SAMPLE_RATE / ATSC_SYMBOL_RATE  # ~1.858

# Segment and field sync constants
SEGMENT_SYMBOLS = 832  # Symbols per data segment (including sync)
SEGMENTS_PER_FIELD = 313  # Data segments per field
SEGMENT_SYNC = np.array([5, -5, -5, 5], dtype=np.float64)  # 4-symbol sync pattern
DATA_SYMBOLS_PER_SEGMENT = 828  # Data symbols after 4-symbol sync

# Trellis encoder constants
TRELLIS_STATES = 8  # 8 states for rate-2/3 trellis
TRELLIS_INPUTS = 4  # 2 bits = 4 possible inputs

# Reed-Solomon constants
RS_BLOCK_SIZE = 207  # Total bytes (data + parity)
RS_DATA_SIZE = 187  # Data bytes
RS_PARITY_SIZE = 20  # Parity bytes (207 - 187)

# MPEG-2 Transport Stream constants
TS_PACKET_SIZE = 188  # Bytes per packet
TS_SYNC_BYTE = 0x47  # Sync byte value
TS_PACKETS_PER_SEGMENT = 1  # One packet per RS block


def channel_to_freq(channel):
    """
    Convert ATSC RF channel number to center frequency in Hz.

    Channel allocations (post-2020 repack):
    - VHF-Lo: Channels 2-6 (54-88 MHz, with gap at 72-76 MHz)
    - VHF-Hi: Channels 7-13 (174-216 MHz)
    - UHF: Channels 14-36 (470-608 MHz)
    """
    if 2 <= channel <= 4:
        # VHF-Lo low: 54-72 MHz
        return (channel - 2) * 6e6 + 57e6  # Center of first channel
    elif 5 <= channel <= 6:
        # VHF-Lo high: 76-88 MHz (gap 72-76)
        return (channel - 5) * 6e6 + 79e6
    elif 7 <= channel <= 13:
        # VHF-Hi: 174-216 MHz
        return (channel - 7) * 6e6 + 177e6
    elif 14 <= channel <= 36:
        # UHF: 470-608 MHz (post-2020 repack)
        return (channel - 14) * 6e6 + 473e6
    else:
        raise ValueError(f"Invalid ATSC channel: {channel}")


class PilotPLL:
    """
    Phase-Locked Loop for ATSC 8VSB pilot tracking.

    Tracks the 310 kHz pilot tone (small carrier added by ATSC)
    and generates a coherent carrier for VSB demodulation.

    Based on PilotPLL pattern from demodulator.py.
    """

    def __init__(self, sample_rate, center_freq=310e3, bandwidth=500, damping=0.707):
        """
        Initialize pilot PLL.

        Args:
            sample_rate: Input sample rate in Hz
            center_freq: Pilot frequency in Hz (310 kHz for ATSC)
            bandwidth: Loop bandwidth in Hz
            damping: Damping factor (0.707 = critical damping)
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.damping = damping

        # Loop filter coefficients (2nd-order Type 2)
        omega_n = 2 * np.pi * bandwidth
        self.Kp = 2 * damping * omega_n
        self.Ki = omega_n ** 2

        # State variables
        self.phase = 0.0
        self.freq_offset = 0.0
        self.integrator = 0.0
        self.locked = False
        self.lock_count = 0

        # Precompute constants
        self._dt = 1.0 / sample_rate
        self._omega_0 = 2 * np.pi * center_freq

    def process(self, vsb_signal):
        """
        Process VSB signal through PLL to extract pilot and generate carrier.

        Args:
            vsb_signal: Real-valued VSB samples (numpy array)

        Returns:
            tuple: (carrier, locked)
                - carrier: Coherent carrier for demodulation
                - locked: True if PLL is locked to pilot
        """
        n = len(vsb_signal)
        carrier = np.zeros(n, dtype=np.float64)

        # Local copies for speed
        phase = self.phase
        integrator = self.integrator
        freq_offset = self.freq_offset
        dt = self._dt
        omega_0 = self._omega_0
        Kp = self.Kp
        Ki = self.Ki

        for i in range(n):
            # NCO output
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            carrier[i] = cos_phase

            # Phase detector: multiply input by quadrature
            phase_error = vsb_signal[i] * sin_phase

            # Loop filter (PI controller)
            integrator += phase_error * Ki * dt
            freq_offset = Kp * phase_error + integrator

            # Update NCO phase
            phase += (omega_0 + freq_offset) * dt

            # Wrap phase
            while phase > np.pi:
                phase -= 2 * np.pi
            while phase < -np.pi:
                phase += 2 * np.pi

        # Store state
        self.phase = phase
        self.integrator = integrator
        self.freq_offset = freq_offset

        # Lock detection (within 100 Hz of center)
        if abs(freq_offset) < 2 * np.pi * 100:
            self.lock_count = min(100, self.lock_count + 1)
        else:
            self.lock_count = max(0, self.lock_count - 5)
        self.locked = self.lock_count >= 50

        return carrier, self.locked

    def reset(self):
        """Reset PLL state."""
        self.phase = 0.0
        self.freq_offset = 0.0
        self.integrator = 0.0
        self.locked = False
        self.lock_count = 0


class VSBFrontEnd:
    """
    VSB Front-end: Channel filtering and carrier recovery.

    Applies:
    1. Channel bandpass filter (6 MHz)
    2. Pilot PLL for carrier recovery
    3. Coherent demodulation to baseband
    """

    def __init__(self, sample_rate=ATSC_SAMPLE_RATE):
        self.sample_rate = sample_rate
        nyq = sample_rate / 2

        # Design channel filter (6 MHz bandwidth, root raised cosine approximation)
        # Use a wideband lowpass after mixing to baseband
        channel_cutoff = 3.5e6 / nyq  # Slightly wider than 3 MHz to capture all symbols
        self.channel_b, self.channel_a = signal.butter(6, channel_cutoff, btype='low')
        self.channel_zi = signal.lfilter_zi(self.channel_b, self.channel_a) * 0

        # Design matched filter (root raised cosine, alpha=0.1152)
        # For 8VSB, the excess bandwidth is 11.52%
        self.alpha = 0.1152
        self._design_rrc_filter()

        # Pilot PLL for carrier recovery
        self.pilot_pll = PilotPLL(
            sample_rate=sample_rate,
            center_freq=ATSC_PILOT_OFFSET,
            bandwidth=500,
            damping=0.707
        )

        # AGC state
        self.agc_gain = 1.0
        self.agc_target = 1.0
        self.agc_alpha = 0.001  # Slow AGC time constant

    def _design_rrc_filter(self):
        """Design root raised cosine filter for ISI-free detection."""
        # Filter spans +/- 6 symbols
        span_symbols = 6
        span_samples = int(span_symbols * SAMPLES_PER_SYMBOL * 2)
        if span_samples % 2 == 0:
            span_samples += 1

        # Time axis in symbol periods
        t = np.linspace(-span_symbols, span_symbols, span_samples)

        # Root raised cosine formula
        alpha = self.alpha
        h = np.zeros(span_samples)
        for i, ti in enumerate(t):
            if abs(ti) < 1e-10:
                # t = 0
                h[i] = 1 - alpha + 4 * alpha / np.pi
            elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-10:
                # t = +/- T/(4*alpha)
                h[i] = alpha / np.sqrt(2) * ((1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                                              (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
            else:
                num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
                h[i] = num / den

        # Normalize
        h = h / np.sum(h)
        self.rrc_filter = h
        self.rrc_zi = signal.lfilter_zi(self.rrc_filter, 1.0) * 0

    def process(self, iq_samples):
        """
        Process IQ samples through front-end.

        Args:
            iq_samples: Complex IQ samples at sample_rate

        Returns:
            tuple: (baseband, carrier_locked)
                - baseband: Real-valued VSB baseband at sample_rate
                - carrier_locked: True if pilot PLL is locked
        """
        if len(iq_samples) == 0:
            return np.array([]), False

        # Convert to real VSB signal (take real part after any frequency correction)
        # The pilot is at 310 kHz from lower band edge
        vsb_real = np.real(iq_samples)

        # Run pilot PLL on real signal
        carrier, locked = self.pilot_pll.process(vsb_real)

        # Coherent demodulation: multiply by carrier
        demod = vsb_real * carrier * 2  # *2 to recover amplitude

        # Channel lowpass filter
        baseband, self.channel_zi = signal.lfilter(
            self.channel_b, self.channel_a, demod, zi=self.channel_zi
        )

        # Apply matched filter (RRC)
        baseband, self.rrc_zi = signal.lfilter(
            self.rrc_filter, 1.0, baseband, zi=self.rrc_zi
        )

        # AGC
        rms = np.sqrt(np.mean(baseband ** 2))
        if rms > 1e-10:
            target_gain = self.agc_target / rms
            self.agc_gain = (1 - self.agc_alpha) * self.agc_gain + self.agc_alpha * target_gain
        baseband = baseband * self.agc_gain

        return baseband, locked

    def reset(self):
        """Reset front-end state."""
        self.channel_zi = signal.lfilter_zi(self.channel_b, self.channel_a) * 0
        self.rrc_zi = signal.lfilter_zi(self.rrc_filter, 1.0) * 0
        self.pilot_pll.reset()
        self.agc_gain = 1.0


class VSBTimingRecovery:
    """
    Symbol timing recovery for 8VSB using Gardner TED with PI loop.

    Based on RDS decoder timing recovery pattern but adapted for
    ATSC symbol rate (~1.86 samples/symbol at 20 MHz).
    """

    def __init__(self, sample_rate=ATSC_SAMPLE_RATE, symbol_rate=ATSC_SYMBOL_RATE):
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate  # ~1.858

        # Timing recovery state (Gardner TED with PI loop)
        self.timing_mu = 0.5  # Fractional sample offset
        self.timing_freq = 0.0  # Fractional accumulator
        self.timing_freq_offset = 0.0  # Symbol rate offset

        # Loop gains (tuned for ATSC timing)
        self.timing_gain_p = 0.01  # Proportional gain
        self.timing_gain_i = 0.001  # Integral gain

        # Sample buffer
        self.sample_buffer = []

        # Statistics
        self.timing_error_avg = 0.0

    def process(self, baseband):
        """
        Extract symbols from baseband using timing recovery.

        Args:
            baseband: Filtered baseband samples

        Returns:
            numpy array of symbols (8-level values)
        """
        self.sample_buffer.extend(baseband.tolist())
        symbols = []

        sps = self.samples_per_symbol
        sps_int = int(sps)
        sps_frac = sps - sps_int
        half_sps = sps / 2.0

        # Need enough samples for Gardner TED
        while len(self.sample_buffer) >= sps_int * 2 + 2:
            mu = self.timing_mu
            mu = max(0, min(1, mu))

            # Previous sample (symbol n-1)
            prev_idx = 0
            prev_sample = self.sample_buffer[prev_idx]

            # Midpoint between symbols
            mid_pos = half_sps
            mid_idx = int(mid_pos)
            mid_frac = mid_pos - mid_idx
            if mid_idx + 1 >= len(self.sample_buffer):
                break
            mid_sample = (1 - mid_frac) * self.sample_buffer[mid_idx] + \
                        mid_frac * self.sample_buffer[mid_idx + 1]

            # Current symbol (with interpolation)
            curr_idx = sps_int
            if curr_idx + 1 >= len(self.sample_buffer):
                break
            curr_sample = (1 - mu) * self.sample_buffer[curr_idx] + \
                         mu * self.sample_buffer[curr_idx + 1]

            symbols.append(curr_sample)

            # Gardner TED for timing adjustment
            timing_error = -(curr_sample - prev_sample) * mid_sample
            norm = abs(curr_sample) + abs(prev_sample) + abs(mid_sample) + 1e-10
            timing_error = timing_error / norm
            timing_error = max(-0.3, min(0.3, timing_error))

            # PI loop
            self.timing_mu += self.timing_gain_p * timing_error
            self.timing_freq_offset += self.timing_gain_i * timing_error
            self.timing_freq_offset = max(-0.5, min(0.5, self.timing_freq_offset))

            # Update timing error average
            self.timing_error_avg = 0.99 * self.timing_error_avg + 0.01 * abs(timing_error)

            # Advance buffer with fractional tracking
            effective_sps_frac = sps_frac + self.timing_freq_offset
            self.timing_freq += effective_sps_frac
            advance = sps_int
            while self.timing_freq >= 1.0:
                advance += 1
                self.timing_freq -= 1.0
            while self.timing_freq < 0.0:
                advance -= 1
                self.timing_freq += 1.0

            # Handle mu wrapping
            while self.timing_mu >= 1.0:
                self.timing_mu -= 1.0
                advance += 1
            while self.timing_mu < 0.0:
                self.timing_mu += 1.0
                advance -= 1

            advance = max(1, advance)
            self.sample_buffer = self.sample_buffer[advance:]

        return np.array(symbols, dtype=np.float64)

    def reset(self):
        """Reset timing recovery state."""
        self.timing_mu = 0.5
        self.timing_freq = 0.0
        self.timing_freq_offset = 0.0
        self.sample_buffer = []
        self.timing_error_avg = 0.0


class SyncDetector:
    """
    ATSC segment and field sync detection.

    Detects:
    - Segment sync: 4-symbol pattern (+5, -5, -5, +5) every 832 symbols
    - Field sync: PN sequence every 313 segments
    """

    def __init__(self):
        # Sync pattern (normalized)
        self.segment_sync = SEGMENT_SYNC / np.linalg.norm(SEGMENT_SYNC)

        # Symbol buffer
        self.symbol_buffer = deque(maxlen=SEGMENT_SYMBOLS * 2)

        # Sync state
        self.synced = False
        self.sync_position = 0  # Position within segment
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0

        # Field sync detection
        self.segment_count = 0
        self.field_synced = False

        # PN sequences for field sync (simplified - just looking for correlation)
        self._init_pn_sequences()

    def _init_pn_sequences(self):
        """Initialize PN sequences used in field sync."""
        # Middle 511 symbols of field sync contain a PN sequence
        # Using a simplified detection based on energy
        self.field_sync_threshold = 0.8

    def process(self, symbols):
        """
        Process symbols and detect sync.

        Args:
            symbols: Array of symbol values

        Returns:
            list of (segment_data, is_field_sync) tuples
        """
        segments = []

        for sym in symbols:
            self.symbol_buffer.append(sym)

            if not self.synced:
                # Search for segment sync
                if len(self.symbol_buffer) >= len(self.segment_sync):
                    self._search_sync()
            else:
                # Track position within segment
                self.sync_position += 1

                if self.sync_position >= SEGMENT_SYMBOLS:
                    # End of segment - verify sync and extract data
                    segment = self._extract_segment()
                    if segment is not None:
                        is_field_sync = self._check_field_sync(segment)
                        segments.append((segment, is_field_sync))

                        if is_field_sync:
                            self.segment_count = 0
                            self.field_synced = True
                        else:
                            self.segment_count += 1

                    self.sync_position = 0

        return segments

    def _search_sync(self):
        """Search for segment sync pattern."""
        if len(self.symbol_buffer) < SEGMENT_SYMBOLS:
            return

        # Get last few symbols for correlation
        recent = np.array(list(self.symbol_buffer)[-len(self.segment_sync):])
        recent_norm = recent / (np.linalg.norm(recent) + 1e-10)

        # Correlate with sync pattern
        corr = abs(np.dot(recent_norm, self.segment_sync))

        if corr > 0.8:  # High correlation threshold
            self.synced = True
            self.sync_position = len(self.segment_sync)
            self.sync_confidence = 1
            self.consecutive_good = 1
            self.consecutive_bad = 0

    def _extract_segment(self):
        """Extract segment data and verify sync."""
        if len(self.symbol_buffer) < SEGMENT_SYMBOLS:
            return None

        # Get segment from buffer
        segment = np.array(list(self.symbol_buffer)[-SEGMENT_SYMBOLS:])

        # Check sync at start of segment
        sync_part = segment[:4]
        sync_norm = sync_part / (np.linalg.norm(sync_part) + 1e-10)
        corr = abs(np.dot(sync_norm, self.segment_sync))

        if corr > 0.7:
            self.consecutive_good += 1
            self.consecutive_bad = 0
            self.sync_confidence = min(100, self.sync_confidence + 1)
        else:
            self.consecutive_bad += 1
            self.consecutive_good = 0
            self.sync_confidence = max(0, self.sync_confidence - 5)

            if self.consecutive_bad >= 10:
                # Lost sync
                self.synced = False
                self.field_synced = False
                return None

        # Return data symbols (excluding 4-symbol sync)
        return segment[4:]

    def _check_field_sync(self, segment_data):
        """Check if this segment is a field sync."""
        # Field sync has distinctive pattern with high energy
        # Simplified detection based on amplitude variance
        if len(segment_data) < 100:
            return False

        # Field sync symbols have higher amplitude than data
        amp_variance = np.var(segment_data[:511])
        return amp_variance > 30  # Threshold for field sync detection

    def reset(self):
        """Reset sync detector state."""
        self.symbol_buffer.clear()
        self.synced = False
        self.sync_position = 0
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0
        self.segment_count = 0
        self.field_synced = False


# Trellis state transition tables (computed at module load)
# State: 3-bit register (8 states)
# Input: 2 bits (4 possibilities)
# Output: 3 bits (8 levels mapped to -7, -5, -3, -1, 1, 3, 5, 7)

def _build_trellis_tables():
    """Build trellis encoder state transition and output tables."""
    # ATSC trellis encoder is a rate-2/3 convolutional encoder
    # 2 input bits produce 3 output bits (one uncoded, two coded)

    next_state = np.zeros((8, 4), dtype=np.int32)
    output = np.zeros((8, 4), dtype=np.int32)

    for state in range(8):
        for inp in range(4):
            # Input bits: X2 (MSB, uncoded), X1 (coded)
            x2 = (inp >> 1) & 1  # Uncoded bit (passes through)
            x1 = inp & 1  # Coded bit

            # State bits: S2, S1, S0
            s2 = (state >> 2) & 1
            s1 = (state >> 1) & 1
            s0 = state & 1

            # Next state: shift register with feedback
            # Z0 = X1 XOR S0
            z0 = x1 ^ s0
            # Z1 = X1 XOR S1
            z1 = x1 ^ s1
            # Z2 = X2 (uncoded, just passes through)
            z2 = x2

            # Next state: shift in X1
            next_s = ((s1 << 2) | (s0 << 1) | x1) & 0x7
            next_state[state, inp] = next_s

            # Output symbol: 3-bit value Z2 Z1 Z0 mapped to 8-level
            out_bits = (z2 << 2) | (z1 << 1) | z0
            output[state, inp] = out_bits

    return next_state, output

TRELLIS_NEXT_STATE, TRELLIS_OUTPUT = _build_trellis_tables()

# 8-level symbol mapping (3-bit to amplitude)
SYMBOL_LEVELS = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def viterbi_decode_numba(symbols, next_state, output, symbol_levels):
        """
        Numba-accelerated Viterbi decoder for ATSC trellis code.

        Args:
            symbols: Received symbol values (soft decisions)
            next_state: State transition table [state, input] -> next_state
            output: Output table [state, input] -> output_level_index
            symbol_levels: 8-level amplitude values

        Returns:
            Decoded 2-bit values
        """
        n_symbols = len(symbols)
        n_states = 8
        n_inputs = 4

        # Path metrics (cumulative distance)
        path_metric = np.full(n_states, np.inf)
        path_metric[0] = 0.0  # Start in state 0

        # Survivor paths (store previous state for each state at each step)
        survivor = np.zeros((n_symbols, n_states), dtype=np.int32)

        # Decoded inputs along path
        decoded_input = np.zeros((n_symbols, n_states), dtype=np.int32)

        for i in range(n_symbols):
            new_metric = np.full(n_states, np.inf)
            sym = symbols[i]

            for state in range(n_states):
                if path_metric[state] == np.inf:
                    continue

                for inp in range(n_inputs):
                    ns = next_state[state, inp]
                    out_idx = output[state, inp]
                    expected = symbol_levels[out_idx]

                    # Branch metric (squared error)
                    branch = (sym - expected) ** 2
                    total = path_metric[state] + branch

                    if total < new_metric[ns]:
                        new_metric[ns] = total
                        survivor[i, ns] = state
                        decoded_input[i, ns] = inp

            path_metric = new_metric

        # Traceback from best final state
        final_state = np.argmin(path_metric)
        decoded = np.zeros(n_symbols, dtype=np.int32)

        state = final_state
        for i in range(n_symbols - 1, -1, -1):
            decoded[i] = decoded_input[i, state]
            state = survivor[i, state]

        return decoded


def viterbi_decode_python(symbols, next_state, output, symbol_levels):
    """
    Pure Python Viterbi decoder (slow, for testing without numba).
    """
    n_symbols = len(symbols)
    n_states = 8

    path_metric = np.full(n_states, np.inf)
    path_metric[0] = 0.0

    survivor = np.zeros((n_symbols, n_states), dtype=np.int32)
    decoded_input = np.zeros((n_symbols, n_states), dtype=np.int32)

    for i in range(n_symbols):
        new_metric = np.full(n_states, np.inf)
        sym = symbols[i]

        for state in range(n_states):
            if path_metric[state] == np.inf:
                continue

            for inp in range(4):
                ns = next_state[state, inp]
                out_idx = output[state, inp]
                expected = symbol_levels[out_idx]

                branch = (sym - expected) ** 2
                total = path_metric[state] + branch

                if total < new_metric[ns]:
                    new_metric[ns] = total
                    survivor[i, ns] = state
                    decoded_input[i, ns] = inp

        path_metric = new_metric

    final_state = np.argmin(path_metric)
    decoded = np.zeros(n_symbols, dtype=np.int32)

    state = final_state
    for i in range(n_symbols - 1, -1, -1):
        decoded[i] = decoded_input[i, state]
        state = survivor[i, state]

    return decoded


class TrellisDecoder:
    """
    Trellis decoder for ATSC 8VSB.

    Decodes the rate-2/3 trellis-coded 8-level symbols back to
    2-bit values using Viterbi algorithm.
    """

    def __init__(self):
        self.next_state = TRELLIS_NEXT_STATE
        self.output = TRELLIS_OUTPUT
        self.symbol_levels = SYMBOL_LEVELS

        # Use numba version if available
        if NUMBA_AVAILABLE:
            self._decode_func = viterbi_decode_numba
            # Warm up JIT compilation
            test_symbols = np.zeros(100, dtype=np.float64)
            self._decode_func(test_symbols, self.next_state, self.output, self.symbol_levels)
        else:
            self._decode_func = viterbi_decode_python

    def decode(self, symbols):
        """
        Decode trellis-coded symbols.

        Args:
            symbols: Array of 8-level symbol values

        Returns:
            Array of decoded 2-bit values
        """
        if len(symbols) == 0:
            return np.array([], dtype=np.uint8)

        decoded = self._decode_func(
            symbols.astype(np.float64),
            self.next_state,
            self.output,
            self.symbol_levels
        )

        return decoded.astype(np.uint8)


class ConvolutionalDeinterleaver:
    """
    Convolutional deinterleaver for ATSC.

    The ATSC interleaver uses a 52-branch convolutional structure:
    - 52 branches with delays 0, 4, 8, ... 204 bytes
    - Each branch handles one byte position in the RS block

    This deinterleaver reverses that process.
    """

    def __init__(self):
        self.n_branches = 52
        self.m = 4  # Delay increment in bytes

        # Initialize delay buffers for each branch
        # Branch i has delay (51 - i) * 4 bytes (reverse of interleaver)
        self.buffers = []
        for i in range(self.n_branches):
            delay = (self.n_branches - 1 - i) * self.m
            self.buffers.append(deque([0] * delay, maxlen=max(1, delay)))

        self.branch_index = 0

    def process(self, data_bytes):
        """
        Deinterleave data bytes.

        Args:
            data_bytes: Array of bytes to deinterleave

        Returns:
            Array of deinterleaved bytes
        """
        output = []

        for byte in data_bytes:
            branch = self.branch_index

            if len(self.buffers[branch]) > 0:
                # Get delayed output
                out_byte = self.buffers[branch][0]
                self.buffers[branch].append(byte)
            else:
                # No delay on this branch
                out_byte = byte

            output.append(out_byte)
            self.branch_index = (self.branch_index + 1) % self.n_branches

        return np.array(output, dtype=np.uint8)

    def reset(self):
        """Reset deinterleaver state."""
        self.buffers = []
        for i in range(self.n_branches):
            delay = (self.n_branches - 1 - i) * self.m
            self.buffers.append(deque([0] * delay, maxlen=max(1, delay)))
        self.branch_index = 0


class ReedSolomonDecoder:
    """
    Reed-Solomon decoder for ATSC RS(207, 187) code.

    Corrects up to 10 byte errors per block (207-187)/2 = 10.
    """

    def __init__(self):
        if REEDSOLO_AVAILABLE:
            # RS(207, 187) with 20 parity bytes
            self.rs = RSCodec(RS_PARITY_SIZE)
        else:
            self.rs = None

        # Statistics
        self.blocks_decoded = 0
        self.blocks_corrected = 0
        self.blocks_failed = 0

    def decode(self, block):
        """
        Decode RS(207, 187) block.

        Args:
            block: 207-byte block (187 data + 20 parity)

        Returns:
            tuple: (data_bytes, corrected, failed)
                - data_bytes: 187-byte decoded data (or None if failed)
                - corrected: Number of bytes corrected
                - failed: True if decoding failed
        """
        if not REEDSOLO_AVAILABLE or self.rs is None:
            # Without RS decoding, just return data portion
            return block[:RS_DATA_SIZE], 0, False

        self.blocks_decoded += 1

        try:
            # reedsolo expects bytes
            if isinstance(block, np.ndarray):
                block = bytes(block)

            decoded, _, errata_pos = self.rs.decode(block)
            corrected = len(errata_pos) if errata_pos else 0

            if corrected > 0:
                self.blocks_corrected += 1

            return np.frombuffer(decoded, dtype=np.uint8), corrected, False

        except ReedSolomonError:
            self.blocks_failed += 1
            return None, 0, True

    def reset(self):
        """Reset statistics."""
        self.blocks_decoded = 0
        self.blocks_corrected = 0
        self.blocks_failed = 0


class TSOutput:
    """
    MPEG-2 Transport Stream output and mpv pipe.

    Assembles decoded bytes into TS packets and pipes to mpv.
    """

    def __init__(self):
        self.mpv_process = None
        self.running = False

        # Packet assembly buffer
        self.packet_buffer = bytearray()

        # Statistics
        self.packets_sent = 0
        self.bytes_sent = 0

    def start(self):
        """Start mpv process for video playback."""
        if self.running:
            return

        try:
            # Start mpv with stdin as input
            self.mpv_process = subprocess.Popen(
                [
                    'mpv',
                    '--demuxer=lavf',
                    '--demuxer-lavf-format=mpegts',
                    '--no-terminal',
                    '--force-window=yes',
                    '--title=ATSC Digital TV',
                    '-'  # Read from stdin
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.running = True
            print("mpv started for video playback")
        except FileNotFoundError:
            print("Error: mpv not found. Install mpv for video playback.")
            self.mpv_process = None
        except Exception as e:
            print(f"Error starting mpv: {e}")
            self.mpv_process = None

    def stop(self):
        """Stop mpv process."""
        self.running = False
        if self.mpv_process:
            try:
                self.mpv_process.stdin.close()
                self.mpv_process.wait(timeout=2)
            except:
                self.mpv_process.kill()
            self.mpv_process = None

    def write(self, data_bytes):
        """
        Write decoded data to output.

        Args:
            data_bytes: Decoded bytes (should be 187-byte TS payload)
        """
        if data_bytes is None:
            return

        # Add to buffer
        self.packet_buffer.extend(data_bytes)

        # Extract complete TS packets
        while len(self.packet_buffer) >= TS_PACKET_SIZE:
            # Look for sync byte
            try:
                sync_pos = self.packet_buffer.index(TS_SYNC_BYTE)
            except ValueError:
                # No sync byte found, discard buffer
                self.packet_buffer.clear()
                break

            if sync_pos > 0:
                # Discard bytes before sync
                del self.packet_buffer[:sync_pos]

            if len(self.packet_buffer) >= TS_PACKET_SIZE:
                packet = bytes(self.packet_buffer[:TS_PACKET_SIZE])
                del self.packet_buffer[:TS_PACKET_SIZE]

                # Verify sync byte
                if packet[0] == TS_SYNC_BYTE:
                    self._send_packet(packet)

    def _send_packet(self, packet):
        """Send TS packet to mpv."""
        if self.mpv_process and self.mpv_process.stdin:
            try:
                self.mpv_process.stdin.write(packet)
                self.mpv_process.stdin.flush()
                self.packets_sent += 1
                self.bytes_sent += len(packet)
            except (BrokenPipeError, OSError):
                # mpv closed
                self.running = False

    def reset(self):
        """Reset output state."""
        self.packet_buffer.clear()
        self.packets_sent = 0
        self.bytes_sent = 0


class ATSCDecoder:
    """
    Main ATSC 8VSB decoder class.

    Coordinates all decoding stages from IQ samples to MPEG-2 TS output.
    """

    def __init__(self, sample_rate=ATSC_SAMPLE_RATE):
        self.sample_rate = sample_rate

        # Initialize processing stages
        self.front_end = VSBFrontEnd(sample_rate)
        self.timing_recovery = VSBTimingRecovery(sample_rate)
        self.sync_detector = SyncDetector()
        self.trellis_decoder = TrellisDecoder()
        self.deinterleaver = ConvolutionalDeinterleaver()
        self.rs_decoder = ReedSolomonDecoder()
        self.ts_output = TSOutput()

        # State
        self.carrier_locked = False
        self.sync_locked = False
        self.running = False

        # Statistics
        self.segments_decoded = 0
        self.rs_errors = 0

        # Byte accumulator for RS blocks
        self.byte_buffer = bytearray()

    def start(self):
        """Start decoder and video output."""
        self.running = True
        self.ts_output.start()

    def stop(self):
        """Stop decoder and video output."""
        self.running = False
        self.ts_output.stop()

    def process(self, iq_samples):
        """
        Process IQ samples through full decoding chain.

        Args:
            iq_samples: Complex IQ samples at sample_rate

        Returns:
            dict with status information
        """
        if len(iq_samples) == 0:
            return self._get_status()

        # Stage 1: Front-end (filtering + carrier recovery)
        baseband, self.carrier_locked = self.front_end.process(iq_samples)

        if not self.carrier_locked:
            return self._get_status()

        # Stage 2: Symbol timing recovery
        symbols = self.timing_recovery.process(baseband)

        if len(symbols) == 0:
            return self._get_status()

        # Stage 3: Sync detection and segment extraction
        segments = self.sync_detector.process(symbols)
        self.sync_locked = self.sync_detector.synced

        for segment_data, is_field_sync in segments:
            if is_field_sync:
                # Field sync segment - don't decode as data
                continue

            # Stage 4: Trellis decoding
            decoded_dibits = self.trellis_decoder.decode(segment_data)

            # Convert 2-bit values to bytes (4 dibits = 1 byte)
            bytes_out = self._dibits_to_bytes(decoded_dibits)

            # Stage 5: Deinterleaving
            deinterleaved = self.deinterleaver.process(bytes_out)

            # Accumulate for RS block
            self.byte_buffer.extend(deinterleaved)

            # Stage 6: RS decoding when we have a full block
            while len(self.byte_buffer) >= RS_BLOCK_SIZE:
                rs_block = bytes(self.byte_buffer[:RS_BLOCK_SIZE])
                del self.byte_buffer[:RS_BLOCK_SIZE]

                data, corrected, failed = self.rs_decoder.decode(rs_block)

                if failed:
                    self.rs_errors += 1
                    continue

                # Stage 7: Output to TS
                self.ts_output.write(data)
                self.segments_decoded += 1

        return self._get_status()

    def _dibits_to_bytes(self, dibits):
        """Convert array of 2-bit values to bytes."""
        # Pad to multiple of 4
        n = len(dibits)
        if n % 4 != 0:
            pad = 4 - (n % 4)
            dibits = np.concatenate([dibits, np.zeros(pad, dtype=np.uint8)])

        # Pack 4 dibits into each byte
        dibits = dibits.reshape(-1, 4)
        bytes_out = (dibits[:, 0] << 6) | (dibits[:, 1] << 4) | \
                   (dibits[:, 2] << 2) | dibits[:, 3]
        return bytes_out.astype(np.uint8)

    def _get_status(self):
        """Get decoder status dictionary."""
        return {
            'carrier_locked': self.carrier_locked,
            'sync_locked': self.sync_locked,
            'sync_confidence': self.sync_detector.sync_confidence,
            'segments_decoded': self.segments_decoded,
            'rs_blocks_decoded': self.rs_decoder.blocks_decoded,
            'rs_blocks_corrected': self.rs_decoder.blocks_corrected,
            'rs_blocks_failed': self.rs_decoder.blocks_failed,
            'ts_packets_sent': self.ts_output.packets_sent,
            'ts_bytes_sent': self.ts_output.bytes_sent,
            'timing_error': self.timing_recovery.timing_error_avg,
        }

    def reset(self):
        """Reset decoder state."""
        self.front_end.reset()
        self.timing_recovery.reset()
        self.sync_detector.reset()
        self.deinterleaver.reset()
        self.rs_decoder.reset()
        self.ts_output.reset()
        self.byte_buffer.clear()
        self.carrier_locked = False
        self.sync_locked = False
        self.segments_decoded = 0
        self.rs_errors = 0


def main():
    """Main entry point for ATSC decoder."""
    parser = argparse.ArgumentParser(
        description='ATSC 8VSB Digital TV Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python atsc_decoder.py --channel 24
    python atsc_decoder.py --freq 533

Requirements:
    - Signal Hound BB60D with 20 MHz sample rate
    - mpv for video playback
    - numba for real-time performance (optional but recommended)
    - reedsolo for error correction (optional)
"""
    )
    parser.add_argument('--channel', type=int,
                        help='ATSC RF channel number (2-36)')
    parser.add_argument('--freq', type=float,
                        help='Center frequency in MHz')
    args = parser.parse_args()

    # Determine frequency
    if args.channel:
        try:
            freq = channel_to_freq(args.channel)
            print(f"Channel {args.channel} -> {freq/1e6:.3f} MHz")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.freq:
        freq = args.freq * 1e6
    else:
        print("Error: Specify --channel or --freq")
        parser.print_help()
        sys.exit(1)

    # Import BB60D
    try:
        from bb60d import BB60D
    except ImportError:
        print("Error: bb60d module not found")
        sys.exit(1)

    # Initialize device
    print(f"Opening BB60D at {freq/1e6:.3f} MHz...")
    device = BB60D()

    try:
        device.open()
        device.configure_iq_streaming(freq, int(ATSC_SAMPLE_RATE))
        print(f"BB60D configured: {device.iq_sample_rate/1e6:.3f} MHz sample rate")

        # Initialize decoder
        decoder = ATSCDecoder(device.iq_sample_rate)
        decoder.start()

        print("\nATSC Decoder Running")
        print("=" * 50)
        print("Press Ctrl+C to stop\n")

        # Main loop
        last_status_time = time.time()
        status_interval = 1.0  # Update status every second

        while True:
            # Fetch IQ samples
            samples_per_block = int(ATSC_SAMPLE_RATE * 0.01)  # 10ms blocks
            iq_data = device.fetch_iq(samples_per_block)

            # Process through decoder
            status = decoder.process(iq_data)

            # Print status periodically
            now = time.time()
            if now - last_status_time >= status_interval:
                last_status_time = now

                carrier = "LOCKED" if status['carrier_locked'] else "unlocked"
                sync = "LOCKED" if status['sync_locked'] else "unlocked"

                print(f"\rCarrier: {carrier:8s} | Sync: {sync:8s} | "
                      f"Segments: {status['segments_decoded']:6d} | "
                      f"RS ok/err: {status['rs_blocks_decoded']:5d}/{status['rs_blocks_failed']:3d} | "
                      f"TS: {status['ts_bytes_sent']/1024:.1f} KB",
                      end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        decoder.stop()
        device.close()
        print("Done.")


if __name__ == '__main__':
    main()

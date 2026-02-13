#!/usr/bin/env python3
"""
FM Demodulator Shared Infrastructure and NBFM Decoder

Shared utilities (EMA, FIR helpers, resampler, soft-clip) used by
PLLStereoDecoder (pll_stereo_decoder.py) and NBFMDecoder (below).
"""

import numpy as np
from scipy import signal


def _ema_alpha_from_tau(tau_s, samples, sample_rate_hz):
    """
    Convert a continuous-time EMA time constant to a block update alpha.

    Using this keeps smoothing behavior stable even if demodulate() is called
    with different block sizes.
    """
    n = int(samples)
    if n <= 0:
        return 1.0
    tau = float(tau_s)
    if tau <= 0.0:
        return 1.0
    fs = float(sample_rate_hz)
    if fs <= 0.0:
        return 1.0
    alpha = 1.0 - np.exp(-n / (tau * fs))
    return float(max(0.0, min(1.0, alpha)))


def _validate_odd_taps(value, name, minimum=3):
    """Return validated odd FIR tap count."""
    taps = int(value)
    if taps < minimum or (taps % 2) == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}, got {value}")
    return taps


def _normalize_resampler_mode(value):
    """Normalize resampler mode string."""
    mode = str(value).strip().lower()
    if mode not in {"interp", "firdecim", "auto"}:
        raise ValueError(f"Unknown resampler_mode '{value}' (expected interp|firdecim|auto)")
    return mode


class _StreamingLinearResampler:
    """
    Phase-continuous streaming linear resampler.

    Keeps one scalar phase in input-sample units and computes output count from
    that phase + ratio each block, avoiding per-block rounding bias.
    """

    def __init__(self):
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def process(self, ratio, *channels):
        if not channels:
            return tuple()
        n_in = len(channels[0])
        if n_in <= 0 or ratio <= 0:
            empty = np.array([], dtype=np.float64)
            return tuple(empty for _ in channels)
        if any(len(ch) != n_in for ch in channels):
            raise ValueError("All channels must have equal length")

        step = 1.0 / ratio
        phase = float(self.phase)
        if phase < 0.0 or phase >= n_in:
            phase = phase % n_in

        # Largest k such that phase + k*step < n_in.
        n_out = int(np.floor((n_in - 1e-12 - phase) / step)) + 1
        if n_out <= 0:
            self.phase = phase - n_in
            empty = np.array([], dtype=np.float64)
            return tuple(empty for _ in channels)

        positions = phase + step * np.arange(n_out, dtype=np.float64)
        self.phase = positions[-1] + step - n_in
        idx = np.arange(n_in, dtype=np.float64)
        return tuple(np.interp(positions, idx, ch) for ch in channels)


class _StreamingFIRDecimator:
    """Stateful FIR decimator (lfilter + phased downsample)."""

    def __init__(self, decimation, taps=127, beta=8.0):
        self.decimation = int(decimation)
        if self.decimation < 2:
            raise ValueError(f"decimation must be >=2, got {decimation}")
        self.taps = _validate_odd_taps(taps, "resampler_taps")
        self.beta = float(beta)
        cutoff = 0.45 / self.decimation
        self.h = signal.firwin(self.taps, cutoff, window=("kaiser", self.beta))
        self.zi = np.zeros(len(self.h) - 1, dtype=np.float64)
        self.phase = 0

    def reset(self):
        self.zi.fill(0.0)
        self.phase = 0

    def process(self, x):
        if len(x) == 0:
            return np.array([], dtype=np.float64)
        y, self.zi = signal.lfilter(self.h, [1.0], x, zi=self.zi)
        out = y[self.phase::self.decimation]
        self.phase = (self.phase - len(y)) % self.decimation
        return out


def _soft_clip(x, threshold=0.8):
    """
    Threshold-based soft clipper with tanh rolloff.

    Signals below threshold pass through unchanged (0% THD).
    Signals above threshold are soft-limited using tanh.

    Args:
        x: Input signal (numpy array)
        threshold: Amplitude below which signal is unchanged (default 0.8)

    Returns:
        Soft-clipped signal, max output approaches 1.0 asymptotically
    """
    result = x.copy()
    above = np.abs(x) > threshold
    if np.any(above):
        sign = np.sign(x[above])
        excess = np.abs(x[above]) - threshold
        # tanh rolloff for excess above threshold, scaled to use remaining headroom
        result[above] = sign * (threshold + np.tanh(excess * 5) * (1 - threshold))
    return result



class NBFMDecoder:
    """
    Narrowband FM decoder for weather radio and similar services.

    NBFM characteristics:
    - Deviation: ~5 kHz (vs 75 kHz for broadcast FM)
    - Mono only (no stereo subcarrier)
    - Audio bandwidth: ~3 kHz
    - No de-emphasis (NWS transmits without pre-emphasis)
    """

    def __init__(self, iq_sample_rate=250000, audio_sample_rate=48000,
                 deviation=5000):
        """
        Initialize NBFM decoder.

        Args:
            iq_sample_rate: Input IQ sample rate in Hz
            audio_sample_rate: Output audio sample rate in Hz
            deviation: FM deviation in Hz (5 kHz typical for NBFM)
        """
        self.iq_sample_rate = iq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

        # State for continuous processing
        self.last_sample = complex(1, 0)

        # Adaptive rate control (same as PLLStereoDecoder)
        self._rate_adjust = 1.0
        self._nominal_ratio = audio_sample_rate / iq_sample_rate
        self._audio_resampler = _StreamingLinearResampler()

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Channel filter BEFORE FM demodulation (critical for noise rejection)
        # NBFM channel is ±12.5 kHz; filter reduces noise bandwidth going into discriminator
        # This prevents high-frequency noise spikes from causing phase jumps
        channel_cutoff = 12500 / nyq
        self.channel_lpf = signal.firwin(101, channel_cutoff, window='hamming')
        self.channel_lpf_state = signal.lfilter_zi(self.channel_lpf, 1.0)

        # Audio lowpass filter (3 kHz for NBFM voice)
        audio_cutoff = 3000 / nyq
        self.audio_lpf = signal.firwin(101, audio_cutoff, window='hamming')
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)

        # SNR measurement state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10

        # Design noise measurement bandpass filter (4-6 kHz)
        # Must be WITHIN the 12.5 kHz channel filter passband (otherwise we measure
        # attenuated noise and get falsely high SNR), but above audio content (3 kHz).
        # FM demod noise PSD increases as f², so measuring at 5 kHz vs 1.5 kHz
        # audio center requires correction factor of (5/1.5)² ≈ 11x.
        noise_low = 4000 / nyq
        noise_high = 6000 / nyq
        self.noise_bpf = signal.firwin(101, [noise_low, noise_high],
                                       pass_zero=False, window='hamming')
        self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
        self.noise_bandwidth = 2000  # Hz (4-6 kHz)
        # FM noise triangle correction: noise at 5 kHz vs 1.5 kHz audio center
        self._fm_noise_correction = (5000 / 1500) ** 2  # ≈ 11x

        # Audio bandwidth for SNR scaling
        self.audio_bandwidth = 3000  # Hz

        # Peak amplitude tracking
        self._peak_amplitude = 0.0

        # Tone controls (bass and treble boost) - same as stereo decoder
        self.bass_boost_enabled = False   # Default off for weather
        self.treble_boost_enabled = False # Default off for weather
        self._setup_tone_filters()

    def _design_low_shelf(self, fc, gain_db, fs):
        """Design low shelf biquad filter coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _design_high_shelf(self, fc, gain_db, fs):
        """Design high shelf biquad filter coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _setup_tone_filters(self):
        """Set up bass and treble shelf filters (+3dB)."""
        fs = self.audio_sample_rate
        self.bass_b, self.bass_a = self._design_low_shelf(250, 3.0, fs)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_b, self.treble_a = self._design_high_shelf(3500, 3.0, fs)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)

    @property
    def snr_db(self):
        """Return the current SNR estimate in dB."""
        return self._snr_db

    @property
    def peak_amplitude(self):
        """Return peak amplitude (before limiting)."""
        return self._peak_amplitude

    @property
    def pilot_detected(self):
        """Always False for NBFM (no stereo pilot)."""
        return False

    @property
    def stereo_blend_factor(self):
        """Always 0 for NBFM (mono only)."""
        return 0.0

    @property
    def rate_adjust(self):
        """Return current rate adjustment factor for adaptive resampling."""
        return self._rate_adjust

    @rate_adjust.setter
    def rate_adjust(self, value):
        """Set rate adjustment factor (0.99-1.01 typical range for clock drift)."""
        self._rate_adjust = max(0.98, min(1.02, value))

    @property
    def last_baseband(self):
        """Returns None - NBFM doesn't provide baseband for RDS."""
        return None

    def demodulate(self, iq_samples):
        """
        Demodulate NBFM signal from IQ samples.

        Args:
            iq_samples: numpy array of complex64 IQ samples

        Returns:
            numpy array of shape (N, 2) with mono audio duplicated to L/R
        """
        if len(iq_samples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Channel filter BEFORE FM demodulation to reduce noise bandwidth
        # This is critical - without it, high-frequency noise causes phase spikes
        # that result in crackling audio when the limiter engages
        iq_filtered, self.channel_lpf_state = signal.lfilter(
            self.channel_lpf, 1.0, iq_samples, zi=self.channel_lpf_state
        )

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_filtered])
        self.last_sample = iq_filtered[-1]

        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Audio lowpass filter (3 kHz)
        audio, self.audio_lpf_state = signal.lfilter(
            self.audio_lpf, 1.0, baseband, zi=self.audio_lpf_state
        )

        # SNR measurement
        # Measure signal power from the filtered audio (0-3 kHz)
        signal_power = np.mean(audio ** 2)

        # Measure noise power in a band above audio but within channel filter (4-6 kHz)
        noise_filtered, self.noise_bpf_state = signal.lfilter(
            self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
        )
        noise_power = np.mean(noise_filtered ** 2)
        # Scale noise from measurement bandwidth to audio bandwidth,
        # and apply FM noise triangle correction (noise PSD ∝ f² after FM demod)
        noise_power_scaled = (noise_power * (self.audio_bandwidth / self.noise_bandwidth)
                              / self._fm_noise_correction)

        # Smooth the measurements
        self._signal_power = 0.9 * self._signal_power + 0.1 * signal_power
        self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_power_scaled, 1e-12)

        if self._noise_power > 0:
            self._snr_db = 10 * np.log10(self._signal_power / self._noise_power)

        # Resample to audio rate with adaptive rate control
        ratio_eff = self._nominal_ratio * self._rate_adjust
        (audio,) = self._audio_resampler.process(ratio_eff, audio)

        # Scale down to provide headroom (0.4 matches panadapter's conservative gain)
        audio = audio * 0.4

        # Apply tone controls
        if self.bass_boost_enabled:
            audio, self.bass_state_l = signal.lfilter(
                self.bass_b, self.bass_a, audio, zi=self.bass_state_l
            )
        if self.treble_boost_enabled:
            audio, self.treble_state_l = signal.lfilter(
                self.treble_b, self.treble_a, audio, zi=self.treble_state_l
            )

        # Track peak amplitude
        peak = np.max(np.abs(audio))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)

        # Apply soft limiting to catch any remaining peaks
        # Threshold-based limiter: linear below 0.8, tanh rolloff above.
        audio = _soft_clip(audio, threshold=0.8)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # Duplicate mono to stereo
        return np.column_stack((audio, audio))

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10
        self._peak_amplitude = 0.0
        self._audio_resampler.reset()

        # Reset filter states
        self.channel_lpf_state = signal.lfilter_zi(self.channel_lpf, 1.0)
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)
        self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)

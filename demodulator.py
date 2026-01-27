#!/usr/bin/env python3
"""
Software FM Stereo Demodulator

Provides FM stereo demodulation from IQ samples using quadrature demodulation
with 19 kHz pilot detection and L-R subcarrier decoding.
"""

import math
import time
import numpy as np
from scipy import signal


class PilotPLL:
    """
    Phase-Locked Loop for FM stereo pilot tracking.

    Tracks the 19 kHz pilot tone and generates coherent
    19 kHz and 38 kHz carriers for stereo decoding.

    Uses a 2nd-order Type 2 PLL with proportional + integral
    loop filter. The feedback loop runs at a decimated rate
    (~4 kHz) since the pilot BPF output has only ~1 kHz
    bandwidth. The tracked phase is interpolated back to full
    rate and carriers are generated with vectorized numpy trig.

    This reduces the Python for-loop from ~8192 iterations to
    ~105 per block while preserving lock performance (the loop
    bandwidth is 100 Hz, well below the ~2 kHz Nyquist of the
    decimated rate).
    """

    # Target rate for the decimated PLL feedback loop (Hz).
    # Must be > 2x loop bandwidth (100 Hz) for stability.
    # 4 kHz gives 40x oversampling of the loop bandwidth.
    PLL_RATE = 4000

    def __init__(self, sample_rate, center_freq=19000,
                 bandwidth=100, damping=0.707):
        """
        Initialize PLL.

        Args:
            sample_rate: Input sample rate in Hz
            center_freq: Pilot frequency in Hz (19000)
            bandwidth: Loop bandwidth in Hz (affects tracking speed vs noise)
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

        # Full-rate constants
        self._dt = 1.0 / sample_rate
        self._omega_0 = 2 * np.pi * center_freq

        # Decimation factor: run feedback loop at ~PLL_RATE
        self._decim = max(1, int(sample_rate / self.PLL_RATE))
        # Decimated time step (for the feedback loop)
        self._dt_decim = self._decim / sample_rate

    def process(self, pilot_signal):
        """
        Process pilot signal through PLL.

        The feedback loop runs on a decimated version of the pilot
        signal. The resulting phase trajectory is interpolated to
        full rate, then vectorized numpy trig generates the carriers.

        Args:
            pilot_signal: Filtered 19 kHz pilot samples (numpy array)

        Returns:
            tuple: (carrier_19k, carrier_38k, locked)
                - carrier_19k: Coherent 19 kHz carrier
                - carrier_38k: Coherent 38 kHz carrier for L-R demod
                - locked: True if PLL is locked to pilot
        """
        n = len(pilot_signal)
        decim = self._decim

        # --- Decimated feedback loop ---
        # Average the pilot signal over each decimation block to get
        # a representative sample for phase detection.
        n_decim = n // decim
        remainder = n - n_decim * decim

        # Reshape and average for decimation (fast, vectorized)
        if n_decim > 0:
            pilot_decim = pilot_signal[:n_decim * decim].reshape(n_decim, decim).mean(axis=1)
        else:
            pilot_decim = np.array([], dtype=pilot_signal.dtype)

        # Handle remainder samples
        if remainder > 0:
            pilot_decim = np.append(pilot_decim, pilot_signal[n_decim * decim:].mean())
            n_decim_total = n_decim + 1
        else:
            n_decim_total = n_decim

        # Run scalar PLL at decimated rate — collect phase at each step
        # We need phase values at decimated points, then interpolate
        phase_points = np.empty(n_decim_total + 1, dtype=np.float64)
        phase_points[0] = self.phase

        phase = self.phase
        integrator = self.integrator
        freq_offset = self.freq_offset
        dt_decim = self._dt_decim
        omega_0 = self._omega_0
        Kp = self.Kp
        Ki = self.Ki

        _sin = math.sin
        _pi = math.pi
        _two_pi = 2.0 * _pi

        for i in range(n_decim_total):
            # Phase detector using current NCO phase vs decimated pilot
            phase_error = pilot_decim[i] * _sin(phase)

            # Loop filter (PI controller)
            integrator += phase_error * Ki * dt_decim
            freq_offset = Kp * phase_error + integrator

            # Advance phase by one decimated step
            phase += (omega_0 + freq_offset) * dt_decim

            # Wrap phase (single wrap is sufficient — phase step is
            # decim * omega_0 * dt ≈ decim * 0.38 rad, well under pi
            # even for decim=78)
            if phase > _pi:
                phase -= _two_pi
            elif phase < -_pi:
                phase += _two_pi

            phase_points[i + 1] = phase

        # Store state
        self.phase = phase
        self.integrator = integrator
        self.freq_offset = freq_offset

        # Lock detection: frequency offset should be small when locked
        # Within 50 Hz of center = locked
        self.locked = abs(freq_offset) < 2 * np.pi * 50

        # --- Interpolate phase to full rate (vectorized) ---
        # phase_points[0] is the phase at sample 0 (start of block)
        # phase_points[k] is the phase at sample k*decim
        # We need phase at every sample 0..n-1

        # Sample indices for the decimated phase points
        decim_indices = np.arange(n_decim_total + 1) * decim
        # Clamp last index to n for the remainder case
        if remainder > 0:
            decim_indices[-1] = n

        # Full-rate sample indices
        full_indices = np.arange(n)

        # Linear interpolation of unwrapped phase
        # Unwrap before interpolation to avoid discontinuities at ±pi
        phase_unwrapped = np.unwrap(phase_points)
        phase_full = np.interp(full_indices, decim_indices, phase_unwrapped)

        # --- Vectorized carrier generation ---
        carrier_19k = np.cos(phase_full)
        carrier_38k = np.cos(2.0 * phase_full)

        return carrier_19k, carrier_38k, self.locked

    def reset(self):
        """Reset PLL state (call when changing frequency)."""
        self.phase = 0.0
        self.freq_offset = 0.0
        self.integrator = 0.0
        self.locked = False


class FMStereoDecoder:
    """
    FM Stereo decoder for broadcast FM.

    Detects 19 kHz pilot tone and decodes L-R difference signal
    from 38 kHz DSB-SC subcarrier.

    FM Stereo signal structure:
    - 0-15 kHz: L+R (mono compatible)
    - 19 kHz: Pilot tone
    - 23-53 kHz: L-R on 38 kHz DSB-SC carrier
    """

    def __init__(self, iq_sample_rate=250000, audio_sample_rate=48000,
                 deviation=75000, deemphasis=75e-6):
        """
        Initialize FM stereo decoder.

        Args:
            iq_sample_rate: Input IQ sample rate in Hz
            audio_sample_rate: Output audio sample rate in Hz
            deviation: FM deviation in Hz (75 kHz for broadcast FM)
            deemphasis: De-emphasis time constant in seconds
        """
        self.iq_sample_rate = iq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

        # Pilot detection state
        self._pilot_detected = False
        self._pilot_level = 0.0
        self.pilot_threshold = 0.05  # Minimum pilot level for detection

        # State for continuous processing
        self.last_sample = complex(1, 0)
        self.pilot_phase = 0.0

        # Calculate resampling ratio
        from math import gcd
        iq_int = int(iq_sample_rate)
        audio_int = int(audio_sample_rate)
        g = gcd(iq_int, audio_int)
        self.resample_up = audio_int // g
        self.resample_down = iq_int // g

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Pilot bandpass filter (18.5-19.5 kHz) — Kaiser beta=7.0 (~70 dB stopband)
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf = signal.firwin(201, [pilot_low, pilot_high],
                                        pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_state = signal.lfilter_zi(self.pilot_bpf, 1.0)

        # L+R lowpass filter (15 kHz) — Kaiser beta=6.0 (~60 dB stopband)
        lr_sum_cutoff = 15000 / nyq
        self.lr_sum_lpf = signal.firwin(127, lr_sum_cutoff, window=('kaiser', 6.0))
        self.lr_sum_lpf_state = signal.lfilter_zi(self.lr_sum_lpf, 1.0)

        # L-R bandpass filter (23-53 kHz) — Kaiser beta=7.0 (~70 dB stopband)
        lr_diff_low = 23000 / nyq
        lr_diff_high = min(53000 / nyq, 0.95)  # Stay below Nyquist
        self.lr_diff_bpf = signal.firwin(201, [lr_diff_low, lr_diff_high],
                                          pass_zero=False, window=('kaiser', 7.0))
        self.lr_diff_bpf_state = signal.lfilter_zi(self.lr_diff_bpf, 1.0)

        # L-R lowpass filter after demodulation (15 kHz) — Kaiser beta=6.0 (~60 dB stopband)
        self.lr_diff_lpf = signal.firwin(127, lr_sum_cutoff, window=('kaiser', 6.0))
        self.lr_diff_lpf_state = signal.lfilter_zi(self.lr_diff_lpf, 1.0)

        # De-emphasis filter (at output audio rate)
        # Use matched-pole first-order IIR for accurate 75µs time constant
        fs = audio_sample_rate
        a = np.exp(-1.0 / (deemphasis * fs))
        self.deem_b = np.array([1.0 - a])
        self.deem_a = np.array([1.0, -a])
        self.deem_state_l = signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = signal.lfilter_zi(self.deem_b, self.deem_a)

        # Design noise measurement bandpass filter (75-95 kHz)
        noise_low = 75000 / nyq
        noise_high = min(95000 / nyq, 0.95)
        if noise_high > noise_low:
            self.noise_bpf = signal.firwin(51, [noise_low, noise_high],
                                           pass_zero=False, window=('kaiser', 5.0))
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = 20000  # Hz
        else:
            self.noise_bpf = None

        # SNR measurement state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10
        # Audio bandwidth: 15 kHz for mono-compatible (L+R), or 53 kHz for full stereo
        self.mono_bandwidth = 15000    # L+R only
        self.stereo_bandwidth = 53000  # Full stereo (L+R + L-R)

        # Peak amplitude tracking (for clipping detection)
        self._peak_amplitude = 0.0

        # Group delay compensation buffer for L+R path.
        # The L-R path passes through an extra BPF (201 taps, delay=100)
        # before reaching the LPF. Both LPFs are 127-tap so their delays
        # cancel. Without compensation the L+R and L-R signals are
        # misaligned by 100 samples (~400 us at 250 kHz), which destroys
        # stereo separation at higher audio frequencies.
        self._lr_sum_delay = (len(self.lr_diff_bpf) - 1) // 2
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Store last baseband for RDS processing
        self._last_baseband = None
        self._last_pilot = None

        # PLL for pilot tracking (optional, default enabled)
        # Provides cleaner 38 kHz carrier with better phase coherency
        self.use_pll = True
        self.pilot_pll = PilotPLL(
            sample_rate=iq_sample_rate,
            center_freq=19000,
            bandwidth=100,    # 100 Hz loop bandwidth
            damping=0.707     # Critical damping
        )

        # Tone controls (bass and treble boost)
        self.bass_boost_enabled = True    # Default on
        self.treble_boost_enabled = True  # Default on
        self._setup_tone_filters()

        # Optional GPU demodulator for baseband step
        self.gpu_demodulator = None
        # Optional GPU resampler for rate conversion step
        self.gpu_resampler = None
        # Optional GPU FIR bank (pilot_bpf, lr_sum_lpf, lr_diff_bpf — shared input)
        self.gpu_fir_bank = None
        # Optional GPU FIR filter for lr_diff_lpf (different input)
        self.gpu_fir_lr_diff = None

        # Stereo blend settings (blend to mono when SNR is low)
        self.stereo_blend_enabled = True
        self.stereo_blend_low = 15.0    # Below this SNR: full mono
        self.stereo_blend_high = 30.0   # Above this SNR: full stereo
        self._blend_factor = 1.0        # Current blend (0=mono, 1=stereo)

        # Per-stage profiling (EMA-smoothed, microseconds)
        self.profile_enabled = False
        self._profile = {
            'fm_demod': 0.0,
            'pilot_bpf': 0.0,
            'pll': 0.0,
            'lr_sum_lpf': 0.0,
            'lr_diff_bpf': 0.0,
            'lr_diff_lpf': 0.0,
            'noise_bpf': 0.0,
            'resample': 0.0,
            'deemphasis': 0.0,
            'tone': 0.0,
            'limiter': 0.0,
            'total': 0.0,
        }

    def _design_low_shelf(self, fc, gain_db, fs):
        """Design low shelf biquad filter coefficients (Audio EQ Cookbook)."""
        A = 10 ** (gain_db / 40)  # sqrt of linear gain
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q = 0.707 for gentle slope

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

        # Normalize
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _design_high_shelf(self, fc, gain_db, fs):
        """Design high shelf biquad filter coefficients (Audio EQ Cookbook)."""
        A = 10 ** (gain_db / 40)  # sqrt of linear gain
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q = 0.707 for gentle slope

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

        # Normalize
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _setup_tone_filters(self):
        """Set up bass and treble shelf filters (+3dB)."""
        fs = self.audio_sample_rate

        # Bass boost: +3dB low shelf at 250 Hz
        self.bass_b, self.bass_a = self._design_low_shelf(250, 3.0, fs)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = signal.lfilter_zi(self.bass_b, self.bass_a)

        # Treble boost: +3dB high shelf at 3500 Hz
        self.treble_b, self.treble_a = self._design_high_shelf(3500, 3.0, fs)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = signal.lfilter_zi(self.treble_b, self.treble_a)

    @property
    def pilot_detected(self):
        """Returns True if 19 kHz pilot tone is detected."""
        return self._pilot_detected

    @property
    def snr_db(self):
        """Return the current SNR estimate in dB."""
        return self._snr_db

    @property
    def peak_amplitude(self):
        """Return peak amplitude (before limiting). >1.0 means limiter is active."""
        return self._peak_amplitude

    @property
    def stereo_blend_factor(self):
        """Return current stereo blend factor (0=mono, 1=full stereo)."""
        return self._blend_factor

    @property
    def profile(self):
        """Return per-stage profiling data (EMA-smoothed microseconds)."""
        return dict(self._profile)

    @property
    def last_baseband(self):
        """Returns the last FM baseband signal for RDS processing.

        This is the demodulated FM signal at the IQ sample rate,
        before audio filtering and decimation. Contains the full
        spectrum including 57 kHz RDS subcarrier.
        """
        return self._last_baseband

    @property
    def last_pilot(self):
        """Returns the last extracted 19 kHz pilot signal.

        Can be used by RDS decoder to derive 57 kHz carrier (3x pilot).
        """
        return self._last_pilot

    def _prof(self, key, t0):
        """Record EMA-smoothed stage timing in microseconds."""
        elapsed = (time.perf_counter() - t0) * 1e6
        self._profile[key] = 0.9 * self._profile[key] + 0.1 * elapsed
        return time.perf_counter()

    def demodulate(self, iq_samples):
        """
        Demodulate FM stereo signal from IQ samples.

        Args:
            iq_samples: numpy array of complex64 IQ samples

        Returns:
            numpy array of shape (N, 2) with L and R channels as float32
        """
        if len(iq_samples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        profiling = self.profile_enabled
        if profiling:
            t_total = time.perf_counter()
            t0 = t_total

        # FM demodulation
        if self.gpu_demodulator is not None:
            # GPU arctangent-differentiate method
            baseband = self.gpu_demodulator.demodulate(iq_samples)
            # Still update last_sample for continuity if GPU is toggled off
            self.last_sample = iq_samples[-1]
        else:
            # CPU quadrature discriminator
            samples = np.concatenate([[self.last_sample], iq_samples])
            self.last_sample = iq_samples[-1]
            product = samples[1:] * np.conj(samples[:-1])
            baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Store baseband for RDS processing (at IQ sample rate, before decimation)
        self._last_baseband = baseband

        if profiling:
            t0 = self._prof('fm_demod', t0)

        # Extract pilot, L+R, and L-R subcarrier via GPU bank or CPU lfilter
        if self.gpu_fir_bank is not None:
            pilot, lr_sum, lr_diff_mod_raw = self.gpu_fir_bank.process(baseband)
        else:
            pilot, self.pilot_bpf_state = signal.lfilter(
                self.pilot_bpf, 1.0, baseband, zi=self.pilot_bpf_state
            )
            lr_sum = None  # computed below
            lr_diff_mod_raw = None

        # Store pilot for RDS decoder
        self._last_pilot = pilot  # No copy - pilot is fresh from filter

        # Measure pilot level for detection
        pilot_power = np.sqrt(np.mean(pilot ** 2))
        # Smooth the pilot level measurement
        self._pilot_level = 0.9 * self._pilot_level + 0.1 * pilot_power

        if profiling:
            t0 = self._prof('pilot_bpf', t0)

        # Pilot detection: require both RMS threshold AND PLL lock (if using PLL)
        # This prevents false positives from broadband noise on blank stations
        carrier_38k = None
        if self._pilot_level > self.pilot_threshold:
            if self.use_pll:
                # Run PLL to check for coherent tone lock
                _, carrier_38k, pll_locked = self.pilot_pll.process(pilot)
                self._pilot_detected = pll_locked
            else:
                # Legacy mode: use RMS threshold only
                pilot_normalized = pilot / (np.abs(pilot).max() + 1e-10)
                carrier_38k = 2 * pilot_normalized ** 2 - 1  # cos(2x) = 2cos^2(x) - 1
                self._pilot_detected = True
        else:
            self._pilot_detected = False

        if profiling:
            t0 = self._prof('pll', t0)

        # Extract L+R (mono, 0-15 kHz) — may already be computed by GPU bank
        if lr_sum is None:
            lr_sum, self.lr_sum_lpf_state = signal.lfilter(
                self.lr_sum_lpf, 1.0, baseband, zi=self.lr_sum_lpf_state
            )

        # Apply group delay compensation: delay L+R to align with L-R path
        delay = self._lr_sum_delay
        delayed = np.empty_like(lr_sum)
        delayed[:delay] = self._lr_sum_delay_buf
        delayed[delay:] = lr_sum[:-delay] if delay > 0 else lr_sum
        self._lr_sum_delay_buf = lr_sum[-delay:].copy()
        lr_sum = delayed

        if profiling:
            t0 = self._prof('lr_sum_lpf', t0)

        if self._pilot_detected:
            # Extract L-R subcarrier region (23-53 kHz)
            if lr_diff_mod_raw is not None:
                lr_diff_mod = lr_diff_mod_raw
            else:
                lr_diff_mod, self.lr_diff_bpf_state = signal.lfilter(
                    self.lr_diff_bpf, 1.0, baseband, zi=self.lr_diff_bpf_state
                )

            if profiling:
                t0 = self._prof('lr_diff_bpf', t0)

            # Demodulate L-R by multiplying with 38 kHz carrier
            lr_diff_demod = lr_diff_mod * carrier_38k * 2  # *2 for DSB-SC gain

            # Lowpass filter the demodulated L-R
            if self.gpu_fir_lr_diff is not None:
                lr_diff = self.gpu_fir_lr_diff.process(lr_diff_demod)
            else:
                lr_diff, self.lr_diff_lpf_state = signal.lfilter(
                    self.lr_diff_lpf, 1.0, lr_diff_demod, zi=self.lr_diff_lpf_state
                )

            if profiling:
                t0 = self._prof('lr_diff_lpf', t0)

            # Matrix decode
            left_stereo = lr_sum + lr_diff
            right_stereo = lr_sum - lr_diff

            # SNR measurement for stereo (use full stereo bandwidth)
            signal_power = np.mean(lr_sum ** 2) + np.mean(lr_diff ** 2)
            audio_bandwidth = self.stereo_bandwidth

            # Apply stereo blend based on SNR (blend to mono when noisy)
            if self.stereo_blend_enabled:
                # Calculate blend factor from SNR (smoothed for stability)
                target_blend = (self._snr_db - self.stereo_blend_low) / (self.stereo_blend_high - self.stereo_blend_low)
                target_blend = max(0.0, min(1.0, target_blend))
                # Smooth blend factor to avoid sudden changes
                self._blend_factor = 0.95 * self._blend_factor + 0.05 * target_blend

                # Blend: 0 = mono (lr_sum), 1 = full stereo
                left = self._blend_factor * left_stereo + (1.0 - self._blend_factor) * lr_sum
                right = self._blend_factor * right_stereo + (1.0 - self._blend_factor) * lr_sum
            else:
                left = left_stereo
                right = right_stereo
        else:
            # No stereo - output mono to both channels
            left = lr_sum
            right = lr_sum
            self._blend_factor = 0.0  # Track that we're in mono

            # SNR measurement for mono (L+R only)
            signal_power = np.mean(lr_sum ** 2)
            audio_bandwidth = self.mono_bandwidth

        # Measure noise power (in high frequency band above all subcarriers)
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)
            # Scale noise to audio bandwidth (noise power density * bandwidth)
            noise_power_scaled = noise_power * (audio_bandwidth / self.noise_bandwidth)

            # Smooth the measurements
            self._signal_power = 0.9 * self._signal_power + 0.1 * signal_power
            self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_power_scaled, 1e-12)

            # Calculate SNR in dB
            if self._noise_power > 0:
                self._snr_db = 10 * np.log10(self._signal_power / self._noise_power)

        if profiling:
            t0 = self._prof('noise_bpf', t0)

        # Resample to audio rate
        if self.gpu_resampler is not None:
            left, right = self.gpu_resampler.resample(left, right)
        else:
            left = signal.resample_poly(left, self.resample_up, self.resample_down)
            right = signal.resample_poly(right, self.resample_up, self.resample_down)

        if profiling:
            t0 = self._prof('resample', t0)

        # Apply de-emphasis to each channel
        left, self.deem_state_l = signal.lfilter(
            self.deem_b, self.deem_a, left, zi=self.deem_state_l
        )
        right, self.deem_state_r = signal.lfilter(
            self.deem_b, self.deem_a, right, zi=self.deem_state_r
        )

        if profiling:
            t0 = self._prof('deemphasis', t0)

        # Apply tone controls (bass and treble boost)
        if self.bass_boost_enabled:
            left, self.bass_state_l = signal.lfilter(
                self.bass_b, self.bass_a, left, zi=self.bass_state_l
            )
            right, self.bass_state_r = signal.lfilter(
                self.bass_b, self.bass_a, right, zi=self.bass_state_r
            )
        if self.treble_boost_enabled:
            left, self.treble_state_l = signal.lfilter(
                self.treble_b, self.treble_a, left, zi=self.treble_state_l
            )
            right, self.treble_state_r = signal.lfilter(
                self.treble_b, self.treble_a, right, zi=self.treble_state_r
            )

        if profiling:
            t0 = self._prof('tone', t0)

        # Track peak amplitude (before limiting, for clipping detection)
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)  # Fast attack, slow decay

        # Apply soft limiting to prevent harsh clipping distortion
        # Scale down slightly to provide headroom for over-modulated stations
        left = left * 0.7
        right = right * 0.7
        # Soft limiter using tanh (smooth saturation instead of hard clipping)
        tanh_scale = np.tanh(1.5)
        left = np.tanh(left * 1.5) / tanh_scale
        right = np.tanh(right * 1.5) / tanh_scale
        # Hard clip to ensure output never exceeds ±1.0 (tanh asymptote is 1.105)
        left = np.clip(left, -1.0, 1.0)
        right = np.clip(right, -1.0, 1.0)
        left = left.astype(np.float32)
        right = right.astype(np.float32)

        if profiling:
            self._prof('limiter', t0)
            elapsed_total = (time.perf_counter() - t_total) * 1e6
            self._profile['total'] = 0.9 * self._profile['total'] + 0.1 * elapsed_total

        return np.column_stack((left, right))

    def set_carrier_mode(self, use_pll=True):
        """
        Set carrier regeneration mode.

        Args:
            use_pll: True for PLL-based (default), False for pilot-squaring
        """
        self.use_pll = use_pll
        if use_pll:
            self.pilot_pll.reset()

    def set_bass_boost(self, enabled):
        """Enable or disable +3dB bass boost at 250 Hz."""
        self.bass_boost_enabled = enabled

    def set_treble_boost(self, enabled):
        """Enable or disable +3dB treble boost at 3.5 kHz."""
        self.treble_boost_enabled = enabled

    def set_tone_controls(self, bass=True, treble=True):
        """
        Set both tone controls at once.

        Args:
            bass: Enable bass boost (+3dB at 250 Hz)
            treble: Enable treble boost (+3dB at 3.5 kHz)
        """
        self.bass_boost_enabled = bass
        self.treble_boost_enabled = treble

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._pilot_detected = False
        self._pilot_level = 0.0
        self.pilot_phase = 0.0
        self._last_baseband = None

        # Reset filter states
        self.pilot_bpf_state = signal.lfilter_zi(self.pilot_bpf, 1.0)
        self.lr_sum_lpf_state = signal.lfilter_zi(self.lr_sum_lpf, 1.0)
        self.lr_diff_bpf_state = signal.lfilter_zi(self.lr_diff_bpf, 1.0)
        self.lr_diff_lpf_state = signal.lfilter_zi(self.lr_diff_lpf, 1.0)
        self.deem_state_l = signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = signal.lfilter_zi(self.deem_b, self.deem_a)
        if self.noise_bpf is not None:
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)

        # Reset group delay compensation buffer
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Reset PLL state
        self.pilot_pll.reset()

        # Reset SNR state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10
        self._peak_amplitude = 0.0
        self._blend_factor = 1.0  # Start at full stereo

        # Reset tone control filter states
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = signal.lfilter_zi(self.treble_b, self.treble_a)


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

        # Calculate resampling ratio
        from math import gcd
        iq_int = int(iq_sample_rate)
        audio_int = int(audio_sample_rate)
        g = gcd(iq_int, audio_int)
        self.resample_up = audio_int // g
        self.resample_down = iq_int // g

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Audio lowpass filter (3 kHz for NBFM voice)
        audio_cutoff = 3000 / nyq
        self.audio_lpf = signal.firwin(101, audio_cutoff, window='hamming')
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)

        # SNR measurement state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10

        # Design noise measurement bandpass filter (20-30 kHz)
        # Must be well outside NBFM signal bandwidth (Carson's rule: 2*(5kHz + 3kHz) = 16kHz)
        # Using 20-30kHz to be safely outside the signal spectrum
        noise_low = 20000 / nyq
        noise_high = min(30000 / nyq, 0.95)
        if noise_high > noise_low:
            self.noise_bpf = signal.firwin(51, [noise_low, noise_high],
                                           pass_zero=False, window='hamming')
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = 10000  # Hz
        else:
            self.noise_bpf = None

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

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_samples])
        self.last_sample = iq_samples[-1]

        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Audio lowpass filter (3 kHz)
        audio, self.audio_lpf_state = signal.lfilter(
            self.audio_lpf, 1.0, baseband, zi=self.audio_lpf_state
        )

        # SNR measurement
        # Measure signal power from the filtered audio
        signal_power = np.mean(audio ** 2)

        # Measure noise power in a band outside the FM signal spectrum
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)
            # Scale noise from measurement bandwidth to audio bandwidth
            noise_power_scaled = noise_power * (self.audio_bandwidth / self.noise_bandwidth)

            # Smooth the measurements
            self._signal_power = 0.9 * self._signal_power + 0.1 * signal_power
            self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_power_scaled, 1e-12)

            if self._noise_power > 0:
                self._snr_db = 10 * np.log10(self._signal_power / self._noise_power)

        # Resample to audio rate
        audio = signal.resample_poly(audio, self.resample_up, self.resample_down)

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

        # Apply soft limiting
        audio = audio * 0.7
        tanh_scale = np.tanh(1.5)
        audio = np.tanh(audio * 1.5) / tanh_scale
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

        # Reset filter states
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)
        if self.noise_bpf is not None:
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)

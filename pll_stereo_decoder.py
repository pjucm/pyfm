#!/usr/bin/env python3
"""
PLL-based FM Stereo Decoder

PLL-based FM stereo decoder using a second-order Type 2 PLL
for 38 kHz carrier regeneration.

- Narrow loop bandwidth (30 Hz) rejects pilot noise
- Phase-coherent carrier tracking
- Explicit lock/unlock detection with hysteresis
"""

import time
import numpy as np
from scipy import signal as sp_signal

from demodulator import (
    _soft_clip,
    _validate_odd_taps,
    _StreamingLinearResampler,
    _ema_alpha_from_tau,
)


try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


def _pll_process_kernel_python(
    pilot_filtered,
    phase,
    integrator,
    pe_filt,
    pe_avg,
    omega0,
    kp,
    ki,
    pe_alpha,
    iq_alpha,
    lock_alpha,
    i_lp,
    q_lp,
):
    """Reference PLL inner loop (Python)."""
    n = len(pilot_filtered)
    carrier_38k = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * np.pi

    for i in range(n):
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        # Use current phase for this sample's carrier output.
        # Emitting after phase advance introduces an ~1-sample lead at 38 kHz,
        # which directly degrades stereo separation.
        carrier_38k[i] = 2.0 * cos_phase * cos_phase - 1.0

        # Phase detector: multiply input by -sin(phase) of NCO
        pe = pilot_filtered[i] * (-sin_phase)

        # IIR lowpass to remove 2×pilot (38 kHz) from PE
        pe_filt = pe_filt + pe_alpha * (pe - pe_filt)

        # Low-pass pilot I/Q for normalized phase metric.
        i_mix = pilot_filtered[i] * cos_phase
        q_mix = pe
        i_lp = i_lp + iq_alpha * (i_mix - i_lp)
        q_lp = q_lp + iq_alpha * (q_mix - q_lp)

        # PI loop filter
        integrator += ki * pe_filt
        correction = kp * pe_filt + integrator

        # NCO: advance phase
        phase += omega0 + correction
        # Wrap to [0, 2π)
        phase = phase % two_pi

        # Lock detector: EMA of squared filtered phase error
        pe_avg = pe_avg + lock_alpha * (pe_filt * pe_filt - pe_avg)

    return carrier_38k, phase, integrator, pe_filt, pe_avg, i_lp, q_lp


_pll_process_kernel_numba = None
if njit is not None:  # pragma: no branch - one-time configuration
    _pll_process_kernel_numba = njit(cache=True)(_pll_process_kernel_python)


_NUMBA_PLL_KERNEL_READY = None


def _numba_pll_kernel_available():
    """Return True when Numba kernel can compile and run."""
    global _NUMBA_PLL_KERNEL_READY
    if _pll_process_kernel_numba is None:
        return False
    if _NUMBA_PLL_KERNEL_READY is None:
        try:
            pilot = np.zeros(1, dtype=np.float64)
            _pll_process_kernel_numba(
                pilot,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            _NUMBA_PLL_KERNEL_READY = True
        except Exception:
            _NUMBA_PLL_KERNEL_READY = False
    return _NUMBA_PLL_KERNEL_READY


def prewarm_numba_pll_kernel():
    """
    Eagerly compile/check the optional Numba PLL kernel.

    Call this before real-time streaming to avoid first-use JIT latency inside
    time-critical audio/IQ paths.
    """
    return _numba_pll_kernel_available()


class PLLStereoDecoder:
    """
    FM Stereo decoder using PLL-based carrier recovery.

    Uses a second-order Type 2 PLL locked to the 19 kHz pilot tone
    to regenerate a phase-coherent 38 kHz carrier via frequency doubling.
    """

    def __init__(self, iq_sample_rate=250000, audio_sample_rate=48000,
                 deviation=75000, deemphasis=75e-6, force_mono=False,
                 stereo_lpf_taps=127, stereo_lpf_beta=6.0,
                 pll_kernel_mode="auto"):
        self.iq_sample_rate = iq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation
        self.force_mono = force_mono
        self.stereo_lpf_taps = _validate_odd_taps(stereo_lpf_taps, "stereo_lpf_taps")
        self.stereo_lpf_beta = float(stereo_lpf_beta)

        mode = str(pll_kernel_mode).strip().lower()
        if mode not in {"auto", "python", "numba"}:
            raise ValueError("pll_kernel_mode must be one of: auto, python, numba")
        if mode == "python":
            backend = "python"
        elif mode == "numba":
            if not _numba_pll_kernel_available():
                raise ValueError("pll_kernel_mode='numba' requested but Numba is unavailable")
            backend = "numba"
        else:
            backend = "numba" if _numba_pll_kernel_available() else "python"
        self._pll_kernel_mode = mode
        self._pll_backend = backend
        self._pll_kernel = (
            _pll_process_kernel_numba if backend == "numba" else _pll_process_kernel_python
        )

        # Pilot detection state
        self._pilot_detected = False
        self._pilot_level = 0.0
        self.pilot_threshold = 0.05

        # State for continuous processing
        self.last_sample = complex(1, 0)

        # Adaptive rate control
        self._rate_adjust = 1.0
        self._nominal_ratio = audio_sample_rate / iq_sample_rate
        self._resampler = _StreamingLinearResampler()

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Pilot bandpass filter (18.5-19.5 kHz)
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf = sp_signal.firwin(201, [pilot_low, pilot_high],
                                           pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_state = sp_signal.lfilter_zi(self.pilot_bpf, 1.0)

        # L+R lowpass filter (15 kHz)
        lr_sum_cutoff = 15000 / nyq
        self.lr_sum_lpf = sp_signal.firwin(
            self.stereo_lpf_taps, lr_sum_cutoff, window=('kaiser', self.stereo_lpf_beta)
        )
        self.lr_sum_lpf_state = sp_signal.lfilter_zi(self.lr_sum_lpf, 1.0)

        # L-R bandpass filter (20-56 kHz)
        lr_diff_low = 20000 / nyq
        lr_diff_high = min(56000 / nyq, 0.95)
        self.lr_diff_bpf = sp_signal.firwin(201, [lr_diff_low, lr_diff_high],
                                             pass_zero=False, window=('kaiser', 7.0))
        self.lr_diff_bpf_state = sp_signal.lfilter_zi(self.lr_diff_bpf, 1.0)

        # L-R lowpass filter after demodulation (15 kHz)
        self.lr_diff_lpf = sp_signal.firwin(
            self.stereo_lpf_taps, lr_sum_cutoff, window=('kaiser', self.stereo_lpf_beta)
        )
        self.lr_diff_lpf_state = sp_signal.lfilter_zi(self.lr_diff_lpf, 1.0)

        # De-emphasis filter (at output audio rate)
        fs = audio_sample_rate
        a = np.exp(-1.0 / (deemphasis * fs))
        self.deem_b = np.array([1.0 - a])
        self.deem_a = np.array([1.0, -a])
        self.deem_state_l = sp_signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = sp_signal.lfilter_zi(self.deem_b, self.deem_a)

        # Noise measurement bandpass filter
        noise_low_hz = 90000
        noise_high_hz = 100000
        noise_low = noise_low_hz / nyq
        noise_high = min(noise_high_hz / nyq, 0.90)
        if noise_high > noise_low and noise_low < 0.90:
            self.noise_bpf = sp_signal.firwin(51, [noise_low, noise_high],
                                               pass_zero=False, window=('kaiser', 5.0))
            self.noise_bpf_state = sp_signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = noise_high_hz - noise_low_hz
        else:
            self.noise_bpf = None

        # SNR measurement state
        self._snr_db = 0.0
        self._noise_power = 1e-10
        self.pilot_bandwidth = 1000
        # Time constants (seconds) for block-size-invariant state smoothing.
        self._pilot_level_tau_s = 0.16
        self._noise_power_tau_s = 0.16
        # Composite stereo quality metric used for blend/RDS gating.
        # This falls back to pilot/PLL-derived quality when the legacy
        # high-band (90-100 kHz) noise estimator is unavailable.
        self._stereo_quality_db = -20.0
        self._pilot_metric_db = 0.0
        self._phase_penalty_db = 0.0
        self._coherence_penalty_db = 0.0
        # Keep this faster than blend smoothing so stereo width does not spend
        # a long time near mono after lock on clean signals.
        self._stereo_quality_tau_s = 0.05
        self._quality_phase_penalty_db = 20.0
        self._quality_coherence_penalty_db = 12.0
        self._quality_coherence_low = 0.30
        self._quality_coherence_high = 0.70

        # Peak amplitude tracking
        self._peak_amplitude = 0.0

        # Group delay compensation buffer for L+R path
        self._lr_sum_delay = (len(self.lr_diff_bpf) - 1) // 2
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Store last baseband for RDS processing
        self._last_baseband = None
        self._last_pilot = None

        # Tone controls
        self.bass_boost_enabled = True
        self.treble_boost_enabled = True
        self._setup_tone_filters()

        # Stereo blend settings
        self.stereo_blend_enabled = True
        self.stereo_blend_low = 10.0
        # Slightly higher upper threshold keeps low-SNR operation closer to
        # mono, improving the IHF/separation balance around ~15 dB RF SNR.
        self.stereo_blend_high = 35.0
        self._blend_factor = 1.0
        self._blend_tau_s = 0.12

        # L-R path gain calibration (helps recover separation lost to small
        # fixed gain mismatch between L+R and L-R decode paths).
        self.lr_gain_calibration_enabled = True
        # Empirical calibration from synthetic bench at 480k/48k.
        # Tuned for best aggregate separation (15-40 dB RF SNR sweep).
        self._lr_diff_gain = 1.0029

        # --- PLL state ---
        # Second-order Type 2 PLL for 19 kHz pilot tracking
        # Adaptive loop bandwidth with fixed damping.
        # - Acquisition: wide loop for fast pull-in
        # - Tracking: nominal low-noise loop
        # - Precision: narrow loop when quality is very high
        # Kp and Ki derived from standard PLL design equations:
        #   ωn = 2π·Bn / (ζ + 1/(4ζ)) ≈ 2π·30 / 1.4142 ≈ 133.3 rad/s
        #   Kp = 2·ζ·ωn / fs
        #   Ki = ωn² / fs²
        self._pll_omega0 = 2 * np.pi * 19000 / iq_sample_rate  # Nominal 19 kHz in rad/sample
        self._pll_phase = 0.0           # NCO phase accumulator
        self._pll_integrator = 0.0      # Loop filter integrator

        # PLL loop gains (adaptive Bn, fixed ζ, fs=iq_sample_rate)
        self._pll_zeta = 0.707
        self._pll_bw_acquire_hz = 100.0
        self._pll_bw_track_hz = 30.0
        # Precision mode narrows the loop under high quality for extra pilot
        # noise rejection while keeping tracking behavior stable.
        self._pll_bw_precision_hz = 20.0
        self._pll_precision_enter_quality_db = 34.0
        self._pll_precision_exit_quality_db = 26.0
        self._pll_loop_bandwidth_hz = 0.0
        self._pll_Kp = 0.0
        self._pll_Ki = 0.0
        self._set_pll_loop_bandwidth(self._pll_bw_acquire_hz)

        # Phase error IIR lowpass to remove 2×pilot (38 kHz) component.
        # Without this, the 38 kHz term in the PE biases the integrator and
        # creates ~28° steady-state phase offset at 38 kHz.
        # Cutoff ~5 kHz: well below 38 kHz for good rejection, well above
        # the 30 Hz loop bandwidth for negligible phase lag (<0.5°).
        # α = 2π·fc / (fs + 2π·fc) ≈ 0.065 at 480 kHz
        self._pll_pe_alpha = 2 * np.pi * 5000 / (iq_sample_rate + 2 * np.pi * 5000)
        self._pll_pe_filtered = 0.0

        # Lock detector: EMA of squared phase error
        self._pll_lock_alpha = 0.001
        self._pll_pe_avg = 1.0  # Start unlocked
        self._pll_locked = False
        self._pll_lock_threshold = 0.01    # Lock when pe_avg < this
        self._pll_unlock_threshold = 0.05  # Unlock when pe_avg > this

        # Normalized phase metric: low-passed pilot I/Q from the same NCO used
        # by the PLL. This reports a physically meaningful phase residual that
        # is far less sensitive to pilot amplitude than _pll_pe_avg.
        self._pll_iq_alpha = 2 * np.pi * 300 / (iq_sample_rate + 2 * np.pi * 300)
        self._pll_i_lp = 0.0
        self._pll_q_lp = 0.0
        self._pll_iq_mag = 0.0

        # Additional lock gates based on pilot I/Q envelope and normalized phase.
        # These prevent false-lock when phase-error energy is low but pilot energy
        # is absent or too weak to support reliable stereo decoding.
        self._pll_iq_lock_threshold = 0.003
        self._pll_iq_unlock_threshold = 0.0015
        self._pll_coherence_lock_threshold = 0.50
        self._pll_coherence_unlock_threshold = 0.35
        self._pll_phase_lock_threshold_deg = 30.0
        self._pll_phase_unlock_threshold_deg = 45.0

        # Per-stage profiling
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
        """Design high shelf biquad filter coefficients (Audio EQ Cookbook)."""
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
        self.bass_state_l = sp_signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = sp_signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_b, self.treble_a = self._design_high_shelf(3500, 3.0, fs)
        self.treble_state_l = sp_signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = sp_signal.lfilter_zi(self.treble_b, self.treble_a)

    @property
    def pilot_detected(self):
        """Returns True if 19 kHz pilot tone is detected (PLL locked)."""
        return self._pll_locked

    @property
    def snr_db(self):
        """Return the current SNR estimate in dB."""
        return self._snr_db

    @property
    def stereo_quality_db(self):
        """Return composite stereo quality metric in dB used for blending."""
        return self._stereo_quality_db

    @property
    def pilot_metric_db(self):
        """Return pilot envelope level relative to lock floor (dB)."""
        return self._pilot_metric_db

    @property
    def phase_penalty_db(self):
        """Return PLL phase error penalty (dB, <=0)."""
        return self._phase_penalty_db

    @property
    def coherence_penalty_db(self):
        """Return pilot/PLL coherence penalty (dB, <=0)."""
        return self._coherence_penalty_db

    @property
    def peak_amplitude(self):
        """Return peak amplitude (before limiting). >1.0 means limiter is active."""
        return self._peak_amplitude

    @property
    def stereo_blend_factor(self):
        """Return current stereo blend factor (0=mono, 1=full stereo)."""
        return self._blend_factor

    @property
    def rate_adjust(self):
        """Return current rate adjustment factor for adaptive resampling."""
        return self._rate_adjust

    @rate_adjust.setter
    def rate_adjust(self, value):
        """Set rate adjustment factor (0.99-1.01 typical range for clock drift)."""
        self._rate_adjust = max(0.98, min(1.02, value))

    @property
    def profile(self):
        """Return per-stage profiling data (EMA-smoothed microseconds)."""
        return dict(self._profile)

    @property
    def last_baseband(self):
        """Returns the last FM baseband signal for RDS processing."""
        return self._last_baseband

    @property
    def last_pilot(self):
        """Returns the last extracted 19 kHz pilot signal."""
        return self._last_pilot

    @property
    def pll_locked(self):
        """Returns True if the PLL is locked to the pilot tone."""
        return self._pll_locked

    @property
    def pll_backend(self):
        """Return active PLL loop backend ("python" or "numba")."""
        return self._pll_backend

    @property
    def pll_phase_error_rms(self):
        """Return legacy detector-metric phase error in degrees."""
        return np.degrees(np.sqrt(max(self._pll_pe_avg, 0.0)))

    @property
    def pll_phase_error_deg(self):
        """Return normalized pilot phase error (degrees), folded modulo 180°."""
        i_lp = self._pll_i_lp
        q_lp = self._pll_q_lp
        if i_lp == 0.0 and q_lp == 0.0:
            return 0.0
        phase = np.arctan2(q_lp, i_lp)
        # Pilot lock has 180° ambiguity; fold to [-90°, +90°].
        phase = ((phase + (np.pi / 2)) % np.pi) - (np.pi / 2)
        return abs(np.degrees(phase))

    @property
    def lr_diff_gain(self):
        """Return current fixed gain calibration applied to decoded L-R."""
        return self._lr_diff_gain

    @lr_diff_gain.setter
    def lr_diff_gain(self, value):
        """Set L-R gain calibration (small range to avoid instability)."""
        self._lr_diff_gain = float(np.clip(value, 0.98, 1.02))

    @property
    def pll_frequency_offset(self):
        """Return PLL frequency offset from nominal 19 kHz in Hz."""
        return self._pll_integrator * self.iq_sample_rate / (2 * np.pi)

    @property
    def pll_loop_bandwidth_hz(self):
        """Return current adaptive PLL loop bandwidth in Hz."""
        return self._pll_loop_bandwidth_hz

    def _prof(self, key, t0):
        """Record EMA-smoothed stage timing in microseconds."""
        elapsed = (time.perf_counter() - t0) * 1e6
        self._profile[key] = 0.9 * self._profile[key] + 0.1 * elapsed
        return time.perf_counter()

    def _set_pll_loop_bandwidth(self, bandwidth_hz):
        """Set PLL loop bandwidth and recompute loop gains."""
        bn_hz = float(max(1.0, bandwidth_hz))
        zeta = self._pll_zeta
        omega_n = 2 * np.pi * bn_hz / (zeta + 1 / (4 * zeta))
        self._pll_Kp = 2 * zeta * omega_n / self.iq_sample_rate
        self._pll_Ki = (omega_n ** 2) / (self.iq_sample_rate ** 2)
        self._pll_loop_bandwidth_hz = bn_hz

    def _update_pll_loop_bandwidth(self):
        """Adapt loop bandwidth based on lock state and stereo quality."""
        if not self._pll_locked:
            desired = self._pll_bw_acquire_hz
        else:
            in_precision = self._pll_loop_bandwidth_hz <= (self._pll_bw_precision_hz + 1e-9)
            if in_precision:
                precision_ok = self._stereo_quality_db >= self._pll_precision_exit_quality_db
            else:
                precision_ok = self._stereo_quality_db >= self._pll_precision_enter_quality_db
            desired = self._pll_bw_precision_hz if precision_ok else self._pll_bw_track_hz

        if abs(desired - self._pll_loop_bandwidth_hz) > 1e-9:
            narrowing = desired < self._pll_loop_bandwidth_hz
            self._set_pll_loop_bandwidth(desired)
            if narrowing:
                # Avoid carrying a wide-loop integrator bias into narrow-loop
                # tracking, which can manifest as residual phase offset.
                self._pll_integrator = 0.0

    def _pll_process(self, pilot_filtered):
        """
        Process pilot signal through PLL, return 38 kHz carrier.

        The PLL tracks the 19 kHz pilot and outputs cos(2·phase) which gives
        a phase-coherent 38 kHz carrier. The frequency doubling resolves the
        180° ambiguity: cos(2(θ+π)) = cos(2θ).

        Args:
            pilot_filtered: BPF-filtered pilot signal (numpy array)

        Returns:
            carrier_38k: 38 kHz carrier signal (numpy array)
        """
        # Local copies for speed in the tight loop
        phase = self._pll_phase
        integrator = self._pll_integrator
        pe_filt = self._pll_pe_filtered
        pe_avg = self._pll_pe_avg
        omega0 = self._pll_omega0
        Kp = self._pll_Kp
        Ki = self._pll_Ki
        pe_alpha = self._pll_pe_alpha
        iq_alpha = self._pll_iq_alpha
        lock_alpha = self._pll_lock_alpha
        i_lp = self._pll_i_lp
        q_lp = self._pll_q_lp
        pilot_filtered = np.asarray(pilot_filtered, dtype=np.float64)
        carrier_38k, phase, integrator, pe_filt, pe_avg, i_lp, q_lp = self._pll_kernel(
            pilot_filtered,
            phase,
            integrator,
            pe_filt,
            pe_avg,
            omega0,
            Kp,
            Ki,
            pe_alpha,
            iq_alpha,
            lock_alpha,
            i_lp,
            q_lp,
        )

        # Store state back
        self._pll_phase = phase
        self._pll_integrator = integrator
        self._pll_pe_filtered = pe_filt
        self._pll_pe_avg = pe_avg
        self._pll_i_lp = i_lp
        self._pll_q_lp = q_lp
        iq_mag = np.sqrt(i_lp * i_lp + q_lp * q_lp)
        self._pll_iq_mag = iq_mag

        pe_lock_ok = pe_avg < self._pll_lock_threshold
        pe_unlock = pe_avg > self._pll_unlock_threshold
        iq_lock_ok = iq_mag > self._pll_iq_lock_threshold
        iq_unlock = iq_mag < self._pll_iq_unlock_threshold
        pilot_level = max(float(self._pilot_level), 1e-12)
        coherence = np.clip(iq_mag / pilot_level, 0.0, 2.0)
        coherence_lock_ok = coherence > self._pll_coherence_lock_threshold
        coherence_unlock = coherence < self._pll_coherence_unlock_threshold

        # Lock detection with hysteresis
        if self._pll_locked:
            if pe_unlock or iq_unlock or coherence_unlock:
                self._pll_locked = False
        else:
            if pe_lock_ok and iq_lock_ok and coherence_lock_ok:
                self._pll_locked = True

        return carrier_38k

    def _resample_channels(self, left, right, stereo_allowed):
        """Resample decoded channels to audio sample rate."""
        ratio_eff = self._nominal_ratio * self._rate_adjust
        if stereo_allowed:
            left, right = self._resampler.process(ratio_eff, left, right)
        else:
            (left,) = self._resampler.process(ratio_eff, left)
            right = left
        return left, right

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

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_samples])
        self.last_sample = iq_samples[-1]
        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Store baseband for RDS processing
        self._last_baseband = baseband

        if profiling:
            t0 = self._prof('fm_demod', t0)

        # Extract pilot (19 kHz)
        pilot, self.pilot_bpf_state = sp_signal.lfilter(
            self.pilot_bpf, 1.0, baseband, zi=self.pilot_bpf_state
        )

        # Store pilot for RDS decoder
        self._last_pilot = pilot

        # Measure pilot level for SNR (still needed even with PLL lock detection)
        pilot_power = np.sqrt(np.mean(pilot ** 2))
        # Smooth pilot level with a continuous-time constant so behavior is
        # consistent across different input block sizes.
        pilot_alpha = _ema_alpha_from_tau(
            self._pilot_level_tau_s, len(baseband), self.iq_sample_rate
        )
        self._pilot_level += pilot_alpha * (pilot_power - self._pilot_level)

        if profiling:
            t0 = self._prof('pilot_bpf', t0)

        # Run PLL on pilot signal (always runs to maintain tracking)
        carrier_38k = self._pll_process(pilot)

        if profiling:
            t0 = self._prof('pll', t0)

        # Pilot-referenced SNR measurement
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = sp_signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)
            pilot_pwr = self._pilot_level ** 2
            noise_in_pilot_bw = noise_power * (self.pilot_bandwidth / self.noise_bandwidth)
            noise_alpha = _ema_alpha_from_tau(
                self._noise_power_tau_s, len(baseband), self.iq_sample_rate
            )
            noise_target = max(noise_in_pilot_bw, 1e-12)
            self._noise_power += noise_alpha * (noise_target - self._noise_power)
            if self._noise_power > 0 and pilot_pwr > 0:
                self._snr_db = 10 * np.log10(pilot_pwr / self._noise_power)

        if profiling:
            t0 = self._prof('noise_bpf', t0)

        # Composite stereo quality metric:
        # - Pilot envelope relative to lock floor
        # - PLL phase quality (normalized residual)
        # - Coherence between pilot RMS and PLL I/Q envelope
        # - Legacy pilot-vs-highband-noise SNR (when available)
        iq_mag = max(float(self._pll_iq_mag), 1e-12)
        pilot_level = max(float(self._pilot_level), 1e-12)
        pilot_floor = max(float(self._pll_iq_unlock_threshold) * 0.1, 1e-12)
        pilot_metric_db = 20.0 * np.log10(iq_mag / pilot_floor)

        phase_deg = float(self.pll_phase_error_deg)
        phase_span = max(
            float(self._pll_phase_unlock_threshold_deg - self._pll_phase_lock_threshold_deg),
            1e-9,
        )
        phase_score = np.clip(
            (self._pll_phase_unlock_threshold_deg - phase_deg) / phase_span,
            0.0,
            1.0,
        )
        phase_penalty_db = (phase_score - 1.0) * self._quality_phase_penalty_db

        coherence = np.clip(iq_mag / pilot_level, 0.0, 2.0)
        coherence_span = max(self._quality_coherence_high - self._quality_coherence_low, 1e-9)
        coherence_score = np.clip(
            (coherence - self._quality_coherence_low) / coherence_span,
            0.0,
            1.0,
        )
        coherence_penalty_db = (coherence_score - 1.0) * self._quality_coherence_penalty_db

        self._pilot_metric_db = pilot_metric_db
        self._phase_penalty_db = phase_penalty_db
        self._coherence_penalty_db = coherence_penalty_db
        quality_proxy_db = pilot_metric_db + phase_penalty_db + coherence_penalty_db
        if self.noise_bpf is not None:
            quality_target_db = min(self._snr_db, quality_proxy_db)
        else:
            quality_target_db = quality_proxy_db
        quality_target_db = float(np.clip(quality_target_db, -20.0, 60.0))
        quality_alpha = _ema_alpha_from_tau(
            self._stereo_quality_tau_s, len(baseband), self.iq_sample_rate
        )
        self._stereo_quality_db += quality_alpha * (quality_target_db - self._stereo_quality_db)
        self._update_pll_loop_bandwidth()

        # Use PLL lock for stereo gating; SNR blend handles mono transition
        # continuously to avoid threshold-induced steps.
        stereo_allowed = self._pll_locked and not self.force_mono

        # Extract L+R (mono, 0-15 kHz)
        lr_sum, self.lr_sum_lpf_state = sp_signal.lfilter(
            self.lr_sum_lpf, 1.0, baseband, zi=self.lr_sum_lpf_state
        )

        # Apply group delay compensation
        delay = self._lr_sum_delay
        delayed = np.empty_like(lr_sum)
        delayed[:delay] = self._lr_sum_delay_buf
        delayed[delay:] = lr_sum[:-delay] if delay > 0 else lr_sum
        self._lr_sum_delay_buf = lr_sum[-delay:].copy()
        lr_sum = delayed

        if profiling:
            t0 = self._prof('lr_sum_lpf', t0)

        if stereo_allowed:
            # Extract L-R subcarrier region (23-53 kHz)
            lr_diff_mod, self.lr_diff_bpf_state = sp_signal.lfilter(
                self.lr_diff_bpf, 1.0, baseband, zi=self.lr_diff_bpf_state
            )

            if profiling:
                t0 = self._prof('lr_diff_bpf', t0)

            # Demodulate L-R using PLL-derived 38 kHz carrier
            lr_diff_demod = lr_diff_mod * carrier_38k * 2  # *2 for DSB-SC gain

            # Lowpass filter the demodulated L-R
            lr_diff, self.lr_diff_lpf_state = sp_signal.lfilter(
                self.lr_diff_lpf, 1.0, lr_diff_demod, zi=self.lr_diff_lpf_state
            )

            if self.lr_gain_calibration_enabled:
                lr_diff = lr_diff * self._lr_diff_gain

            if profiling:
                t0 = self._prof('lr_diff_lpf', t0)

            # Matrix decode
            left_stereo = lr_sum + lr_diff
            right_stereo = lr_sum - lr_diff

            # Stereo blend based on composite stereo quality
            if self.stereo_blend_enabled:
                target_blend = (
                    (self._stereo_quality_db - self.stereo_blend_low)
                    / (self.stereo_blend_high - self.stereo_blend_low)
                )
                target_blend = max(0.0, min(1.0, target_blend))
                blend_alpha = _ema_alpha_from_tau(
                    self._blend_tau_s, len(baseband), self.iq_sample_rate
                )
                self._blend_factor += blend_alpha * (target_blend - self._blend_factor)
                left = self._blend_factor * left_stereo + (1.0 - self._blend_factor) * lr_sum
                right = self._blend_factor * right_stereo + (1.0 - self._blend_factor) * lr_sum
            else:
                left = left_stereo
                right = right_stereo
        else:
            left = lr_sum
            right = lr_sum
            self._blend_factor = 0.0

        # Resample to audio rate with adaptive rate control
        left, right = self._resample_channels(left, right, stereo_allowed)

        if profiling:
            t0 = self._prof('resample', t0)

        # De-emphasis
        left, self.deem_state_l = sp_signal.lfilter(
            self.deem_b, self.deem_a, left, zi=self.deem_state_l
        )
        right, self.deem_state_r = sp_signal.lfilter(
            self.deem_b, self.deem_a, right, zi=self.deem_state_r
        )

        if profiling:
            t0 = self._prof('deemphasis', t0)

        # Scale for headroom
        left = left * 0.65
        right = right * 0.65

        # Tone controls
        if self.bass_boost_enabled:
            left, self.bass_state_l = sp_signal.lfilter(
                self.bass_b, self.bass_a, left, zi=self.bass_state_l
            )
            right, self.bass_state_r = sp_signal.lfilter(
                self.bass_b, self.bass_a, right, zi=self.bass_state_r
            )
        if self.treble_boost_enabled:
            left, self.treble_state_l = sp_signal.lfilter(
                self.treble_b, self.treble_a, left, zi=self.treble_state_l
            )
            right, self.treble_state_r = sp_signal.lfilter(
                self.treble_b, self.treble_a, right, zi=self.treble_state_r
            )

        if profiling:
            t0 = self._prof('tone', t0)

        # Peak amplitude tracking
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)

        # Soft limiting + hard clip
        left = _soft_clip(left, threshold=0.8)
        right = _soft_clip(right, threshold=0.8)
        left = np.clip(left, -1.0, 1.0)
        right = np.clip(right, -1.0, 1.0)
        left = left.astype(np.float32)
        right = right.astype(np.float32)

        if profiling:
            self._prof('limiter', t0)
            elapsed_total = (time.perf_counter() - t_total) * 1e6
            self._profile['total'] = 0.9 * self._profile['total'] + 0.1 * elapsed_total

        return np.column_stack((left, right))

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._pilot_detected = False
        self._pilot_level = 0.0
        self._last_baseband = None

        # Reset filter states
        self.pilot_bpf_state = sp_signal.lfilter_zi(self.pilot_bpf, 1.0)
        self.lr_sum_lpf_state = sp_signal.lfilter_zi(self.lr_sum_lpf, 1.0)
        self.lr_diff_bpf_state = sp_signal.lfilter_zi(self.lr_diff_bpf, 1.0)
        self.lr_diff_lpf_state = sp_signal.lfilter_zi(self.lr_diff_lpf, 1.0)
        self.deem_state_l = sp_signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = sp_signal.lfilter_zi(self.deem_b, self.deem_a)
        if self.noise_bpf is not None:
            self.noise_bpf_state = sp_signal.lfilter_zi(self.noise_bpf, 1.0)

        # Reset group delay compensation buffer
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Reset resampler state
        self._resampler.reset()

        # Reset SNR state
        self._snr_db = 0.0
        self._noise_power = 1e-10
        self._stereo_quality_db = -20.0
        self._pilot_metric_db = 0.0
        self._phase_penalty_db = 0.0
        self._coherence_penalty_db = 0.0
        self._peak_amplitude = 0.0
        self._blend_factor = 1.0

        # Reset tone control filter states
        self.bass_state_l = sp_signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = sp_signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = sp_signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = sp_signal.lfilter_zi(self.treble_b, self.treble_a)

        # Reset PLL state
        self._pll_phase = 0.0
        self._pll_integrator = 0.0
        self._pll_pe_filtered = 0.0
        self._pll_pe_avg = 1.0  # Start unlocked
        self._pll_locked = False
        self._pll_i_lp = 0.0
        self._pll_q_lp = 0.0
        self._pll_iq_mag = 0.0
        self._set_pll_loop_bandwidth(self._pll_bw_acquire_hz)

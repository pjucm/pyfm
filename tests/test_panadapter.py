#!/usr/bin/env python3
r"""
Panadapter Demodulator Test Suite

Tests the panadapter's WBFMStereoDemodulator and NBFMDemodulator classes,
which wrap PLLStereoDecoder and standalone NBFM demodulation respectively.

Usage:
    python test_panadapter.py          # Run all tests with detailed output
    pytest test_panadapter.py -v       # Run with pytest (if installed)

WBFMStereoDemodulator Signal Flow:
    IQ (960 kHz) -> FIR decimation (2x) -> PLLStereoDecoder (480 kHz) -> Audio (48 kHz)
    Also supports frequency offset shifting and squelch gating.

NBFMDemodulator Signal Flow:
    IQ (480 kHz) -> Channel BPF -> Decimate to 32 kHz -> FM Demod -> Audio LPF -> Resample (48 kHz)

Validation Targets (WBFM):
    FM demod accuracy: Correlation > 0.99
    Audio SNR (clean): > 30 dB
    THD+N: < -30 dB (< 3.2% distortion)
    Stereo separation: > 30 dB
    L/R timing: < 5 samples at 48 kHz
    Frequency response: +/- 3 dB below 1 kHz

Validation Targets (NBFM):
    Tone detection: Correct frequency recovered
    Audio SNR (clean): > 25 dB
    Squelch: Blocks output when signal absent
"""

import numpy as np
from scipy import signal as scipy_signal
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import panadapter directly.
# Note: this test module depends on panadapter runtime deps (PyQt5/pyqtgraph/sounddevice).
import panadapter

NBFMDemodulator = panadapter.NBFMDemodulator
WBFMStereoDemodulator = panadapter.WBFMStereoDemodulator
PLLStereoDecoder = panadapter.PLLStereoDecoder

AUDIO_SAMPLE_RATE = panadapter.AUDIO_SAMPLE_RATE
NBFM_DEVIATION = panadapter.NBFM_DEVIATION
WBFM_DEVIATION = panadapter.WBFM_DEVIATION
ICOM_SAMPLE_RATE_FM = panadapter.ICOM_SAMPLE_RATE_FM
ICOM_SAMPLE_RATE_WEATHER = panadapter.ICOM_SAMPLE_RATE_WEATHER
PRIMARY_WBFM_DECODER = panadapter.DEFAULT_STEREO_DECODER


# =============================================================================
# Helper Functions
# =============================================================================

def process_with_settling(demod, iq, block_size=8192):
    """
    Process IQ samples in blocks through a panadapter demodulator.

    Args:
        demod: WBFMStereoDemodulator or NBFMDemodulator instance
        iq: Complex I/Q samples
        block_size: Block size for processing

    Returns:
        Concatenated audio output array
    """
    audio_chunks = []
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) > 0:
            audio = demod.process(block)
            if audio is not None:
                audio_chunks.append(audio)

    if not audio_chunks:
        return np.zeros((0, 2), dtype=np.float32)

    return np.vstack(audio_chunks) if audio_chunks[0].ndim == 2 else np.concatenate(audio_chunks)


def process_wbfm_with_baseband(demod, iq, block_size=8192):
    """
    Process IQ through WBFM demodulator and collect FM-demod baseband.

    Returns:
        tuple: (audio_out, baseband_out)
    """
    audio_chunks = []
    baseband_chunks = []

    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) == 0:
            continue

        audio = demod.process(block)
        if audio is None:
            continue

        audio_chunks.append(audio)
        if demod.last_baseband is not None and len(demod.last_baseband) > 0:
            baseband_chunks.append(np.array(demod.last_baseband, copy=True))

    if audio_chunks:
        audio_out = np.vstack(audio_chunks)
    else:
        audio_out = np.zeros((0, 2), dtype=np.float32)

    if baseband_chunks:
        baseband_out = np.concatenate(baseband_chunks)
    else:
        baseband_out = np.array([], dtype=np.float64)

    return audio_out, baseband_out


# =============================================================================
# Signal Generation Functions
# =============================================================================

def generate_test_tone(freq, duration, sample_rate, amplitude=1.0):
    """Generate a pure sine wave test tone."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_fm_stereo_multiplex(left, right, sample_rate, pilot_amplitude=0.09,
                                  include_pilot=True, subcarrier_phase='neg_cos'):
    """
    Generate FM stereo multiplex baseband signal.

    FM Multiplex structure:
    - 0-15 kHz: L+R (mono sum, scaled by 0.9 for pilot headroom)
    - 19 kHz: Pilot tone (9% of max deviation)
    - 23-53 kHz: L-R DSB-SC modulated onto 38 kHz carrier
    """
    n = len(left)
    t = np.arange(n) / sample_rate

    lr_sum = (left + right) / 2
    lr_diff = (left - right) / 2

    if include_pilot:
        pilot = pilot_amplitude * np.sin(2 * np.pi * 19000 * t)
    else:
        pilot = np.zeros(n)

    if subcarrier_phase == 'neg_cos':
        carrier_38k = -np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'cos':
        carrier_38k = np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'sin':
        carrier_38k = np.sin(2 * np.pi * 38000 * t)
    else:
        raise ValueError(f"Unknown subcarrier_phase: {subcarrier_phase}")

    lr_diff_mod = lr_diff * carrier_38k
    multiplex = lr_sum * 0.9 + pilot + lr_diff_mod * 0.9

    return multiplex


def fm_modulate(baseband, sample_rate, deviation=75000):
    """FM modulate baseband signal to complex I/Q."""
    dt = 1.0 / sample_rate
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


def nbfm_modulate(audio, sample_rate, deviation=5000):
    """NBFM modulate audio signal to complex I/Q."""
    dt = 1.0 / sample_rate
    phase = 2 * np.pi * deviation * np.cumsum(audio) * dt
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


# =============================================================================
# Measurement Utilities
# =============================================================================

def goertzel_power(x, target_freq, sample_rate):
    """Compute power at a specific frequency using Goertzel algorithm."""
    n = len(x)
    k = int(0.5 + n * target_freq / sample_rate)
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    power = power / (n * n / 2)
    return power


def fft_bin_band_power(x, target_freq, sample_rate, side_bins=1):
    """
    Measure tone power using Hann-windowed FFT around the nearest bin.

    Summing ±side_bins around the target bin reduces sensitivity to small
    frequency offsets and block boundary phase.
    """
    n = len(x)
    if n <= 0:
        return 0.0

    window = np.hanning(n)
    x_windowed = x * window
    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    target_bin = int(np.argmin(np.abs(freqs - target_freq)))
    lo = max(0, target_bin - side_bins)
    hi = min(len(fft), target_bin + side_bins + 1)

    return float(np.sum(np.abs(fft[lo:hi]) ** 2))


def classify_stereo_phase_result(left_power, right_power):
    """Classify stereo phase behavior from left/right tone power distribution."""
    total = left_power + right_power
    if total < 1e-12:
        return "FAILED (no signal)"
    if left_power / total > 0.9:
        return "CORRECT (signal in L)"
    if right_power / total > 0.9:
        return "INVERTED (signal in R)"
    if abs(left_power - right_power) / max(left_power, right_power, 1e-12) < 0.5:
        return "FAILED (no separation)"
    return "PARTIAL"


def measure_snr(x, signal_freq, sample_rate, signal_bw=100):
    """Measure SNR of a signal with a known tone frequency."""
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    signal_mask = np.abs(freqs - signal_freq) <= signal_bw / 2
    signal_power = np.sum(power_spectrum[signal_mask])

    noise_mask = ~signal_mask
    noise_power = np.sum(power_spectrum[noise_mask])

    if noise_power <= 0:
        return np.inf

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def measure_thd_n(x, fundamental_freq, sample_rate, n_harmonics=5):
    """Measure Total Harmonic Distortion + Noise (THD+N)."""
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    fundamental_mask = np.abs(freqs - fundamental_freq) <= 50
    fundamental_power = np.sum(power_spectrum[fundamental_mask])

    harmonic_power = 0.0
    for h in range(2, n_harmonics + 1):
        harm_freq = fundamental_freq * h
        if harm_freq >= sample_rate / 2:
            break
        harm_mask = np.abs(freqs - harm_freq) <= 50
        harmonic_power += np.sum(power_spectrum[harm_mask])

    total_power = np.sum(power_spectrum)
    noise_power = total_power - fundamental_power - harmonic_power

    thd_n_power = harmonic_power + noise_power
    if fundamental_power <= 0:
        return 0.0

    thd_n_db = 10 * np.log10(thd_n_power / fundamental_power)
    return thd_n_db


def design_a_weighting(fs):
    """
    Design an IEC 61672 A-weighting filter.

    Analog prototype has 4 zeros at s=0 and poles at:
        20.598997 Hz (×2), 107.65265 Hz, 737.86223 Hz, 12194.217 Hz (×2)

    Returns:
        SOS array for use with scipy.signal.sosfilt, normalized to 0 dB at 1 kHz
    """
    z = [0, 0, 0, 0]
    p = [-2 * np.pi * 20.598997,
         -2 * np.pi * 20.598997,
         -2 * np.pi * 107.65265,
         -2 * np.pi * 737.86223,
         -2 * np.pi * 12194.217,
         -2 * np.pi * 12194.217]
    k = (2 * np.pi * 12194.217) ** 2

    zd, pd, kd = scipy_signal.bilinear_zpk(z, p, k, fs)
    sos = scipy_signal.zpk2sos(zd, pd, kd)

    w, h = scipy_signal.sosfreqz(sos, worN=[2 * np.pi * 1000 / fs])
    sos[0, :3] /= np.abs(h[0])

    return sos


def find_step_crossing(x, threshold=0.5):
    """Find the sample index where signal crosses threshold."""
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        return -1

    x_norm = (x - x_min) / (x_max - x_min)

    for i in range(len(x_norm) - 1):
        if x_norm[i] < threshold <= x_norm[i + 1]:
            frac = (threshold - x_norm[i]) / (x_norm[i + 1] - x_norm[i])
            return i + frac

    return -1


# =============================================================================
# WBFM Test Functions (via WBFMStereoDemodulator)
# =============================================================================

def create_wbfm_demod(iq_rate=ICOM_SAMPLE_RATE_FM, audio_rate=AUDIO_SAMPLE_RATE,
                       disable_deemphasis=True, stereo_decoder=None):
    """Create a WBFMStereoDemodulator configured for testing."""
    decoder_name = PRIMARY_WBFM_DECODER if stereo_decoder is None else stereo_decoder
    demod = WBFMStereoDemodulator(
        input_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        stereo_decoder=decoder_name
    )
    if disable_deemphasis:
        # Disable de-emphasis on the inner decoder for clean measurements
        # Attributes: deem_b, deem_a, deem_state_l, deem_state_r
        dec = demod.stereo_decoder
        dec.deem_b = np.array([1.0])
        dec.deem_a = np.array([1.0, 0.0])
        dec.deem_state_l = np.zeros(1)
        dec.deem_state_r = np.zeros(1)
    demod.stereo_decoder.bass_boost_enabled = False
    demod.stereo_decoder.treble_boost_enabled = False
    demod.stereo_decoder.stereo_blend_enabled = False
    return demod


def expected_decoder_name(requested_decoder):
    """Return the effective decoder name after panadapter fallback logic."""
    if requested_decoder == 'pll' and PLLStereoDecoder is None:
        return 'squaring'
    return requested_decoder


def test_wbfm_primary_decoder_selected():
    """
    Verify WBFMStereoDemodulator selects the intended primary decoder path.

    This keeps coverage on the production-default WBFM decode path.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Primary Decoder Selection")
    print("=" * 60)
    demod = create_wbfm_demod(stereo_decoder=PRIMARY_WBFM_DECODER)
    expected = expected_decoder_name(PRIMARY_WBFM_DECODER)
    selected = demod.stereo_decoder_name == expected

    print(f"  Requested decoder: {PRIMARY_WBFM_DECODER}")
    print(f"  Expected decoder:  {expected}")
    print(f"  Active decoder:    {demod.stereo_decoder_name}")
    print(f"  Decoder class:     {type(demod.stereo_decoder).__name__}")

    passed = selected
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, f"Expected decoder '{expected}', got '{demod.stereo_decoder_name}'"
    return passed


def test_wbfm_demod_accuracy():
    """
    Test FM demodulation accuracy through WBFMStereoDemodulator.

    Generates known baseband, FM modulates at 960 kHz, demodulates through
    the full decimation + stereo decoder chain.

    Pass criteria: Correlation > 0.99, amplitude error < 2%
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Demodulation Accuracy")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM  # 960 kHz
    duration = 0.1
    test_freq = 1000

    baseband_in = generate_test_tone(test_freq, duration, iq_rate, amplitude=0.5)
    iq = fm_modulate(baseband_in, iq_rate, deviation=WBFM_DEVIATION)

    # Use WBFMStereoDemodulator directly with 480 kHz output so the inner decoder
    # runs without resampling and we can compare baseband fidelity.
    demod = create_wbfm_demod(
        iq_rate=iq_rate,
        audio_rate=iq_rate // 2,
        stereo_decoder=PRIMARY_WBFM_DECODER
    )
    _, baseband_out = process_wbfm_with_baseband(demod, iq)
    assert len(baseband_out) > 0, "No baseband data produced by WBFMStereoDemodulator.process()"

    # Generate reference baseband at decimated rate
    if demod.decimation > 1:
        baseband_filtered = scipy_signal.lfilter(demod.decim_filter, 1.0, baseband_in)
        baseband_ref = baseband_filtered[::demod.decimation]
    else:
        baseband_ref = baseband_in

    # Trim to same length
    min_len = min(len(baseband_ref), len(baseband_out))
    skip = 2000
    assert min_len > 2 * skip, f"Insufficient baseband length for accuracy check (len={min_len})"
    baseband_ref_trimmed = baseband_ref[skip:min_len - skip]
    baseband_out_trimmed = baseband_out[skip:min_len - skip]

    correlation = np.corrcoef(baseband_ref_trimmed, baseband_out_trimmed)[0, 1]

    rms_in = np.sqrt(np.mean(baseband_ref_trimmed ** 2))
    rms_out = np.sqrt(np.mean(baseband_out_trimmed ** 2))
    amplitude_ratio = rms_out / rms_in
    amplitude_error = abs(1.0 - amplitude_ratio) * 100

    print(f"  Decoder under test: {demod.stereo_decoder_name}")
    print(f"  IQ rate: {iq_rate/1000:.0f} kHz -> decimated to {demod.decimated_rate/1000:.0f} kHz")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Amplitude ratio: {amplitude_ratio:.4f}")
    print(f"  Amplitude error: {amplitude_error:.2f}%")

    # Relaxed vs test_pjfm.py (0.999): the anti-aliasing decimation filter
    # introduces minor phase distortion that lowers correlation slightly
    passed = correlation > 0.99 and amplitude_error < 2.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, (
        f"Demod accuracy failed: correlation={correlation:.6f}, amplitude_error={amplitude_error:.2f}%"
    )
    return passed


def test_wbfm_audio_snr():
    """
    Test decoded audio SNR through WBFMStereoDemodulator.

    Pass criteria: SNR > 30 dB
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Audio SNR (Clean Input)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * test_freq * t)
    right = 0.5 * np.sin(2 * np.pi * test_freq * t)

    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Keep de-emphasis enabled for system-level SNR (realistic panadapter config)
    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, disable_deemphasis=False)
    audio = process_with_settling(demod, iq)

    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]

    snr = measure_snr(left_out, test_freq, audio_rate)

    print(f"  IQ rate: {iq_rate/1000:.0f} kHz, decimation: {demod.decimation}x")
    print(f"  Test frequency: {test_freq} Hz")
    print(f"  Measured SNR: {snr:.1f} dB")

    passed = snr > 30
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >30 dB)")
    assert passed, f"WBFM audio SNR below target: {snr:.2f} dB"
    return passed


def test_wbfm_thd_n():
    """
    Test THD+N through WBFMStereoDemodulator.

    Target: THD+N < -30 dB
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM THD+N")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Keep de-emphasis enabled for system-level THD+N measurement
    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, disable_deemphasis=False)
    audio = process_with_settling(demod, iq)

    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]

    thd_n = measure_thd_n(left_out, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  THD+N: {thd_n:.1f} dB ({10**(thd_n/20)*100:.2f}%)")

    passed = thd_n < -30
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: < -30 dB)")
    assert passed, f"WBFM THD+N above target: {thd_n:.2f} dB"
    return passed


def test_wbfm_mono_decode():
    """
    Test mono decoding (no pilot tone) through WBFMStereoDemodulator.

    Uses pilot-squaring decoder to match legacy pjfm validation targets.

    Pass criteria: no pilot detect, L/R correlation > 0.999, correct frequency
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Mono Decoding (No Pilot)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.3
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq = fm_modulate(multiplex, iq_rate)

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='squaring')
    audio = process_with_settling(demod, iq)

    pilot_detected = demod.pilot_detected

    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]
    right_out = audio[skip:-skip, 1]

    lr_corr = np.corrcoef(left_out, right_out)[0, 1]

    left_power_1k = goertzel_power(left_out, test_freq, audio_rate)
    left_power_total = np.mean(left_out ** 2)
    freq_ratio = left_power_1k / left_power_total if left_power_total > 0 else 0

    print(f"  Pilot detected: {pilot_detected}")
    print(f"  L/R correlation: {lr_corr:.6f}")
    print(f"  1 kHz power ratio: {freq_ratio:.4f}")

    passed = not pilot_detected and lr_corr > 0.999 and freq_ratio > 0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, (
        f"WBFM mono decode failed: pilot={pilot_detected}, corr={lr_corr:.6f}, freq_ratio={freq_ratio:.4f}"
    )
    return passed


def test_wbfm_stereo_decode():
    """
    Test stereo decoding with different tones on L and R.

    Uses pilot-squaring decoder to match legacy pjfm validation targets.

    Pass criteria: Pilot detected, channel separation > 20 dB
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Stereo Decoding (1 kHz L, 2 kHz R)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 1.0
    left_freq = 1000
    right_freq = 2000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * left_freq * t)
    right = 0.5 * np.sin(2 * np.pi * right_freq * t)

    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='squaring')
    audio = process_with_settling(demod, iq)

    pilot_detected = demod.pilot_detected

    skip = len(audio) // 2
    left_out = audio[skip:, 0]
    right_out = audio[skip:, 1]

    # Use windowed FFT band power to avoid apparent asymmetry from exact-bin Goertzel.
    left_1k_power = fft_bin_band_power(left_out, left_freq, audio_rate, side_bins=1)
    left_2k_power = fft_bin_band_power(left_out, right_freq, audio_rate, side_bins=1)
    right_1k_power = fft_bin_band_power(right_out, left_freq, audio_rate, side_bins=1)
    right_2k_power = fft_bin_band_power(right_out, right_freq, audio_rate, side_bins=1)

    left_separation = 10 * np.log10(left_1k_power / (left_2k_power + 1e-12))
    right_separation = 10 * np.log10(right_2k_power / (right_1k_power + 1e-12))

    print(f"  Pilot detected: {pilot_detected}")
    print("  Separation metric: Hann FFT bin power (target ±1 bin)")
    print(f"  Left channel separation: {left_separation:.1f} dB")
    print(f"  Right channel separation: {right_separation:.1f} dB")

    min_separation = min(left_separation, right_separation)

    passed = pilot_detected and min_separation > 20
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >20 dB separation)")
    assert passed, (
        f"WBFM stereo decode failed: pilot={pilot_detected}, min_separation={min_separation:.2f} dB"
    )
    return passed


def test_wbfm_stereo_separation():
    """
    Test stereo separation across frequency range.

    Uses pilot-squaring decoder to match legacy pjfm validation targets.

    Pass criteria: >30 dB separation at all tested frequencies
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Stereo Separation vs Frequency")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 1.0

    # 12 kHz excluded: its L-R DSB-SC sideband (38±12 = 26/50 kHz) is
    # attenuated by the 31-tap decimation filter's transition band
    test_freqs = [100, 1000, 5000, 10000]

    all_passed = True
    separations = []

    for freq in test_freqs:
        demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='squaring')

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        left = 0.5 * np.sin(2 * np.pi * freq * t)
        right = np.zeros(n_samples)

        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = process_with_settling(demod, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = fft_bin_band_power(left_out, freq, audio_rate, side_bins=1)
        right_power = fft_bin_band_power(right_out, freq, audio_rate, side_bins=1)

        separation = 10 * np.log10(left_power / (right_power + 1e-12))
        separations.append(separation)

        print(f"  {freq:5d} Hz: {separation:5.1f} dB", end="")
        if separation < 30:
            print(" (FAIL)")
            all_passed = False
        else:
            print(" (PASS)")

    min_sep = min(separations)
    print(f"\n  Minimum separation: {min_sep:.1f} dB")
    print(f"  Result: {'PASS' if all_passed else 'FAIL'} (target: >30 dB at all frequencies)")
    assert all_passed, f"WBFM stereo separation below target (min={min_sep:.2f} dB)"
    return all_passed


def test_wbfm_subcarrier_phase():
    """
    Test pilot-squaring decoder sensitivity to subcarrier phase.

    neg_cos: Should work correctly
    cos: Should give inverted stereo
    sin: Should fail (90 degree phase error)
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Subcarrier Phase Sensitivity")
    print("=" * 60)
    print()
    print("  The pilot-squaring decoder produces -cos(2wt) carrier")
    print("  This test verifies correct behavior with different TX phases")
    print()

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 1.0

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate

    left = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right = np.zeros(n_samples)

    results = {}

    for phase_name in ['neg_cos', 'cos', 'sin']:
        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase=phase_name)
        iq = fm_modulate(multiplex, iq_rate)

        # This phase-relationship test is specific to pilot-squaring behavior.
        demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='squaring')
        audio = process_with_settling(demod, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        # Use windowed FFT band power so classification is insensitive to
        # small tone/bin mismatches after resampling.
        left_power = fft_bin_band_power(left_out, 1000, audio_rate, side_bins=1)
        right_power = fft_bin_band_power(right_out, 1000, audio_rate, side_bins=1)
        status = classify_stereo_phase_result(left_power, right_power)

        results[phase_name] = (left_power, right_power, status)
        print(f"  TX={phase_name:8s}: L={left_power:.6f}, R={right_power:.6f} -> {status}")

    print()

    neg_cos_correct = 'CORRECT' in results['neg_cos'][2]
    sin_fails = 'FAILED' in results['sin'][2]

    passed = neg_cos_correct and sin_fails
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print(f"  (neg_cos should work, sin should fail due to 90 degree phase error)")
    assert passed, f"Unexpected phase sensitivity results: {results}"
    return passed


def test_wbfm_group_delay():
    """
    Test L/R timing alignment through WBFMStereoDemodulator.

    Uses pilot-squaring decoder to match legacy pjfm validation targets.

    Pass criteria: L/R timing difference < 5 samples at 48 kHz
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Group Delay Alignment (L/R Timing)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.1

    n_samples = int(duration * iq_rate)

    step_point = n_samples // 2
    step = np.zeros(n_samples)
    step[step_point:] = 0.5

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='squaring')

    multiplex = generate_fm_stereo_multiplex(step, step, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)
    audio = process_with_settling(demod, iq)

    left = audio[:, 0]
    right = audio[:, 1]

    left_crossing = find_step_crossing(left)
    right_crossing = find_step_crossing(right)

    if left_crossing < 0 or right_crossing < 0:
        raise AssertionError("Could not find step crossings")

    lr_diff_samples = abs(left_crossing - right_crossing)
    lr_diff_us = lr_diff_samples * (1e6 / audio_rate)

    print(f"  Left crossing at: {left_crossing:.2f} samples")
    print(f"  Right crossing at: {right_crossing:.2f} samples")
    print(f"  L/R crossing difference: {lr_diff_samples:.2f} samples ({lr_diff_us:.0f} us)")

    passed = lr_diff_samples < 5
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: <5 samples)")
    assert passed, f"WBFM group delay mismatch too large: {lr_diff_samples:.2f} samples"
    return passed


def test_wbfm_frequency_response():
    """
    Test frequency response through WBFMStereoDemodulator.

    Pass criteria: +/- 3 dB below 1 kHz
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Frequency Response")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.2

    test_freqs = [100, 200, 500, 1000, 2000, 5000, 8000, 10000, 12000, 14000]

    responses = []

    for freq in test_freqs:
        demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        mono = 0.5 * np.sin(2 * np.pi * freq * t)

        multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = process_with_settling(demod, iq)

        skip = int(0.05 * audio_rate)
        left_out = audio[skip:-skip, 0]

        power = goertzel_power(left_out, freq, audio_rate)
        responses.append(power)

    ref_idx = test_freqs.index(1000)
    ref_power = responses[ref_idx]

    print(f"  {'Freq':>6s}  {'Response':>10s}  {'Relative':>10s}")
    print(f"  {'----':>6s}  {'--------':>10s}  {'--------':>10s}")

    max_deviation = 0
    for freq, power in zip(test_freqs, responses):
        rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
        print(f"  {freq:6d}  {power:10.6f}  {rel_db:+7.2f} dB")
        max_deviation = max(max_deviation, abs(rel_db))

    low_freq_max_dev = 0
    for freq, power in zip(test_freqs, responses):
        if freq <= 1000:
            rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
            low_freq_max_dev = max(low_freq_max_dev, abs(rel_db))

    print(f"\n  Max deviation (all): {max_deviation:.2f} dB")
    print(f"  Max deviation (<= 1 kHz): {low_freq_max_dev:.2f} dB")

    passed = low_freq_max_dev < 3.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: +-3 dB below 1 kHz)")
    assert passed, f"WBFM low-frequency response deviation too high: {low_freq_max_dev:.2f} dB"
    return passed


def test_wbfm_snr_with_noise():
    """
    Test WBFM decoder behavior with noisy input.

    Verifies graceful degradation with stereo blend.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM SNR with Noisy Input")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5

    input_snr_levels = [40, 30, 20, 10]
    test_freq = 1000

    print(f"  {'Input SNR':>12s}  {'Output SNR':>12s}  {'Pilot':>8s}  {'Blend':>8s}")
    print(f"  {'--------':>12s}  {'----------':>12s}  {'-----':>8s}  {'-----':>8s}")

    all_passed = True
    rng = np.random.default_rng(0)

    for input_snr in input_snr_levels:
        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        left = 0.5 * np.sin(2 * np.pi * test_freq * t)
        right = 0.5 * np.sin(2 * np.pi * test_freq * t)

        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
        iq_clean = fm_modulate(multiplex, iq_rate)

        signal_power = np.mean(np.abs(iq_clean) ** 2)
        noise_power = signal_power / (10 ** (input_snr / 10))
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        )
        iq_noisy = iq_clean + noise.astype(np.complex64)

        demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, disable_deemphasis=False)
        demod.stereo_decoder.stereo_blend_enabled = True

        audio = process_with_settling(demod, iq_noisy)

        pilot_detected = demod.pilot_detected
        blend_factor = demod.stereo_blend_factor

        skip = int(0.1 * audio_rate)
        left_out = audio[skip:-skip, 0]
        output_snr = measure_snr(left_out, test_freq, audio_rate)

        print(f"  {input_snr:10d} dB  {output_snr:10.1f} dB  {'Yes' if pilot_detected else 'No':>8s}  {blend_factor:8.2f}")

        if input_snr >= 40 and output_snr < 30:
            all_passed = False

    print(f"\n  Result: {'PASS' if all_passed else 'FAIL'} (graceful degradation)")
    assert all_passed, "WBFM noisy-input behavior failed minimum high-SNR output target"
    return all_passed


def test_wbfm_ihf_snr():
    """
    Test IHF/EIA-style audio-referenced SNR through WBFMStereoDemodulator.

    Measures A-weighted, de-emphasized audio SNR — the metric used in hi-fi
    tuner specifications.  De-emphasis + A-weighting adds ~25-30 dB over the
    pilot-referenced RF measurement.

    Tests both squaring and PLL decoder paths in stereo and mono modes.

    Pass thresholds: stereo > 35 dB, mono > 35 dB.  The WBFMStereoDemodulator's
    decimation filter limits the processing floor to ~38 dB; the test confirms
    the decoder achieves this with both A-weighting and de-emphasis active.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM IHF/EIA-Style SNR (A-weighted, de-emphasized)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM  # 960 kHz
    audio_rate = AUDIO_SAMPLE_RATE  # 48 kHz
    duration = 1.0
    test_freq = 1000
    rf_snr_db = 40

    rng = np.random.default_rng(42)

    # Generate stereo 1 kHz tone at 0.5 amplitude
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t)

    multiplex = generate_fm_stereo_multiplex(tone, tone, iq_rate, subcarrier_phase='neg_cos')
    iq_clean = fm_modulate(multiplex, iq_rate)

    # Add AWGN at specified RF SNR
    sig_power = np.mean(np.abs(iq_clean) ** 2)
    noise_power = sig_power / (10 ** (rf_snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    iq_noisy = iq_clean + noise.astype(np.complex64)

    # Design A-weighting filter at audio rate
    a_weight_sos = design_a_weighting(audio_rate)

    def measure_ihf_snr(audio_channel):
        """A-weight audio and measure tone-vs-residual SNR."""
        weighted = scipy_signal.sosfilt(a_weight_sos, audio_channel)
        n = len(weighted)
        window = np.hanning(n)
        x_w = weighted * window

        fft_out = np.fft.rfft(x_w)
        freqs = np.fft.rfftfreq(n, 1.0 / audio_rate)
        power_spec = np.abs(fft_out) ** 2 / n

        tone_mask = np.abs(freqs - test_freq) <= 50
        tone_power = np.sum(power_spec[tone_mask])
        residual_power = np.sum(power_spec[~tone_mask])

        if residual_power <= 0:
            return np.inf
        return 10 * np.log10(tone_power / residual_power)

    print(f"\n  {'Decoder':<10s}  {'Mode':<8s}  {'IHF SNR':>10s}  {'Pilot SNR':>10s}")
    print(f"  {'-------':<10s}  {'----':<8s}  {'-------':>10s}  {'---------':>10s}")

    all_passed = True

    for decoder_name in ['squaring', 'pll']:
        # --- Stereo ---
        demod_stereo = create_wbfm_demod(
            iq_rate=iq_rate, audio_rate=audio_rate,
            disable_deemphasis=False, stereo_decoder=decoder_name
        )
        audio_stereo = process_with_settling(demod_stereo, iq_noisy)
        skip = int(0.1 * audio_rate)
        left_stereo = audio_stereo[skip:-skip, 0]
        stereo_ihf = measure_ihf_snr(left_stereo)
        pilot_snr = demod_stereo.stereo_decoder.snr_db if hasattr(demod_stereo.stereo_decoder, 'snr_db') else 0.0

        print(f"  {decoder_name:<10s}  {'stereo':<8s}  {stereo_ihf:8.1f} dB  {pilot_snr:8.1f} dB")

        if stereo_ihf <= 35:
            all_passed = False

        # --- Mono ---
        demod_mono = create_wbfm_demod(
            iq_rate=iq_rate, audio_rate=audio_rate,
            disable_deemphasis=False, stereo_decoder=decoder_name
        )
        demod_mono.stereo_decoder.force_mono = True
        audio_mono = process_with_settling(demod_mono, iq_noisy)
        left_mono = audio_mono[skip:-skip, 0]
        mono_ihf = measure_ihf_snr(left_mono)

        print(f"  {decoder_name:<10s}  {'mono':<8s}  {mono_ihf:8.1f} dB")

        if mono_ihf <= 35:
            all_passed = False

    print(f"\n  Result: {'PASS' if all_passed else 'FAIL'} (stereo >35 dB, mono >35 dB)")
    assert all_passed, "WBFM IHF/EIA SNR below threshold"
    return all_passed


def test_wbfm_freq_offset():
    """
    Test WBFMStereoDemodulator with a tuning offset.

    Generates signal at an offset from center, uses set_tuned_offset() to tune.

    Pass criteria: Correct tone recovered, SNR > 25 dB
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Frequency Offset Tuning")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000
    offset_hz = 100000  # 100 kHz offset

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Generate FM at baseband
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq_baseband = fm_modulate(multiplex, iq_rate)

    # Shift signal to offset frequency
    shift = np.exp(2j * np.pi * offset_hz * t)
    iq_offset = (iq_baseband * shift).astype(np.complex64)

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    demod.set_tuned_offset(offset_hz)

    audio = process_with_settling(demod, iq_offset)

    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]

    power_1k = goertzel_power(left_out, test_freq, audio_rate)
    total_power = np.mean(left_out ** 2)
    freq_ratio = power_1k / total_power if total_power > 0 else 0

    snr = measure_snr(left_out, test_freq, audio_rate)

    print(f"  Offset: {offset_hz/1000:.0f} kHz")
    print(f"  1 kHz power ratio: {freq_ratio:.4f}")
    print(f"  SNR: {snr:.1f} dB")

    passed = freq_ratio > 0.3 and snr > 25
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: ratio>0.3, SNR>25 dB)")
    assert passed, f"WBFM offset tuning failed: freq_ratio={freq_ratio:.4f}, snr={snr:.2f} dB"
    return passed


def test_wbfm_squelch():
    """
    Test WBFMStereoDemodulator squelch behavior.

    Pass criteria: Returns None when signal below squelch threshold
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM Squelch")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    n_samples = 8192

    demod = create_wbfm_demod(iq_rate=iq_rate)
    demod.set_squelch(-20)  # Set squelch at -20 dB

    # Very weak noise-only signal (well below squelch)
    rng = np.random.default_rng(0)
    noise = 1e-6 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
    result = demod.process(noise)

    squelched = result is None
    print(f"  Squelch level: -20 dB")
    print(f"  Noise-only result: {'squelched (None)' if squelched else 'NOT squelched'}")

    # Strong signal (above squelch)
    demod.reset()
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * 1000 * t)
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq_strong = fm_modulate(multiplex, iq_rate)
    result_strong = demod.process(iq_strong)

    unsquelched = result_strong is not None
    print(f"  Strong signal result: {'audio present' if unsquelched else 'SQUELCHED (unexpected)'}")

    passed = squelched and unsquelched
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, f"WBFM squelch behavior incorrect: squelched={squelched}, unsquelched={unsquelched}"
    return passed


# =============================================================================
# PLL-Specific WBFM Coverage
# =============================================================================

def test_wbfm_pll_mono_decode():
    """
    PLL path mono decode coverage (no pilot tone).

    Pass criteria: mono output (L/R correlation > 0.999) and 1 kHz recovery.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM PLL Mono Decode (No Pilot)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.3
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq = fm_modulate(multiplex, iq_rate)

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='pll')
    audio = process_with_settling(demod, iq)

    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]
    right_out = audio[skip:-skip, 1]

    lr_corr = np.corrcoef(left_out, right_out)[0, 1]
    left_power_1k = goertzel_power(left_out, test_freq, audio_rate)
    left_power_total = np.mean(left_out ** 2)
    freq_ratio = left_power_1k / left_power_total if left_power_total > 0 else 0

    print(f"  Pilot detected: {demod.pilot_detected}")
    print(f"  L/R correlation: {lr_corr:.6f}")
    print(f"  1 kHz power ratio: {freq_ratio:.4f}")

    passed = lr_corr > 0.999 and freq_ratio > 0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: corr>0.999, ratio>0.5)")
    assert passed, f"WBFM PLL mono decode failed: corr={lr_corr:.6f}, freq_ratio={freq_ratio:.4f}"
    return passed


def test_wbfm_pll_stereo_decode():
    """
    PLL path stereo decode coverage.

    Pass criteria: pilot lock and measurable channel separation.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM PLL Stereo Decode (1 kHz L, 2 kHz R)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 1.0
    left_freq = 1000
    right_freq = 2000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * left_freq * t)
    right = 0.5 * np.sin(2 * np.pi * right_freq * t)

    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='pll')
    audio = process_with_settling(demod, iq)

    skip = len(audio) // 2
    left_out = audio[skip:, 0]
    right_out = audio[skip:, 1]

    left_1k_power = fft_bin_band_power(left_out, left_freq, audio_rate, side_bins=1)
    left_2k_power = fft_bin_band_power(left_out, right_freq, audio_rate, side_bins=1)
    right_1k_power = fft_bin_band_power(right_out, left_freq, audio_rate, side_bins=1)
    right_2k_power = fft_bin_band_power(right_out, right_freq, audio_rate, side_bins=1)

    left_separation = 10 * np.log10(left_1k_power / (left_2k_power + 1e-12))
    right_separation = 10 * np.log10(right_2k_power / (right_1k_power + 1e-12))
    min_separation = min(left_separation, right_separation)

    print(f"  Pilot detected: {demod.pilot_detected}")
    print("  Separation metric: Hann FFT bin power (target ±1 bin)")
    print(f"  Left channel separation: {left_separation:.1f} dB")
    print(f"  Right channel separation: {right_separation:.1f} dB")

    passed = demod.pilot_detected and min_separation > 8
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: pilot lock, >8 dB separation)")
    assert passed, (
        f"WBFM PLL stereo decode failed: pilot={demod.pilot_detected}, min_separation={min_separation:.2f} dB"
    )
    return passed


def test_wbfm_pll_subcarrier_phase():
    """
    Test PLL decoder behavior vs TX subcarrier phase.

    Even with PLL carrier tracking, stereo decode assumes the standard pilot-to-
    subcarrier phase relationship:
    - neg_cos: correct decode
    - cos: inverted channels
    - sin: no usable separation (quadrature mismatch)
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM PLL Subcarrier Phase Sensitivity")
    print("=" * 60)
    print()
    print("  PLL tracks pilot phase, then derives 38 kHz by frequency doubling")
    print("  This test verifies behavior for different TX subcarrier phases")
    print()

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 1.0

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate

    left = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right = np.zeros(n_samples)

    results = {}

    for phase_name in ['neg_cos', 'cos', 'sin']:
        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase=phase_name)
        iq = fm_modulate(multiplex, iq_rate)

        demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='pll')
        audio = process_with_settling(demod, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = fft_bin_band_power(left_out, 1000, audio_rate, side_bins=1)
        right_power = fft_bin_band_power(right_out, 1000, audio_rate, side_bins=1)
        status = classify_stereo_phase_result(left_power, right_power)

        results[phase_name] = (left_power, right_power, status)
        print(f"  TX={phase_name:8s}: L={left_power:.6f}, R={right_power:.6f} -> {status}")

    print()

    neg_cos_correct = 'CORRECT' in results['neg_cos'][2]
    cos_inverted = 'INVERTED' in results['cos'][2]
    sin_fails = 'FAILED' in results['sin'][2]

    passed = neg_cos_correct and cos_inverted and sin_fails
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print("  (neg_cos should decode, cos should invert, sin should lose separation)")
    assert passed, f"Unexpected PLL phase sensitivity results: {results}"
    return passed


def test_wbfm_pll_group_delay():
    """
    PLL path timing alignment coverage.

    Pass criteria: L/R crossing difference < 10 samples at 48 kHz.
    """
    print("\n" + "=" * 60)
    print("TEST: WBFM PLL Group Delay Alignment")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_FM
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.1

    n_samples = int(duration * iq_rate)
    step = np.zeros(n_samples)
    step[n_samples // 2:] = 0.5

    demod = create_wbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate, stereo_decoder='pll')
    multiplex = generate_fm_stereo_multiplex(step, step, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)
    audio = process_with_settling(demod, iq)

    left_crossing = find_step_crossing(audio[:, 0])
    right_crossing = find_step_crossing(audio[:, 1])
    if left_crossing < 0 or right_crossing < 0:
        raise AssertionError("Could not find PLL step crossings")

    lr_diff_samples = abs(left_crossing - right_crossing)
    lr_diff_us = lr_diff_samples * (1e6 / audio_rate)

    print(f"  Left crossing at: {left_crossing:.2f} samples")
    print(f"  Right crossing at: {right_crossing:.2f} samples")
    print(f"  L/R crossing difference: {lr_diff_samples:.2f} samples ({lr_diff_us:.0f} us)")

    passed = lr_diff_samples < 10
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: <10 samples)")
    assert passed, f"WBFM PLL group delay mismatch too large: {lr_diff_samples:.2f} samples"
    return passed


# =============================================================================
# NBFM Test Functions (via NBFMDemodulator)
# =============================================================================

def create_nbfm_demod(iq_rate=ICOM_SAMPLE_RATE_WEATHER, audio_rate=AUDIO_SAMPLE_RATE):
    """Create an NBFMDemodulator configured for testing."""
    demod = NBFMDemodulator(
        input_sample_rate=iq_rate,
        audio_sample_rate=audio_rate
    )
    return demod


def test_nbfm_tone_recovery():
    """
    Test NBFM tone recovery.

    Generates a 1 kHz tone with 5 kHz deviation, verifies correct frequency.

    Pass criteria: SNR > 20 dB at 1 kHz
    """
    print("\n" + "=" * 60)
    print("TEST: NBFM Tone Recovery")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_WEATHER
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t)

    iq = nbfm_modulate(tone, iq_rate, deviation=NBFM_DEVIATION)

    demod = create_nbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    audio = process_with_settling(demod, iq)

    skip = int(0.05 * audio_rate)
    audio_trimmed = audio[skip:-skip]

    snr = measure_snr(audio_trimmed, test_freq, audio_rate)

    print(f"  IQ rate: {iq_rate/1000:.0f} kHz")
    print(f"  IF rate: {demod.actual_if_rate:.0f} Hz (decimation {demod.decimation}x)")
    print(f"  Test frequency: {test_freq} Hz")
    print(f"  SNR at {test_freq} Hz: {snr:.1f} dB")

    passed = snr > 20
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: SNR > 20 dB)")
    assert passed, f"NBFM tone recovery SNR below target: {snr:.2f} dB"
    return passed


def test_nbfm_audio_snr():
    """
    Test NBFM decoded audio SNR with clean input.

    Pass criteria: SNR > 25 dB
    """
    print("\n" + "=" * 60)
    print("TEST: NBFM Audio SNR (Clean Input)")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_WEATHER
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t)

    iq = nbfm_modulate(tone, iq_rate, deviation=NBFM_DEVIATION)

    demod = create_nbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    audio = process_with_settling(demod, iq)

    skip = int(0.05 * audio_rate)
    audio_trimmed = audio[skip:-skip]

    snr = measure_snr(audio_trimmed, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  Measured SNR: {snr:.1f} dB")

    passed = snr > 25
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >25 dB)")
    assert passed, f"NBFM audio SNR below target: {snr:.2f} dB"
    return passed


def test_nbfm_squelch():
    """
    Test NBFM squelch behavior.

    Pass criteria: Returns None when signal below squelch threshold
    """
    print("\n" + "=" * 60)
    print("TEST: NBFM Squelch")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_WEATHER
    n_samples = 8192

    demod = create_nbfm_demod(iq_rate=iq_rate)
    demod.set_squelch(-20)

    # Very weak noise
    rng = np.random.default_rng(0)
    noise = 1e-6 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
    result = demod.process(noise)

    squelched = result is None
    print(f"  Squelch level: -20 dB")
    print(f"  Noise-only result: {'squelched (None)' if squelched else 'NOT squelched'}")

    # Strong signal
    demod.reset()
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    iq_strong = nbfm_modulate(tone, iq_rate, deviation=NBFM_DEVIATION)
    result_strong = demod.process(iq_strong)

    unsquelched = result_strong is not None
    print(f"  Strong signal result: {'audio present' if unsquelched else 'SQUELCHED (unexpected)'}")

    passed = squelched and unsquelched
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, f"NBFM squelch behavior incorrect: squelched={squelched}, unsquelched={unsquelched}"
    return passed


def test_nbfm_freq_offset():
    """
    Test NBFM with tuning offset.

    Pass criteria: Correct tone recovered with offset tuning
    """
    print("\n" + "=" * 60)
    print("TEST: NBFM Frequency Offset Tuning")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_WEATHER
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000
    offset_hz = 25000  # 25 kHz offset (one weather channel)

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Generate NBFM at baseband
    iq_baseband = nbfm_modulate(tone, iq_rate, deviation=NBFM_DEVIATION)

    # Shift to offset
    shift = np.exp(2j * np.pi * offset_hz * t)
    iq_offset = (iq_baseband * shift).astype(np.complex64)

    demod = create_nbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    demod.set_tuned_offset(offset_hz)

    audio = process_with_settling(demod, iq_offset)

    skip = int(0.05 * audio_rate)
    audio_trimmed = audio[skip:-skip]

    snr = measure_snr(audio_trimmed, test_freq, audio_rate)

    print(f"  Offset: {offset_hz/1000:.0f} kHz")
    print(f"  SNR at {test_freq} Hz: {snr:.1f} dB")

    passed = snr > 15
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: SNR > 15 dB)")
    assert passed, f"NBFM offset tuning SNR below target: {snr:.2f} dB"
    return passed


def test_nbfm_hum_filter():
    """
    Test NBFM hum reduction filter.

    Generates tone with 60 Hz hum, verifies hum filter attenuates it.

    Pass criteria: 60 Hz power reduced by >10 dB with filter enabled
    """
    print("\n" + "=" * 60)
    print("TEST: NBFM Hum Filter")
    print("=" * 60)

    iq_rate = ICOM_SAMPLE_RATE_WEATHER
    audio_rate = AUDIO_SAMPLE_RATE
    duration = 0.5
    test_freq = 1000
    hum_freq = 60

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    # Voice tone + 60 Hz hum
    audio_in = 0.5 * np.sin(2 * np.pi * test_freq * t) + 0.3 * np.sin(2 * np.pi * hum_freq * t)

    iq = nbfm_modulate(audio_in, iq_rate, deviation=NBFM_DEVIATION)

    # Without hum filter
    demod_no_hum = create_nbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    demod_no_hum.set_hum_filter(False)
    audio_no_hum = process_with_settling(demod_no_hum, iq)

    # With hum filter
    demod_hum = create_nbfm_demod(iq_rate=iq_rate, audio_rate=audio_rate)
    demod_hum.set_hum_filter(True)
    audio_hum = process_with_settling(demod_hum, iq)

    skip = int(0.1 * audio_rate)
    audio_no_hum_t = audio_no_hum[skip:-skip]
    audio_hum_t = audio_hum[skip:-skip]

    hum_power_off = goertzel_power(audio_no_hum_t, hum_freq, audio_rate)
    hum_power_on = goertzel_power(audio_hum_t, hum_freq, audio_rate)

    if hum_power_off > 0 and hum_power_on > 0:
        hum_reduction_db = 10 * np.log10(hum_power_off / hum_power_on)
    else:
        hum_reduction_db = 0

    # Check voice tone is preserved
    voice_power_off = goertzel_power(audio_no_hum_t, test_freq, audio_rate)
    voice_power_on = goertzel_power(audio_hum_t, test_freq, audio_rate)

    if voice_power_off > 0 and voice_power_on > 0:
        voice_change_db = 10 * np.log10(voice_power_on / voice_power_off)
    else:
        voice_change_db = 0

    print(f"  60 Hz hum reduction: {hum_reduction_db:.1f} dB")
    print(f"  1 kHz voice change: {voice_change_db:+.1f} dB")

    passed = hum_reduction_db > 10 and abs(voice_change_db) < 3
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: hum -10 dB, voice +-3 dB)")
    assert passed, (
        f"NBFM hum filter failed: hum_reduction={hum_reduction_db:.2f} dB, voice_change={voice_change_db:.2f} dB"
    )
    return passed


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("PANADAPTER DEMODULATOR TEST SUITE")
    print("=" * 60)
    print("\nTesting panadapter.py demodulator classes")
    print(f"  Primary WBFM decoder under test: {PRIMARY_WBFM_DECODER}")
    print("  WBFMStereoDemodulator: IQ -> Decimate -> (PLL or squaring stereo decoder) -> Audio")
    print("  NBFMDemodulator: IQ -> Channel BPF -> FM Demod -> Audio")

    tests = [
        # WBFM tests (via WBFMStereoDemodulator)
        ("WBFM Primary Decoder Selection", test_wbfm_primary_decoder_selected),
        ("WBFM Demod Accuracy", test_wbfm_demod_accuracy),
        ("WBFM Audio SNR", test_wbfm_audio_snr),
        ("WBFM THD+N", test_wbfm_thd_n),
        ("WBFM Mono Decoding", test_wbfm_mono_decode),
        ("WBFM Stereo Decoding", test_wbfm_stereo_decode),
        ("WBFM Stereo Separation", test_wbfm_stereo_separation),
        ("WBFM Subcarrier Phase", test_wbfm_subcarrier_phase),
        ("WBFM Group Delay", test_wbfm_group_delay),
        ("WBFM Frequency Response", test_wbfm_frequency_response),
        ("WBFM SNR with Noise", test_wbfm_snr_with_noise),
        ("WBFM IHF/EIA SNR", test_wbfm_ihf_snr),
        ("WBFM Frequency Offset", test_wbfm_freq_offset),
        ("WBFM Squelch", test_wbfm_squelch),
        ("WBFM PLL Mono Decode", test_wbfm_pll_mono_decode),
        ("WBFM PLL Stereo Decode", test_wbfm_pll_stereo_decode),
        ("WBFM PLL Subcarrier Phase", test_wbfm_pll_subcarrier_phase),
        ("WBFM PLL Group Delay", test_wbfm_pll_group_delay),
        # NBFM tests (via NBFMDemodulator)
        ("NBFM Tone Recovery", test_nbfm_tone_recovery),
        ("NBFM Audio SNR", test_nbfm_audio_snr),
        ("NBFM Squelch", test_nbfm_squelch),
        ("NBFM Frequency Offset", test_nbfm_freq_offset),
        ("NBFM Hum Filter", test_nbfm_hum_filter),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s}  {status}")

    print(f"\n  {passed_count}/{total_count} tests passed")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

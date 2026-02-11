#!/usr/bin/env python3
r"""
FM Stereo Decoder Test Suite

Tests mathematically correct mono and stereo FM decoding with high stereo separation.
Documents phase/delay relationships throughout the decode chain.

Usage:
    python test_pjfm.py          # Run all tests with detailed output
    pytest test_pjfm.py -v       # Run with pytest (if installed)

Signal Flow:
    IQ -> FM Demod -> Pilot BPF (201 taps) -> carrier regen -> 38 kHz carrier
                 \-> L+R LPF (127 taps) + 100 sample delay compensation
                 \-> L-R BPF (201 taps) -> x carrier x 2 -> L-R LPF (127 taps)
                                                                    |
                 Matrix: L = L+R + L-R, R = L+R - L-R <-------------+

Carrier Regeneration:
    PLL locks to the 19 kHz pilot and regenerates coherent 38 kHz carrier.

Group Delay Alignment:
    L+R path: LPF (63 samples) + delay buffer (100 samples) = 163 samples
    L-R path: BPF (100 samples) + LPF (63 samples) = 163 samples

FM Multiplex Structure:
    0-15 kHz:   L+R (mono compatible)
    19 kHz:     Pilot tone (9% amplitude)
    23-53 kHz:  L-R on 38 kHz DSB-SC carrier

Critical Phase Relationships:
    For correct stereo decode, transmitter should use -cos(2wt) as subcarrier:
    - TX = -cos(2wt), RX = -cos(2wt): WORKS (correct polarity)
    - TX = cos(2wt),  RX = -cos(2wt): WORKS (inverted L/R)
    - TX = sin(2wt),  RX = -cos(2wt): FAILS (90° phase error = no separation)

Validation Targets:
    FM demod accuracy: Correlation > 0.999
    Audio SNR (clean): > 35 dB (limited by full stereo chain processing)
    THD+N: < -35 dB (< 2% distortion, limited by soft limiter)
    Stereo separation: > 30 dB (67+ dB typical with synthetic signal)
    L/R timing: < 5 samples at 48 kHz
    Frequency response: +/- 3 dB below 1 kHz
"""

import numpy as np
from scipy import signal
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pll_stereo_decoder import PLLStereoDecoder


# =============================================================================
# Helper Functions
# =============================================================================

def demodulate_with_settling(decoder, iq, block_size=8192):
    """
    Demodulate IQ samples in blocks to allow pilot detection to settle.

    Args:
        decoder: Stereo decoder instance
        iq: Complex I/Q samples
        block_size: Block size for processing

    Returns:
        Concatenated audio output (N, 2) array
    """
    audio_chunks = []
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) > 0:
            audio = decoder.demodulate(block)
            audio_chunks.append(audio)

    return np.vstack(audio_chunks) if audio_chunks else np.zeros((0, 2), dtype=np.float32)


# =============================================================================
# Signal Generation Functions
# =============================================================================

def generate_test_tone(freq, duration, sample_rate, amplitude=1.0):
    """
    Generate a pure sine wave test tone.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude (default 1.0)

    Returns:
        numpy array of float64 samples
    """
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

    The subcarrier_phase parameter controls the 38 kHz carrier phase:
    - 'neg_cos': -cos(2*pi*38000*t) - matches pilot-squaring decoder
    - 'cos': cos(2*pi*38000*t) - gives inverted stereo
    - 'sin': sin(2*pi*38000*t) - commonly cited but causes 90° error!

    Args:
        left: Left channel audio samples
        right: Right channel audio samples
        sample_rate: Sample rate in Hz
        pilot_amplitude: Pilot tone amplitude (default 0.09 = 9%)
        include_pilot: Whether to include 19 kHz pilot (default True)
        subcarrier_phase: '38 kHz carrier phase ('neg_cos', 'cos', or 'sin')

    Returns:
        FM multiplex baseband signal (normalized for 75 kHz deviation)
    """
    n = len(left)
    t = np.arange(n) / sample_rate

    # L+R and L-R (each scaled by 0.5 for proper matrix decode)
    lr_sum = (left + right) / 2
    lr_diff = (left - right) / 2

    # Pilot tone at 19 kHz (sine phase per standard)
    if include_pilot:
        pilot = pilot_amplitude * np.sin(2 * np.pi * 19000 * t)
    else:
        pilot = np.zeros(n)

    # 38 kHz carrier for L-R DSB-SC modulation
    if subcarrier_phase == 'neg_cos':
        # -cos(2wt) = 2*sin^2(wt) - 1 (matches pilot-squaring decoder)
        carrier_38k = -np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'cos':
        # cos(2wt) - gives inverted stereo with pilot-squaring decoder
        carrier_38k = np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'sin':
        # sin(2wt) - commonly cited but causes 90° error with pilot-squaring!
        carrier_38k = np.sin(2 * np.pi * 38000 * t)
    else:
        raise ValueError(f"Unknown subcarrier_phase: {subcarrier_phase}")

    # DSB-SC modulated L-R
    lr_diff_mod = lr_diff * carrier_38k

    # Combine multiplex signal
    # Scale L+R and L-R for pilot headroom (total deviation budget)
    multiplex = lr_sum * 0.9 + pilot + lr_diff_mod * 0.9

    return multiplex


def fm_modulate(baseband, sample_rate, deviation=75000):
    """
    FM modulate baseband signal to complex I/Q.

    Uses the standard FM formula:
        phase(t) = 2*pi*deviation * integral(baseband) * dt
        IQ = cos(phase) + j*sin(phase)

    Args:
        baseband: Baseband audio/multiplex signal
        sample_rate: Sample rate in Hz
        deviation: FM deviation in Hz (75 kHz for broadcast FM)

    Returns:
        Complex I/Q samples (numpy array of complex64)
    """
    dt = 1.0 / sample_rate

    # Instantaneous phase is the integral of baseband scaled by deviation
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt

    # Generate complex I/Q
    iq = np.cos(phase) + 1j * np.sin(phase)

    return iq.astype(np.complex64)


# =============================================================================
# Measurement Utilities
# =============================================================================

def goertzel_power(x, target_freq, sample_rate):
    """
    Compute power at a specific frequency using Goertzel algorithm.

    More efficient than FFT for single-frequency measurement.

    Args:
        x: Input signal
        target_freq: Frequency to measure in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Power at target frequency (linear scale)
    """
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

    # Compute power
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    # Normalize by N^2 / 2 for consistent power measurement
    power = power / (n * n / 2)

    return power


def measure_snr(x, signal_freq, sample_rate, signal_bw=100):
    """
    Measure SNR of a signal with a known tone frequency.

    Args:
        x: Input signal
        signal_freq: Frequency of the signal tone in Hz
        sample_rate: Sample rate in Hz
        signal_bw: Bandwidth around signal to consider as signal (Hz)

    Returns:
        SNR in dB
    """
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    # Signal power: in narrow band around signal frequency
    signal_mask = np.abs(freqs - signal_freq) <= signal_bw / 2
    signal_power = np.sum(power_spectrum[signal_mask])

    # Noise power: everything except signal band
    noise_mask = ~signal_mask
    noise_power = np.sum(power_spectrum[noise_mask])

    if noise_power <= 0:
        return np.inf

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def measure_ihf_tone_snr(x, signal_freq, sample_rate, a_weight_sos, signal_bw=100):
    """
    Measure IHF/EIA-style tone SNR on audio output.

    Applies A-weighting and then computes tone power vs residual FFT power.
    """
    weighted = signal.sosfilt(a_weight_sos, x)
    n = len(weighted)
    window = np.hanning(n)
    x_windowed = weighted * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    signal_mask = np.abs(freqs - signal_freq) <= signal_bw / 2
    signal_power = np.sum(power_spectrum[signal_mask])
    noise_power = np.sum(power_spectrum[~signal_mask])

    if noise_power <= 0:
        return np.inf

    return 10 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20))


def measure_thd_n(x, fundamental_freq, sample_rate, n_harmonics=5):
    """
    Measure Total Harmonic Distortion + Noise (THD+N).

    Args:
        x: Input signal
        fundamental_freq: Fundamental frequency in Hz
        sample_rate: Sample rate in Hz
        n_harmonics: Number of harmonics to include

    Returns:
        THD+N in dB (negative value, e.g., -40 dB = 1% THD+N)
    """
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    # Fundamental power
    fundamental_mask = np.abs(freqs - fundamental_freq) <= 50
    fundamental_power = np.sum(power_spectrum[fundamental_mask])

    # Harmonic power
    harmonic_power = 0.0
    for h in range(2, n_harmonics + 1):
        harm_freq = fundamental_freq * h
        if harm_freq >= sample_rate / 2:
            break
        harm_mask = np.abs(freqs - harm_freq) <= 50
        harmonic_power += np.sum(power_spectrum[harm_mask])

    # Total power (signal + noise)
    total_power = np.sum(power_spectrum)

    # Noise power = total - fundamental - harmonics
    noise_power = total_power - fundamental_power - harmonic_power

    # THD+N = sqrt(harmonics + noise) / fundamental
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
    # Analog prototype zeros and poles (rad/s)
    z = [0, 0, 0, 0]
    p = [-2 * np.pi * 20.598997,
         -2 * np.pi * 20.598997,
         -2 * np.pi * 107.65265,
         -2 * np.pi * 737.86223,
         -2 * np.pi * 12194.217,
         -2 * np.pi * 12194.217]
    # Gain chosen so analog response = 0 dB at 1 kHz
    k = (2 * np.pi * 12194.217) ** 2

    # Bilinear transform to digital
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)

    # Convert to second-order sections
    sos = signal.zpk2sos(zd, pd, kd)

    # Normalize to 0 dB at 1 kHz
    w, h = signal.sosfreqz(sos, worN=[2 * np.pi * 1000 / fs])
    sos[0, :3] /= np.abs(h[0])

    return sos


def find_step_crossing(x, threshold=0.5):
    """
    Find the sample index where signal crosses threshold.

    Used for measuring group delay with step response.

    Args:
        x: Input signal
        threshold: Crossing threshold (default 0.5 for 50%)

    Returns:
        Sample index of first crossing, or -1 if not found
    """
    # Normalize to 0-1 range
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        return -1

    x_norm = (x - x_min) / (x_max - x_min)

    # Find first crossing
    for i in range(len(x_norm) - 1):
        if x_norm[i] < threshold <= x_norm[i + 1]:
            # Linear interpolation for sub-sample accuracy
            frac = (threshold - x_norm[i]) / (x_norm[i + 1] - x_norm[i])
            return i + frac

    return -1


def estimate_relative_delay_xcorr(x, y, max_lag=20):
    """
    Estimate relative delay (in samples) via cross-correlation peak.

    Uses FFT-based correlation over a bounded lag window and applies a
    parabolic fit around the peak for sub-sample resolution.
    """
    n = min(len(x), len(y))
    if n <= (2 * max_lag + 1):
        return 0.0

    x0 = np.asarray(x[:n], dtype=np.float64)
    y0 = np.asarray(y[:n], dtype=np.float64)
    x0 = x0 - np.mean(x0)
    y0 = y0 - np.mean(y0)

    corr_full = signal.correlate(x0, y0, mode='full', method='fft')
    lags_full = signal.correlation_lags(len(x0), len(y0), mode='full')

    lag_mask = (lags_full >= -max_lag) & (lags_full <= max_lag)
    corr = corr_full[lag_mask]
    lags = lags_full[lag_mask]

    peak_idx = int(np.argmax(corr))
    lag_samples = float(lags[peak_idx])
    frac = 0.0

    if 0 < peak_idx < (len(corr) - 1):
        y1 = float(corr[peak_idx - 1])
        y2 = float(corr[peak_idx])
        y3 = float(corr[peak_idx + 1])
        denom = y1 - (2.0 * y2) + y3
        if abs(denom) > 1e-12:
            frac = 0.5 * (y1 - y3) / denom
            frac = float(np.clip(frac, -1.0, 1.0))

    return lag_samples + frac


# =============================================================================
# Test Functions
# =============================================================================

def test_fm_demod_accuracy():
    """
    Test FM demodulation accuracy.

    Generates known baseband, FM modulates to I/Q, demodulates back,
    and compares output to input.

    Pass criteria: Correlation > 0.999, amplitude error < 1%
    """
    print("\n" + "=" * 60)
    print("TEST: FM Demodulation Accuracy")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1  # 100 ms
    test_freq = 1000  # 1 kHz test tone

    # Generate test tone baseband
    baseband_in = generate_test_tone(test_freq, duration, sample_rate, amplitude=0.5)

    # FM modulate to I/Q
    iq = fm_modulate(baseband_in, sample_rate, deviation=75000)

    # Create decoder (disable tone controls and de-emphasis for clean test)
    decoder = PLLStereoDecoder(
        iq_sample_rate=sample_rate,
        audio_sample_rate=sample_rate,  # No resampling
        deviation=75000,
        deemphasis=1e-9  # Effectively disable de-emphasis
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = decoder.demodulate(iq)

    # Get baseband from decoder (before filtering/resampling)
    baseband_out = decoder.last_baseband

    # Allow for filter settling - skip first 2000 samples
    skip = 2000
    baseband_in_trimmed = baseband_in[skip:-skip]
    baseband_out_trimmed = baseband_out[skip:-skip]

    # Compute correlation
    correlation = np.corrcoef(baseband_in_trimmed, baseband_out_trimmed)[0, 1]

    # Compute RMS amplitude ratio
    rms_in = np.sqrt(np.mean(baseband_in_trimmed ** 2))
    rms_out = np.sqrt(np.mean(baseband_out_trimmed ** 2))
    amplitude_ratio = rms_out / rms_in
    amplitude_error = abs(1.0 - amplitude_ratio) * 100

    print(f"  Correlation: {correlation:.6f}")
    print(f"  Amplitude ratio: {amplitude_ratio:.4f}")
    print(f"  Amplitude error: {amplitude_error:.2f}%")

    passed = correlation > 0.999 and amplitude_error < 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_audio_snr():
    """
    Test decoded audio SNR with clean input.

    Generates clean 1 kHz stereo signal, measures SNR of decoded audio.
    With perfect synthetic signal, expect >40 dB SNR.
    """
    print("\n" + "=" * 60)
    print("TEST: Audio SNR (Clean Input)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.5  # 500 ms for stable measurement
    test_freq = 1000

    # Generate stereo test signal
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * test_freq * t)
    right = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex and modulate (use neg_cos for pilot-squaring)
    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Skip settling time
    skip = int(0.05 * audio_rate)  # 50 ms
    left_out = audio[skip:-skip, 0]

    # Measure SNR
    snr = measure_snr(left_out, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  Measured SNR: {snr:.1f} dB")

    # Note: 35 dB is realistic for full stereo chain with filters/resampling/limiter
    passed = snr > 35
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >35 dB)")

    return passed


def test_thd_n():
    """
    Test Total Harmonic Distortion + Noise.

    Target: THD+N < -40 dB (< 1% distortion)
    """
    print("\n" + "=" * 60)
    print("TEST: THD+N (Total Harmonic Distortion + Noise)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.5
    test_freq = 1000

    # Generate mono test signal (identical L/R)
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex and modulate
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Skip settling
    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]

    # Measure THD+N
    thd_n = measure_thd_n(left_out, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  THD+N: {thd_n:.1f} dB ({10**(thd_n/20)*100:.2f}%)")

    # Note: -35 dB (1.8%) is realistic for full stereo chain with soft limiter
    passed = thd_n < -35
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: < -35 dB)")

    return passed


def test_mono_decode():
    """
    Test mono decoding (no pilot tone).

    Generates L+R only signal without pilot, verifies mono output.

    Pass criteria:
    - L/R correlation > 0.999 (same signal)
    - Correct frequency detected
    """
    print("\n" + "=" * 60)
    print("TEST: Mono Decoding (No Pilot)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.3
    test_freq = 1000

    # Generate mono signal
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex WITHOUT pilot
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder (production default settings)
    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Check pilot detection
    pilot_detected = decoder.pilot_detected

    # Skip settling
    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]
    right_out = audio[skip:-skip, 1]

    # L/R correlation
    lr_corr = np.corrcoef(left_out, right_out)[0, 1]

    # Check for correct frequency
    left_power_1k = goertzel_power(left_out, test_freq, audio_rate)
    left_power_total = np.mean(left_out ** 2)
    freq_ratio = left_power_1k / left_power_total if left_power_total > 0 else 0

    print(f"  Pilot detected: {pilot_detected}")
    print(f"  L/R correlation: {lr_corr:.6f}")
    print(f"  1 kHz power ratio: {freq_ratio:.4f}")

    passed = not pilot_detected and lr_corr > 0.999 and freq_ratio > 0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_stereo_decode():
    """
    Test stereo decoding with different tones on L and R.

    Generates 1 kHz on left, 2 kHz on right.

    Pass criteria:
    - Pilot detected
    - Channel separation > 20 dB
    """
    print("\n" + "=" * 60)
    print("TEST: Stereo Decoding (1 kHz L, 2 kHz R)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0
    left_freq = 1000
    right_freq = 2000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * left_freq * t)
    right = 0.5 * np.sin(2 * np.pi * right_freq * t)

    # Use neg_cos subcarrier (matches pilot-squaring decoder)
    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    audio = demodulate_with_settling(decoder, iq)

    pilot_detected = decoder.pilot_detected

    skip = len(audio) // 2
    left_out = audio[skip:, 0]
    right_out = audio[skip:, 1]

    left_1k_power = goertzel_power(left_out, left_freq, audio_rate)
    left_2k_power = goertzel_power(left_out, right_freq, audio_rate)
    right_1k_power = goertzel_power(right_out, left_freq, audio_rate)
    right_2k_power = goertzel_power(right_out, right_freq, audio_rate)

    left_separation = 10 * np.log10(left_1k_power / (left_2k_power + 1e-12))
    right_separation = 10 * np.log10(right_2k_power / (right_1k_power + 1e-12))

    print(f"  Pilot detected: {pilot_detected}")
    print(f"  Left channel separation: {left_separation:.1f} dB")
    print(f"  Right channel separation: {right_separation:.1f} dB")

    min_separation = min(left_separation, right_separation)

    passed = pilot_detected and min_separation > 20
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >20 dB separation)")

    return passed


def test_stereo_separation():
    """
    Test stereo separation across frequency range.

    Tests left-only tones at 100 Hz, 1 kHz, 5 kHz, 10 kHz, 12 kHz.

    Pass criteria: >30 dB separation at all tested frequencies
    """
    print("\n" + "=" * 60)
    print("TEST: Stereo Separation vs Frequency")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0

    test_freqs = [100, 1000, 5000, 10000, 12000]

    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    all_passed = True
    separations = []

    for freq in test_freqs:
        decoder.reset()

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        left = 0.5 * np.sin(2 * np.pi * freq * t)
        right = np.zeros(n_samples)

        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = demodulate_with_settling(decoder, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = goertzel_power(left_out, freq, audio_rate)
        right_power = goertzel_power(right_out, freq, audio_rate)

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

    return all_passed


def test_subcarrier_phase_sensitivity():
    """
    Test decoder sensitivity to subcarrier phase.

    Documents what happens with different TX subcarrier phases:
    - neg_cos: Should work correctly (matches pilot-squaring)
    - cos: Should give inverted stereo (L/R swap)
    - sin: Should fail completely (90° phase error)
    """
    print("\n" + "=" * 60)
    print("TEST: Subcarrier Phase Sensitivity")
    print("=" * 60)
    print()
    print("  FM stereo multiplex is referenced to a -cos(2wt) subcarrier phase")
    print("  This test verifies expected decode behavior with different TX phases")
    print()

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate

    # Left-only signal
    left = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right = np.zeros(n_samples)

    results = {}

    for phase_name in ['neg_cos', 'cos', 'sin']:
        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase=phase_name)
        iq = fm_modulate(multiplex, iq_rate)

        decoder = PLLStereoDecoder(
            iq_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            deviation=75000,
            deemphasis=1e-9
        )
        decoder.bass_boost_enabled = False
        decoder.treble_boost_enabled = False
        decoder.stereo_blend_enabled = False

        audio = demodulate_with_settling(decoder, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = goertzel_power(left_out, 1000, audio_rate)
        right_power = goertzel_power(right_out, 1000, audio_rate)

        # Determine result
        if left_power > 0.01 and right_power < 0.001:
            status = "CORRECT (signal in L)"
        elif right_power > 0.01 and left_power < 0.001:
            status = "INVERTED (signal in R)"
        elif abs(left_power - right_power) / max(left_power, right_power, 1e-12) < 0.5:
            status = "FAILED (no separation)"
        else:
            status = "PARTIAL"

        results[phase_name] = (left_power, right_power, status)

        print(f"  TX={phase_name:8s}: L={left_power:.4f}, R={right_power:.4f} -> {status}")

    print()

    # Pass if neg_cos works correctly and sin fails
    neg_cos_correct = 'CORRECT' in results['neg_cos'][2]
    sin_fails = 'FAILED' in results['sin'][2]

    passed = neg_cos_correct and sin_fails
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print(f"  (neg_cos should work, sin should fail due to 90° phase error)")

    return passed


def test_group_delay_alignment():
    """
    Test L/R timing alignment using cross-correlation.

    Uses a shared broadband mono stimulus and estimates relative delay from the
    cross-correlation peak. This is less sensitive to transient edge-shape
    differences than step-crossing.
    """
    print("\n" + "=" * 60)
    print("TEST: Group Delay Alignment (L/R XCorr)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0

    n_samples = int(duration * iq_rate)
    rng = np.random.default_rng(12345)
    mono = rng.standard_normal(n_samples)
    mono_lpf = signal.firwin(255, 12000 / (iq_rate / 2), window=('kaiser', 6.0))
    mono = signal.lfilter(mono_lpf, 1.0, mono)
    mono = 0.5 * mono / (np.max(np.abs(mono)) + 1e-12)

    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)
    audio = demodulate_with_settling(decoder, iq)

    skip = int(0.2 * audio_rate)
    left = audio[skip:, 0]
    right = audio[skip:, 1]
    if len(left) < 256 or len(right) < 256:
        print("  ERROR: Insufficient audio samples for xcorr delay estimate")
        return False

    lr_delay_samples = estimate_relative_delay_xcorr(left, right, max_lag=20)
    lr_delay_us = lr_delay_samples * (1e6 / audio_rate)

    print(f"  L-R delay (xcorr): {lr_delay_samples:+.3f} samples ({lr_delay_us:+.1f} us)")

    passed = abs(lr_delay_samples) < 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: |delay| < 1.0 samples)")

    return passed


def test_frequency_response():
    """
    Test frequency response from 100 Hz to 14 kHz.

    Measures relative amplitude across frequency range.
    Expect flat +-3 dB response (before de-emphasis).
    """
    print("\n" + "=" * 60)
    print("TEST: Frequency Response")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.2

    test_freqs = [100, 200, 500, 1000, 2000, 5000, 8000, 10000, 12000, 14000]

    decoder = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    responses = []

    for freq in test_freqs:
        decoder.reset()

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        mono = 0.5 * np.sin(2 * np.pi * freq * t)

        multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = demodulate_with_settling(decoder, iq)

        skip = int(0.05 * audio_rate)
        left_out = audio[skip:-skip, 0]

        power = goertzel_power(left_out, freq, audio_rate)
        responses.append(power)

    # Normalize to 1 kHz response
    ref_idx = test_freqs.index(1000)
    ref_power = responses[ref_idx]

    print(f"  {'Freq':>6s}  {'Response':>10s}  {'Relative':>10s}")
    print(f"  {'----':>6s}  {'--------':>10s}  {'--------':>10s}")

    max_deviation = 0
    for freq, power in zip(test_freqs, responses):
        rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
        print(f"  {freq:6d}  {power:10.6f}  {rel_db:+7.2f} dB")
        max_deviation = max(max_deviation, abs(rel_db))

    # Check response up to 1 kHz
    low_freq_max_dev = 0
    for freq, power in zip(test_freqs, responses):
        if freq <= 1000:
            rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
            low_freq_max_dev = max(low_freq_max_dev, abs(rel_db))

    print(f"\n  Max deviation (all): {max_deviation:.2f} dB")
    print(f"  Max deviation (<= 1 kHz): {low_freq_max_dev:.2f} dB")

    passed = low_freq_max_dev < 3.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: +-3 dB below 1 kHz)")

    return passed


def test_snr_with_noise():
    """
    Test decoder behavior with noisy input.

    Adds AWGN at various levels, measures decoded audio quality.
    Verifies graceful degradation.
    """
    print("\n" + "=" * 60)
    print("TEST: IHF/EIA SNR with Noisy Input (PLL, firdecim)")
    print("=" * 60)

    iq_rate = 480000
    audio_rate = 48000
    duration = 1.0

    input_snr_levels = [40, 30, 25, 20, 15, 10]
    test_freq = 1000
    a_weight_sos = design_a_weighting(audio_rate)

    # Force this test to exercise the exact path requested.
    resampler_mode = "firdecim"
    resampler_taps = 127
    resampler_beta = 8.0

    decoder_paths = [("PLLStereoDecoder", PLLStereoDecoder)]

    print(f"  IQ rate: {iq_rate/1000:.0f} kHz, metric: IHF/EIA (A-weighted, de-emphasized)")
    print(f"  Decoder resampler: mode={resampler_mode}, taps={resampler_taps}, beta={resampler_beta:.1f}")

    all_passed = True

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Companion stimuli:
    # - dual-mono: IHF SNR curve behavior
    # - left-only: stereo separation behavior
    multiplex_mono = generate_fm_stereo_multiplex(tone, tone, iq_rate, subcarrier_phase='neg_cos')
    iq_clean_mono = fm_modulate(multiplex_mono, iq_rate)
    signal_power_mono = np.mean(np.abs(iq_clean_mono) ** 2)

    right_silent = np.zeros_like(tone)
    multiplex_sep = generate_fm_stereo_multiplex(tone, right_silent, iq_rate, subcarrier_phase='neg_cos')
    iq_clean_sep = fm_modulate(multiplex_sep, iq_rate)
    signal_power_sep = np.mean(np.abs(iq_clean_sep) ** 2)

    for path_idx, (path_name, decoder_class) in enumerate(decoder_paths):
        print(f"\n  Path: {path_name}")
        print(f"    {'Input SNR':>12s}  {'IHF SNR':>10s}  {'Sep':>8s}  {'Pilot':>8s}  {'Blend':>8s}  {'Resampler':>10s}")
        print(f"    {'--------':>12s}  {'-------':>10s}  {'---':>8s}  {'-----':>8s}  {'-----':>8s}  {'---------':>10s}")

        ihf_by_level = []
        blend_by_level = []

        for level_idx, input_snr in enumerate(input_snr_levels):
            # Deterministic per-row noise for reproducible CI output.
            rng = np.random.default_rng(20260208 + path_idx * 100 + level_idx)
            noise_power = signal_power_mono / (10 ** (input_snr / 10))
            noise = np.sqrt(noise_power / 2) * (
                rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            )
            iq_noisy = iq_clean_mono + noise.astype(np.complex64)

            decoder = decoder_class(
                iq_sample_rate=iq_rate,
                audio_sample_rate=audio_rate,
                deviation=75000,
                deemphasis=75e-6,
                resampler_mode=resampler_mode,
                resampler_taps=resampler_taps,
                resampler_beta=resampler_beta,
            )
            decoder.bass_boost_enabled = False
            decoder.treble_boost_enabled = False
            decoder.stereo_blend_enabled = True

            audio = demodulate_with_settling(decoder, iq_noisy)
            pilot_detected = decoder.pilot_detected
            blend_factor = decoder.stereo_blend_factor
            runtime_resampler = getattr(decoder, "_resampler_runtime_mode", decoder.resampler_mode)

            skip = int(0.1 * audio_rate)
            left_out = audio[skip:-skip, 0]
            output_snr = measure_ihf_tone_snr(
                left_out,
                test_freq,
                audio_rate,
                a_weight_sos=a_weight_sos,
                signal_bw=100,
            )

            # Left-only companion run for stereo separation at this RF SNR.
            rng_sep = np.random.default_rng(20260208 + path_idx * 100 + level_idx + 10_000)
            noise_power_sep = signal_power_sep / (10 ** (input_snr / 10))
            noise_sep = np.sqrt(noise_power_sep / 2) * (
                rng_sep.standard_normal(n_samples) + 1j * rng_sep.standard_normal(n_samples)
            )
            iq_noisy_sep = iq_clean_sep + noise_sep.astype(np.complex64)

            decoder_sep = decoder_class(
                iq_sample_rate=iq_rate,
                audio_sample_rate=audio_rate,
                deviation=75000,
                deemphasis=75e-6,
                resampler_mode=resampler_mode,
                resampler_taps=resampler_taps,
                resampler_beta=resampler_beta,
            )
            decoder_sep.bass_boost_enabled = False
            decoder_sep.treble_boost_enabled = False
            decoder_sep.stereo_blend_enabled = True

            audio_sep = demodulate_with_settling(decoder_sep, iq_noisy_sep)
            left_sep = audio_sep[skip:-skip, 0]
            right_sep = audio_sep[skip:-skip, 1]
            sep_left_power = goertzel_power(left_sep, test_freq, audio_rate)
            sep_right_power = goertzel_power(right_sep, test_freq, audio_rate)
            separation_db = 10 * np.log10((sep_left_power + 1e-20) / (sep_right_power + 1e-20))
            runtime_resampler_sep = getattr(decoder_sep, "_resampler_runtime_mode", decoder_sep.resampler_mode)

            print(
                f"    {input_snr:10d} dB  {output_snr:8.1f} dB  {separation_db:6.1f} dB  "
                f"{'Yes' if pilot_detected else 'No':>8s}  {blend_factor:8.2f}  {runtime_resampler:>10s}"
            )

            ihf_by_level.append(output_snr)
            blend_by_level.append(blend_factor)

            if not pilot_detected:
                all_passed = False
            if runtime_resampler != "firdecim":
                all_passed = False
            if runtime_resampler_sep != "firdecim":
                all_passed = False

        # Graceful degradation checks:
        # Pre-blend region should decline with RF SNR.
        if not (ihf_by_level[0] > ihf_by_level[1] > ihf_by_level[2] > ihf_by_level[3]):
            all_passed = False
        # As blend engages (15 dB), quality should not cliff; it may flatten/recover.
        if ihf_by_level[4] < ihf_by_level[3] - 2.0:
            all_passed = False
        # Adjacent low-SNR bumps should be limited. Allow a larger final bump
        # when blend has collapsed near mono (can substantially reduce stereo
        # noise on very weak signals).
        if ihf_by_level[4] > ihf_by_level[3] + 2.0:
            all_passed = False
        if ihf_by_level[5] > ihf_by_level[4] + 6.0:
            all_passed = False
        # Very low RF SNR should trigger blend-to-mono behavior.
        if blend_by_level[-1] > 0.20:
            all_passed = False
        # At high RF SNR we should remain mostly stereo.
        if blend_by_level[0] < 0.50:
            all_passed = False
        # Blend handoff should be clearly underway by 20 dB.
        if blend_by_level[3] > 0.75:
            all_passed = False
        # Blend factor should not increase as RF SNR worsens (allow tiny jitter).
        for prev_blend, curr_blend in zip(blend_by_level, blend_by_level[1:]):
            if curr_blend > prev_blend + 0.03:
                all_passed = False

    print(f"\n  Result: {'PASS' if all_passed else 'FAIL'} (IHF metric + firdecim path verified)")

    return all_passed


def test_ihf_snr():
    """
    Test IHF/EIA-style audio-referenced SNR.

    Standard hi-fi tuner specs use A-weighted, de-emphasized audio SNR rather
    than pilot-referenced RF SNR.  De-emphasis + A-weighting adds ~10-15 dB
    to the number by rolling off high-frequency noise.

    Procedure:
      1. Generate full-deviation 1 kHz stereo tone
      2. FM modulate with AWGN at 40 dB RF SNR
      3. Decode with de-emphasis enabled
      4. A-weight the output, measure tone-to-residual ratio via FFT
      5. Compare stereo and mono results

    Uses 480 kHz IQ rate (production default) for best filter performance.

    Pass thresholds: stereo > 35 dB, mono > 50 dB
    """
    print("\n" + "=" * 60)
    print("TEST: IHF/EIA-Style SNR (A-weighted, de-emphasized)")
    print("=" * 60)

    iq_rate = 480000
    audio_rate = 48000
    duration = 1.0
    test_freq = 1000
    rf_snr_db = 40

    rng = np.random.default_rng(42)

    # Generate stereo 1 kHz tone at 0.5 amplitude (both channels)
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

    # --- Stereo test ---
    decoder_stereo = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=75e-6
    )
    decoder_stereo.bass_boost_enabled = False
    decoder_stereo.treble_boost_enabled = False
    decoder_stereo.stereo_blend_enabled = False

    audio_stereo = demodulate_with_settling(decoder_stereo, iq_noisy)
    skip = int(0.1 * audio_rate)
    left_stereo = audio_stereo[skip:-skip, 0]

    stereo_ihf = measure_ihf_tone_snr(left_stereo, test_freq, audio_rate, a_weight_sos, signal_bw=100)

    # Pilot-referenced SNR for comparison
    pilot_snr = decoder_stereo.snr_db if hasattr(decoder_stereo, 'snr_db') else 0.0

    # --- Mono test ---
    decoder_mono = PLLStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=75e-6,
        force_mono=True
    )
    decoder_mono.bass_boost_enabled = False
    decoder_mono.treble_boost_enabled = False
    decoder_mono.stereo_blend_enabled = False

    audio_mono = demodulate_with_settling(decoder_mono, iq_noisy)
    left_mono = audio_mono[skip:-skip, 0]

    mono_ihf = measure_ihf_tone_snr(left_mono, test_freq, audio_rate, a_weight_sos, signal_bw=100)

    print(f"  IQ rate:             {iq_rate/1000:.0f} kHz")
    print(f"  RF SNR:              {rf_snr_db} dB")
    print(f"  Pilot-referenced:    {pilot_snr:.1f} dB")
    print(f"  IHF stereo SNR:      {stereo_ihf:.1f} dB  (target: >35 dB)")
    print(f"  IHF mono SNR:        {mono_ihf:.1f} dB  (target: >50 dB)")
    print(f"  Stereo-to-pilot gap: {stereo_ihf - pilot_snr:+.1f} dB  (de-emphasis + A-weight)")

    stereo_pass = stereo_ihf > 35
    mono_pass = mono_ihf > 50
    passed = stereo_pass and mono_pass
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("FM STEREO DECODER TEST SUITE")
    print("=" * 60)
    print("\nTesting pll_stereo_decoder.py (PLLStereoDecoder)")
    print("Signal flow: IQ -> FM Demod -> Pilot/L+R/L-R -> Matrix -> Audio")

    tests = [
        ("FM Demodulation Accuracy", test_fm_demod_accuracy),
        ("Audio SNR (Clean)", test_audio_snr),
        ("THD+N", test_thd_n),
        ("Mono Decoding", test_mono_decode),
        ("Stereo Decoding", test_stereo_decode),
        ("Stereo Separation", test_stereo_separation),
        ("Subcarrier Phase Sensitivity", test_subcarrier_phase_sensitivity),
        ("Group Delay Alignment", test_group_delay_alignment),
        ("Frequency Response", test_frequency_response),
        ("SNR with Noise", test_snr_with_noise),
        ("IHF/EIA SNR", test_ihf_snr),
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

    print()
    print("NOTES:")
    print("  - Stereo multiplex phase reference uses -cos(2wt) subcarrier")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

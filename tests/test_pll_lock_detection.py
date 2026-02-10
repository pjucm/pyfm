#!/usr/bin/env python3
"""
PLL lock detector behavior tests.

These tests encode the expected lock-gating behavior for amplitude-aware
pilot detection:
- Normal pilot level should lock.
- No pilot (or extremely weak pilot) should remain unlocked.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pll_stereo_decoder import PLLStereoDecoder


def _generate_fm_iq(
    duration_s=0.6,
    iq_sample_rate=480_000,
    pilot_amplitude=0.09,
    include_pilot=True,
):
    """Generate mono FM IQ with optional 19 kHz pilot."""
    n = int(duration_s * iq_sample_rate)
    t = np.arange(n, dtype=np.float64) / iq_sample_rate

    # Mono program material only (L-R = 0) to isolate pilot-lock behavior.
    mono = 0.5 * np.sin(2.0 * np.pi * 1_000.0 * t)
    pilot = (
        pilot_amplitude * np.sin(2.0 * np.pi * 19_000.0 * t)
        if include_pilot
        else np.zeros(n, dtype=np.float64)
    )

    multiplex = 0.9 * mono + pilot
    phase = 2.0 * np.pi * 75_000.0 * np.cumsum(multiplex) / iq_sample_rate
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


def _lock_fraction(iq_samples, block_size=2_048, warmup_blocks=8):
    """Return PLL lock fraction after warmup for a decode run."""
    decoder = PLLStereoDecoder(iq_sample_rate=480_000, audio_sample_rate=48_000)
    decoder.stereo_blend_enabled = False
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False

    lock_history = []
    for idx in range(0, len(iq_samples), block_size):
        block = iq_samples[idx:idx + block_size]
        if len(block) == 0:
            continue
        decoder.demodulate(block)
        lock_history.append(bool(decoder.pll_locked))

    if len(lock_history) <= warmup_blocks:
        return float(np.mean(lock_history)) if lock_history else 0.0
    return float(np.mean(lock_history[warmup_blocks:]))


def _generate_left_only_stereo_iq(
    duration_s=1.0,
    iq_sample_rate=192_000,
    left_freq_hz=1_000.0,
    left_amplitude=0.6,
    pilot_amplitude=0.09,
):
    """Generate FM stereo IQ with left-only program content plus pilot."""
    n = int(duration_s * iq_sample_rate)
    t = np.arange(n, dtype=np.float64) / iq_sample_rate
    left = left_amplitude * np.sin(2.0 * np.pi * left_freq_hz * t)
    right = np.zeros(n, dtype=np.float64)

    lr_sum = (left + right) / 2.0
    lr_diff = (left - right) / 2.0
    pilot = pilot_amplitude * np.sin(2.0 * np.pi * 19_000.0 * t)
    carrier_38k = -np.cos(2.0 * np.pi * 38_000.0 * t)

    multiplex = lr_sum * 0.9 + pilot + (lr_diff * carrier_38k) * 0.9
    phase = 2.0 * np.pi * 75_000.0 * np.cumsum(multiplex) / iq_sample_rate
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


def _goertzel_power(x, target_freq, sample_rate):
    """Measure narrowband tone power."""
    n = len(x)
    if n <= 0:
        return 0.0
    k = int(0.5 + n * target_freq / sample_rate)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return power / (n * n / 2.0)


def test_pll_lock_requires_nontrivial_pilot_amplitude():
    """
    Lock should depend on pilot amplitude, not just low phase-error energy.

    This guards against false-lock where no-pilot input decays the PE metric and
    causes pll_locked=True despite negligible pilot energy.
    """
    strong_lock_fraction = _lock_fraction(
        _generate_fm_iq(include_pilot=True, pilot_amplitude=0.09)
    )
    no_pilot_lock_fraction = _lock_fraction(
        _generate_fm_iq(include_pilot=False, pilot_amplitude=0.0)
    )
    weak_pilot_lock_fraction = _lock_fraction(
        _generate_fm_iq(include_pilot=True, pilot_amplitude=1e-4)
    )

    assert strong_lock_fraction >= 0.80, (
        f"Expected strong pilot to lock reliably, got {strong_lock_fraction:.3f}"
    )
    assert no_pilot_lock_fraction <= 0.05, (
        f"False lock with no pilot: lock_fraction={no_pilot_lock_fraction:.3f}"
    )
    assert weak_pilot_lock_fraction <= 0.10, (
        f"False lock with ultra-weak pilot: lock_fraction={weak_pilot_lock_fraction:.3f}"
    )


def test_pll_blend_quality_fallback_when_noise_band_unavailable():
    """
    At lower IQ rates, stereo blend should still engage when pilot quality is good.

    This protects the composite-quality fallback path: when the legacy 90-100 kHz
    noise-band metric is unavailable, blend must not collapse to mono for clean,
    strongly locked pilot conditions.
    """
    iq = _generate_left_only_stereo_iq(iq_sample_rate=192_000)

    decoder = PLLStereoDecoder(iq_sample_rate=192_000, audio_sample_rate=48_000)
    decoder.stereo_blend_enabled = True
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False

    audio_chunks = []
    for idx in range(0, len(iq), 2_048):
        audio_chunks.append(decoder.demodulate(iq[idx:idx + 2_048]))
    audio = np.vstack(audio_chunks)

    # Precondition for this test: legacy high-band SNR path is unavailable.
    assert decoder.noise_bpf is None
    assert decoder.pilot_detected

    skip = int(0.10 * 48_000)
    left = audio[skip:, 0]
    right = audio[skip:, 1]
    left_power = _goertzel_power(left, 1_000.0, 48_000)
    right_leak = _goertzel_power(right, 1_000.0, 48_000)
    separation_db = 10.0 * np.log10((left_power + 1e-20) / (right_leak + 1e-20))

    assert decoder.stereo_blend_factor >= 0.70, (
        f"Blend collapsed to mono at 192 kHz: {decoder.stereo_blend_factor:.3f}"
    )
    assert separation_db >= 5.0, (
        f"Expected non-trivial stereo separation with strong pilot, got {separation_db:.2f} dB"
    )


def test_pll_bandwidth_switches_from_acquisition_after_lock():
    """
    Adaptive loop bandwidth should start wide, then narrow once locked.

    This verifies acquisition->tracking/precision transition on a clean pilot.
    """
    decoder = PLLStereoDecoder(iq_sample_rate=480_000, audio_sample_rate=48_000)
    decoder.stereo_blend_enabled = False
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False

    assert decoder.pll_loop_bandwidth_hz >= 80.0, (
        f"Expected startup acquisition bandwidth, got {decoder.pll_loop_bandwidth_hz:.1f} Hz"
    )

    iq = _generate_fm_iq(duration_s=0.8, include_pilot=True, pilot_amplitude=0.09)
    for idx in range(0, len(iq), 2_048):
        decoder.demodulate(iq[idx:idx + 2_048])

    assert decoder.pilot_detected, "PLL failed to lock in strong-pilot condition"
    assert decoder.pll_loop_bandwidth_hz <= 35.0, (
        f"Expected narrowed loop bandwidth after lock, got {decoder.pll_loop_bandwidth_hz:.1f} Hz"
    )


def test_pll_bandwidth_returns_to_acquisition_when_pilot_lost():
    """
    Adaptive loop bandwidth should return to acquisition mode after unlock.
    """
    decoder = PLLStereoDecoder(iq_sample_rate=480_000, audio_sample_rate=48_000)
    decoder.stereo_blend_enabled = False
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False

    iq_lock = _generate_fm_iq(duration_s=0.6, include_pilot=True, pilot_amplitude=0.09)
    for idx in range(0, len(iq_lock), 2_048):
        decoder.demodulate(iq_lock[idx:idx + 2_048])

    assert decoder.pilot_detected, "Precondition failed: decoder did not lock"
    assert decoder.pll_loop_bandwidth_hz <= 35.0

    iq_lost = _generate_fm_iq(duration_s=0.6, include_pilot=False, pilot_amplitude=0.0)
    for idx in range(0, len(iq_lost), 2_048):
        decoder.demodulate(iq_lost[idx:idx + 2_048])

    assert not decoder.pilot_detected, "Expected unlock after pilot removal"
    assert decoder.pll_loop_bandwidth_hz >= 80.0, (
        f"Expected acquisition bandwidth after unlock, got {decoder.pll_loop_bandwidth_hz:.1f} Hz"
    )


def _audio_tone_snr_db(x, tone_hz=1_000.0, sample_rate=48_000):
    """Estimate tone-vs-residual SNR via FFT."""
    n = len(x)
    if n <= 0:
        return -120.0
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    power = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    tone_mask = np.abs(freqs - tone_hz) <= 50.0
    tone_power = np.sum(power[tone_mask])
    residual_power = np.sum(power[~tone_mask])
    return 10.0 * np.log10((tone_power + 1e-20) / (residual_power + 1e-20))


def test_pll_backend_auto_matches_python_for_snr_and_separation():
    """
    Fast PLL backend must preserve decode quality metrics vs Python reference.

    The backend may be Python (fallback) or Numba (when installed), but output
    quality should stay effectively unchanged.
    """
    iq = _generate_left_only_stereo_iq(iq_sample_rate=480_000, duration_s=1.0)

    dec_py = PLLStereoDecoder(
        iq_sample_rate=480_000,
        audio_sample_rate=48_000,
        pll_kernel_mode="python",
    )
    dec_auto = PLLStereoDecoder(
        iq_sample_rate=480_000,
        audio_sample_rate=48_000,
        pll_kernel_mode="auto",
    )
    for dec in (dec_py, dec_auto):
        dec.stereo_blend_enabled = False
        dec.bass_boost_enabled = False
        dec.treble_boost_enabled = False

    out_py = []
    out_auto = []
    for idx in range(0, len(iq), 2_048):
        block = iq[idx:idx + 2_048]
        out_py.append(dec_py.demodulate(block))
        out_auto.append(dec_auto.demodulate(block))
    audio_py = np.vstack(out_py)
    audio_auto = np.vstack(out_auto)

    assert dec_py.pll_backend == "python"
    assert dec_auto.pll_backend in {"python", "numba"}
    assert dec_py.pilot_detected == dec_auto.pilot_detected

    skip = int(0.10 * 48_000)
    left_py = audio_py[skip:, 0]
    right_py = audio_py[skip:, 1]
    left_auto = audio_auto[skip:, 0]
    right_auto = audio_auto[skip:, 1]

    sep_py = 10.0 * np.log10(
        (_goertzel_power(left_py, 1_000.0, 48_000) + 1e-20)
        / (_goertzel_power(right_py, 1_000.0, 48_000) + 1e-20)
    )
    sep_auto = 10.0 * np.log10(
        (_goertzel_power(left_auto, 1_000.0, 48_000) + 1e-20)
        / (_goertzel_power(right_auto, 1_000.0, 48_000) + 1e-20)
    )
    snr_py = _audio_tone_snr_db(left_py, tone_hz=1_000.0, sample_rate=48_000)
    snr_auto = _audio_tone_snr_db(left_auto, tone_hz=1_000.0, sample_rate=48_000)

    assert abs(sep_auto - sep_py) <= 0.5, (
        f"Separation drift too large: python={sep_py:.2f} dB auto={sep_auto:.2f} dB"
    )
    assert abs(snr_auto - snr_py) <= 0.5, (
        f"SNR drift too large: python={snr_py:.2f} dB auto={snr_auto:.2f} dB"
    )

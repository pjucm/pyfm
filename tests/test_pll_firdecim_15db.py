#!/usr/bin/env python3
"""
Standalone 15 dB RF-SNR matrix for stereo decoder/resampler evaluation.

Matrix:
- Decoder: PLLStereoDecoder
- Resamplers: interp, firdecim

Metrics:
- IHF/EIA-style audio SNR (A-weighted, de-emphasized)
- Stereo separation (left-only program, 1 kHz leak ratio)
"""

import configparser
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pll_stereo_decoder import PLLStereoDecoder


PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PJFM_CFG = os.path.join(PROJECT_ROOT, "pjfm.cfg")

RF_SNR_DB = 15.0
AUDIO_SAMPLE_RATE = 48_000
TEST_FREQ_HZ = 1_000.0
TEST_DURATION_S = 2.0
SKIP_AUDIO_S = 0.30
DEFAULT_BLOCK_SIZE = 8192
BLOCK_SIZE_SWEEP = (1024, 2048, 4096, 8192)

DECODER_MATRIX = (
    ("PLLStereoDecoder", PLLStereoDecoder),
)
RESAMPLER_MODES = ("interp", "firdecim")


def _load_settings():
    """Load relevant defaults from pjfm.cfg with safe fallbacks."""
    settings = {
        "iq_sample_rate": 480_000,
        "stereo_lpf_taps": 255,
        "stereo_lpf_beta": 6.0,
        "stereo_resampler_taps": 127,
        "stereo_resampler_beta": 8.0,
    }
    if not os.path.exists(PJFM_CFG):
        return settings

    cfg = configparser.ConfigParser()
    cfg.read(PJFM_CFG)
    if not cfg.has_section("radio"):
        return settings

    radio = cfg["radio"]
    if "iq_sample_rate" in radio:
        settings["iq_sample_rate"] = int(radio.get("iq_sample_rate"))
    if "stereo_lpf_taps" in radio:
        settings["stereo_lpf_taps"] = int(radio.get("stereo_lpf_taps"))
    if "stereo_lpf_beta" in radio:
        settings["stereo_lpf_beta"] = float(radio.get("stereo_lpf_beta"))
    if "stereo_resampler_taps" in radio:
        settings["stereo_resampler_taps"] = int(radio.get("stereo_resampler_taps"))
    if "stereo_resampler_beta" in radio:
        settings["stereo_resampler_beta"] = float(radio.get("stereo_resampler_beta"))
    return settings


def _generate_fm_stereo_multiplex(left, right, sample_rate):
    n = len(left)
    t = np.arange(n, dtype=np.float64) / sample_rate
    lr_sum = (left + right) / 2.0
    lr_diff = (left - right) / 2.0
    pilot = 0.09 * np.sin(2.0 * np.pi * 19_000.0 * t)
    carrier_38k = -np.cos(2.0 * np.pi * 38_000.0 * t)
    return lr_sum * 0.9 + pilot + (lr_diff * carrier_38k) * 0.9


def _fm_modulate(baseband, sample_rate, deviation_hz=75_000.0):
    phase = 2.0 * np.pi * deviation_hz * np.cumsum(baseband) / sample_rate
    return (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)


def _add_awgn_for_rf_snr(iq_clean, rf_snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.abs(iq_clean) ** 2)
    noise_power = signal_power / (10.0 ** (rf_snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(iq_clean)) + 1j * rng.standard_normal(len(iq_clean))
    )
    return iq_clean + noise.astype(np.complex64)


def _demodulate_stream(decoder, iq_samples, block_size=DEFAULT_BLOCK_SIZE):
    chunks = []
    for idx in range(0, len(iq_samples), block_size):
        chunks.append(decoder.demodulate(iq_samples[idx:idx + block_size]))
    return np.vstack(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)


def _design_a_weighting(fs):
    z = [0, 0, 0, 0]
    p = [
        -2 * np.pi * 20.598997,
        -2 * np.pi * 20.598997,
        -2 * np.pi * 107.65265,
        -2 * np.pi * 737.86223,
        -2 * np.pi * 12194.217,
        -2 * np.pi * 12194.217,
    ]
    k = (2 * np.pi * 12194.217) ** 2
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    _, h = signal.sosfreqz(sos, worN=[2 * np.pi * 1000 / fs])
    sos[0, :3] /= np.abs(h[0])
    return sos


def _measure_ihf_tone_snr(audio_channel, tone_hz, sample_rate_hz):
    a_weight_sos = _design_a_weighting(sample_rate_hz)
    weighted = signal.sosfilt(a_weight_sos, audio_channel)
    n = len(weighted)
    x = weighted * np.hanning(n)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    power = np.abs(fft) ** 2 / n
    tone_mask = np.abs(freqs - tone_hz) <= 50.0
    tone_power = np.sum(power[tone_mask])
    residual_power = np.sum(power[~tone_mask])
    return 10.0 * np.log10((tone_power + 1e-20) / (residual_power + 1e-20))


def _goertzel_power(x, target_freq, sample_rate):
    n = len(x)
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


def _build_decoder(settings, decoder_class, resampler_mode):
    decoder = decoder_class(
        iq_sample_rate=settings["iq_sample_rate"],
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        deviation=75_000,
        deemphasis=75e-6,
        stereo_lpf_taps=settings["stereo_lpf_taps"],
        stereo_lpf_beta=settings["stereo_lpf_beta"],
        resampler_mode=resampler_mode,
        resampler_taps=settings["stereo_resampler_taps"],
        resampler_beta=settings["stereo_resampler_beta"],
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = True
    return decoder


def _expected_runtime_mode(requested_mode):
    if requested_mode == "firdecim":
        return "firdecim"
    return "interp"


def _run_matrix(block_size=DEFAULT_BLOCK_SIZE):
    settings = _load_settings()
    iq_rate = settings["iq_sample_rate"]
    n = int(TEST_DURATION_S * iq_rate)
    t = np.arange(n, dtype=np.float64) / iq_rate
    tone = 0.5 * np.sin(2.0 * np.pi * TEST_FREQ_HZ * t)
    zero = np.zeros_like(tone)

    # Build shared noisy stimuli once so all matrix entries compare apples-to-apples.
    mpx_snr = _generate_fm_stereo_multiplex(tone, tone, iq_rate)
    iq_clean_snr = _fm_modulate(mpx_snr, iq_rate)
    iq_noisy_snr = _add_awgn_for_rf_snr(iq_clean_snr, RF_SNR_DB, seed=20260215)

    mpx_sep = _generate_fm_stereo_multiplex(tone, zero, iq_rate)
    iq_clean_sep = _fm_modulate(mpx_sep, iq_rate)
    iq_noisy_sep = _add_awgn_for_rf_snr(iq_clean_sep, RF_SNR_DB, seed=20261215)

    skip = int(SKIP_AUDIO_S * AUDIO_SAMPLE_RATE)
    rows = []

    for decoder_name, decoder_class in DECODER_MATRIX:
        for requested_mode in RESAMPLER_MODES:
            dec_snr = _build_decoder(settings, decoder_class, requested_mode)
            dec_sep = _build_decoder(settings, decoder_class, requested_mode)

            audio_snr = _demodulate_stream(dec_snr, iq_noisy_snr, block_size=block_size)
            audio_sep = _demodulate_stream(dec_sep, iq_noisy_sep, block_size=block_size)

            left_snr = audio_snr[skip:-skip, 0]
            ihf_snr_db = _measure_ihf_tone_snr(left_snr, TEST_FREQ_HZ, AUDIO_SAMPLE_RATE)

            left_sep = audio_sep[skip:-skip, 0]
            right_sep = audio_sep[skip:-skip, 1]
            left_power = _goertzel_power(left_sep, TEST_FREQ_HZ, AUDIO_SAMPLE_RATE)
            right_leak = _goertzel_power(right_sep, TEST_FREQ_HZ, AUDIO_SAMPLE_RATE)
            separation_db = 10.0 * np.log10((left_power + 1e-20) / (right_leak + 1e-20))

            rows.append(
                {
                    "decoder": decoder_name,
                    "requested_resampler": requested_mode,
                    "runtime_resampler_snr": str(
                        getattr(dec_snr, "_resampler_runtime_mode", "unknown")
                    ),
                    "runtime_resampler_sep": str(
                        getattr(dec_sep, "_resampler_runtime_mode", "unknown")
                    ),
                    "ihf_snr_db": float(ihf_snr_db),
                    "stereo_separation_db": float(separation_db),
                    "pilot_snr_db": float(dec_snr.snr_db),
                    "blend_factor": float(dec_snr.stereo_blend_factor),
                    "pilot_detected": bool(dec_snr.pilot_detected),
                    "iq_sample_rate": int(iq_rate),
                    "block_size": int(block_size),
                    "resampler_taps": int(settings["stereo_resampler_taps"]),
                    "resampler_beta": float(settings["stereo_resampler_beta"]),
                }
            )

    return rows


def test_decoder_resampler_matrix_at_15db_rf_snr():
    rows = _run_matrix(block_size=DEFAULT_BLOCK_SIZE)

    for row in rows:
        expected_runtime = _expected_runtime_mode(row["requested_resampler"])
        assert row["runtime_resampler_snr"] == expected_runtime
        assert row["runtime_resampler_sep"] == expected_runtime
        assert row["pilot_detected"]
        assert np.isfinite(row["ihf_snr_db"])
        assert np.isfinite(row["stereo_separation_db"])
        assert 0.0 <= row["blend_factor"] <= 1.0
        assert row["ihf_snr_db"] >= 20.0
        assert row["stereo_separation_db"] >= -1.0

    # At this RF SNR with blend active, we still expect at least one combo to
    # retain non-trivial stereo information.
    assert max(row["stereo_separation_db"] for row in rows) >= 3.0


def test_decoder_resampler_matrix_block_size_invariance_at_15db_rf_snr():
    # Evaluate a small block-size sweep and ensure metrics remain close.
    # This guards against regressions where decoder quality depends on how
    # input IQ is chunked rather than signal content.
    by_block = {bs: _run_matrix(block_size=bs) for bs in BLOCK_SIZE_SWEEP}

    grouped = {}
    for block_size, rows in by_block.items():
        for row in rows:
            key = (row["decoder"], row["requested_resampler"])
            bucket = grouped.setdefault(key, {"ihf": [], "sep": []})
            bucket["ihf"].append(row["ihf_snr_db"])
            bucket["sep"].append(row["stereo_separation_db"])

    for key, metrics in grouped.items():
        ihf_span = max(metrics["ihf"]) - min(metrics["ihf"])
        sep_span = max(metrics["sep"]) - min(metrics["sep"])
        assert ihf_span <= 0.75, f"{key} IHF span too large across blocks: {ihf_span:.2f} dB"
        assert sep_span <= 0.75, f"{key} separation span too large across blocks: {sep_span:.2f} dB"


if __name__ == "__main__":
    rows = _run_matrix(block_size=DEFAULT_BLOCK_SIZE)
    iq_rate = rows[0]["iq_sample_rate"] if rows else 0
    block_size = rows[0]["block_size"] if rows else 0
    taps = rows[0]["resampler_taps"] if rows else 0
    beta = rows[0]["resampler_beta"] if rows else 0.0

    print(f"RF SNR: {RF_SNR_DB:.1f} dB")
    print(f"IQ sample rate: {iq_rate/1000:.0f} kHz")
    print(f"Demod block size: {block_size}")
    print(f"Resampler taps/beta: {taps} / {beta:.1f}")
    print("")
    print(
        f"{'Decoder':18s} {'Req':8s} {'Runtime':8s} "
        f"{'IHF SNR':>8s} {'Sep':>8s} {'PilotSNR':>9s} {'Blend':>7s} {'Pilot':>6s}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row['decoder']:18s} "
            f"{row['requested_resampler']:8s} "
            f"{row['runtime_resampler_snr']:8s} "
            f"{row['ihf_snr_db']:8.2f} "
            f"{row['stereo_separation_db']:8.2f} "
            f"{row['pilot_snr_db']:9.2f} "
            f"{row['blend_factor']:7.2f} "
            f"{'Yes' if row['pilot_detected'] else 'No':>6s}"
        )

    print("")
    print("Block-size invariance (span across 1024/2048/4096/8192):")
    by_block = {bs: _run_matrix(block_size=bs) for bs in BLOCK_SIZE_SWEEP}
    grouped = {}
    for rows_bs in by_block.values():
        for row in rows_bs:
            key = (row["decoder"], row["requested_resampler"])
            bucket = grouped.setdefault(key, {"ihf": [], "sep": []})
            bucket["ihf"].append(row["ihf_snr_db"])
            bucket["sep"].append(row["stereo_separation_db"])
    for key, metrics in grouped.items():
        ihf_span = max(metrics["ihf"]) - min(metrics["ihf"])
        sep_span = max(metrics["sep"]) - min(metrics["sep"])
        print(
            f"  {key[0]:18s} {key[1]:8s} "
            f"IHF span={ihf_span:.2f} dB, Sep span={sep_span:.2f} dB"
        )

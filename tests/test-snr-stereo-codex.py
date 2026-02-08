#!/usr/bin/env python3
"""Clean-room SNR and stereo-separation test bench for FM stereo decoders."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

from demodulator import FMStereoDecoder
from pll_stereo_decoder import PLLStereoDecoder


IQ_SAMPLE_RATE = 250_000
AUDIO_SAMPLE_RATE = 48_000
FM_DEVIATION_HZ = 75_000

TEST_DURATION_S = 1.8
CHUNK_SAMPLES = 25_000
TEST_TONE_HZ = 1_000.0
WARMUP_AUDIO_S = 1.0


def _synthesize_stereo_fm_iq(
    rf_snr_db,
    *,
    left_amp=0.45,
    right_amp=0.0,
    pilot_amp=0.12,
    tone_hz=TEST_TONE_HZ,
    phase_noise_sigma_rad=0.0,
    phase_noise_pole=0.9995,
    seed=0,
):
    """Build noisy FM IQ from an ideal stereo multiplex waveform."""
    n = int(TEST_DURATION_S * IQ_SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / IQ_SAMPLE_RATE

    left = left_amp * np.sin(2.0 * np.pi * tone_hz * t)
    right = right_amp * np.sin(2.0 * np.pi * 1_500.0 * t)

    # Stereo MPX: (L+R), 19 kHz pilot, and (L-R) on 38 kHz DSB-SC.
    stereo_mpx = (
        0.5 * (left + right)
        + pilot_amp * np.cos(2.0 * np.pi * 19_000.0 * t)
        + 0.5 * (left - right) * np.cos(2.0 * np.pi * 38_000.0 * t)
    )

    phase = np.cumsum(2.0 * np.pi * FM_DEVIATION_HZ * stereo_mpx / IQ_SAMPLE_RATE)
    if phase_noise_sigma_rad > 0.0:
        phase_noise = _ar1_phase_noise(
            n,
            sigma_rad=phase_noise_sigma_rad,
            pole=phase_noise_pole,
            seed=seed + 100_000,
        )
        phase = phase + phase_noise
    iq_clean = np.exp(1j * phase)

    rng = np.random.default_rng(seed)
    noise_power = 10.0 ** (-rf_snr_db / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return (iq_clean + noise).astype(np.complex64)


def _ar1_phase_noise(n, *, sigma_rad, pole, seed):
    """Generate correlated phase jitter with a first-order AR process."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    phase_noise = np.zeros(n, dtype=np.float64)
    for idx in range(1, n):
        phase_noise[idx] = pole * phase_noise[idx - 1] + sigma_rad * white[idx]
    return phase_noise


def _decode_stream(decoder_cls, iq_samples):
    """Run decoder in streaming chunks and return (decoder_instance, audio)."""
    decoder = decoder_cls(
        iq_sample_rate=IQ_SAMPLE_RATE,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        deviation=FM_DEVIATION_HZ,
    )

    # Keep the test focused on stereo decode quality only.
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False
    decoder.stereo_blend_low = -100.0

    out = []
    for start in range(0, len(iq_samples), CHUNK_SAMPLES):
        out.append(decoder.demodulate(iq_samples[start:start + CHUNK_SAMPLES]))

    return decoder, np.vstack(out)


def _tone_power(signal, tone_hz, sample_rate_hz):
    """Estimate single-tone power using IQ projection."""
    n = len(signal)
    if n == 0:
        return 0.0
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    c = np.cos(2.0 * np.pi * tone_hz * t)
    s = np.sin(2.0 * np.pi * tone_hz * t)
    i = (2.0 / n) * np.dot(signal, c)
    q = (2.0 / n) * np.dot(signal, s)
    return float(np.hypot(i, q) ** 2)


def _stereo_separation_db(audio_stereo, tone_hz):
    """Compute left-vs-right crosstalk for left-only program audio."""
    trim = int(WARMUP_AUDIO_S * AUDIO_SAMPLE_RATE)
    analysis = audio_stereo[trim:]
    if len(analysis) < int(0.2 * AUDIO_SAMPLE_RATE):
        raise AssertionError("Not enough post-warmup audio for stereo-separation analysis")

    left_pwr = _tone_power(analysis[:, 0], tone_hz, AUDIO_SAMPLE_RATE)
    right_leak_pwr = _tone_power(analysis[:, 1], tone_hz, AUDIO_SAMPLE_RATE)
    return 10.0 * np.log10((left_pwr + 1e-20) / (right_leak_pwr + 1e-20))


def _run_decoder_checks(decoder_cls, *, min_separation_db):
    high_snr_iq = _synthesize_stereo_fm_iq(35.0, seed=101)
    low_snr_iq = _synthesize_stereo_fm_iq(20.0, seed=202)
    separation_iq = _synthesize_stereo_fm_iq(45.0, seed=303)

    high_snr_decoder, _ = _decode_stream(decoder_cls, high_snr_iq)
    low_snr_decoder, _ = _decode_stream(decoder_cls, low_snr_iq)
    sep_decoder, sep_audio = _decode_stream(decoder_cls, separation_iq)

    snr_delta = high_snr_decoder.snr_db - low_snr_decoder.snr_db
    separation_db = _stereo_separation_db(sep_audio, TEST_TONE_HZ)

    result = {
        "kind": "snr_stereo",
        "decoder": decoder_cls.__name__,
        "snr_high_db": float(high_snr_decoder.snr_db),
        "snr_low_db": float(low_snr_decoder.snr_db),
        "snr_delta_db": float(snr_delta),
        "separation_db": float(separation_db),
        "pilot_detected": bool(sep_decoder.pilot_detected),
        "min_separation_db": float(min_separation_db),
    }

    assert np.isfinite(high_snr_decoder.snr_db), "High-SNR run produced non-finite decoder SNR"
    assert np.isfinite(low_snr_decoder.snr_db), "Low-SNR run produced non-finite decoder SNR"
    assert snr_delta > 10.0, (
        f"{decoder_cls.__name__}: decoder SNR did not track RF noise change "
        f"(delta={snr_delta:.2f} dB)"
    )
    assert sep_decoder.pilot_detected, f"{decoder_cls.__name__}: pilot was not detected in clean run"
    assert separation_db >= min_separation_db, (
        f"{decoder_cls.__name__}: stereo separation too low "
        f"({separation_db:.2f} dB < {min_separation_db:.2f} dB)"
    )
    return result


def _run_phase_noise_checks(
    decoder_cls,
    *,
    phase_noise_sigma_rad,
    phase_noise_pole,
    min_noisy_separation_db,
    max_separation_loss_db,
):
    clean_iq = _synthesize_stereo_fm_iq(45.0, seed=404)
    noisy_iq = _synthesize_stereo_fm_iq(
        45.0,
        phase_noise_sigma_rad=phase_noise_sigma_rad,
        phase_noise_pole=phase_noise_pole,
        seed=404,
    )

    clean_decoder, clean_audio = _decode_stream(decoder_cls, clean_iq)
    noisy_decoder, noisy_audio = _decode_stream(decoder_cls, noisy_iq)

    clean_sep_db = _stereo_separation_db(clean_audio, TEST_TONE_HZ)
    noisy_sep_db = _stereo_separation_db(noisy_audio, TEST_TONE_HZ)
    sep_loss_db = clean_sep_db - noisy_sep_db

    result = {
        "kind": "phase_noise",
        "decoder": decoder_cls.__name__,
        "phase_noise_sigma_rad": float(phase_noise_sigma_rad),
        "phase_noise_pole": float(phase_noise_pole),
        "clean_separation_db": float(clean_sep_db),
        "noisy_separation_db": float(noisy_sep_db),
        "separation_loss_db": float(sep_loss_db),
        "noisy_snr_db": float(noisy_decoder.snr_db),
        "pilot_detected": bool(noisy_decoder.pilot_detected),
        "min_noisy_separation_db": float(min_noisy_separation_db),
        "max_separation_loss_db": float(max_separation_loss_db),
    }

    assert clean_decoder.pilot_detected, f"{decoder_cls.__name__}: pilot was not detected in clean phase-noise baseline"
    assert noisy_decoder.pilot_detected, f"{decoder_cls.__name__}: pilot was lost with phase noise"
    assert np.isfinite(noisy_decoder.snr_db), f"{decoder_cls.__name__}: noisy-phase SNR is non-finite"
    assert noisy_sep_db >= min_noisy_separation_db, (
        f"{decoder_cls.__name__}: separation under phase noise too low "
        f"({noisy_sep_db:.2f} dB < {min_noisy_separation_db:.2f} dB)"
    )
    assert sep_loss_db <= max_separation_loss_db, (
        f"{decoder_cls.__name__}: phase-noise separation loss too high "
        f"({sep_loss_db:.2f} dB > {max_separation_loss_db:.2f} dB)"
    )
    return result


def _print_result(result):
    if result["kind"] == "snr_stereo":
        message = (
            f"{result['decoder']}: "
            f"snr_high={result['snr_high_db']:.2f} dB, "
            f"snr_low={result['snr_low_db']:.2f} dB, "
            f"snr_delta={result['snr_delta_db']:.2f} dB, "
            f"separation={result['separation_db']:.2f} dB "
            f"(min {result['min_separation_db']:.2f} dB), "
            f"pilot_detected={result['pilot_detected']}"
        )
    elif result["kind"] == "phase_noise":
        message = (
            f"{result['decoder']}: "
            f"phase_noise_sigma={result['phase_noise_sigma_rad']:.5f} rad/sample, "
            f"phase_noise_pole={result['phase_noise_pole']:.5f}, "
            f"clean_sep={result['clean_separation_db']:.2f} dB, "
            f"noisy_sep={result['noisy_separation_db']:.2f} dB, "
            f"sep_loss={result['separation_loss_db']:.2f} dB "
            f"(max {result['max_separation_loss_db']:.2f} dB), "
            f"noisy_snr={result['noisy_snr_db']:.2f} dB, "
            f"pilot_detected={result['pilot_detected']}"
        )
    else:
        raise ValueError(f"Unknown result kind: {result['kind']}")

    print(message, flush=True)


def test_fm_stereo_decoder_snr_and_stereo_separation():
    result = _run_decoder_checks(FMStereoDecoder, min_separation_db=35.0)
    _print_result(result)


def test_pll_stereo_decoder_snr_and_stereo_separation():
    result = _run_decoder_checks(PLLStereoDecoder, min_separation_db=45.0)
    _print_result(result)


def test_fm_stereo_decoder_phase_noise():
    result = _run_phase_noise_checks(
        FMStereoDecoder,
        phase_noise_sigma_rad=0.02,
        phase_noise_pole=0.9995,
        min_noisy_separation_db=24.0,
        max_separation_loss_db=30.0,
    )
    _print_result(result)


def test_pll_stereo_decoder_phase_noise():
    result = _run_phase_noise_checks(
        PLLStereoDecoder,
        phase_noise_sigma_rad=0.02,
        phase_noise_pole=0.9995,
        min_noisy_separation_db=50.0,
        max_separation_loss_db=10.0,
    )
    _print_result(result)


if __name__ == "__main__":
    fm_result = _run_decoder_checks(FMStereoDecoder, min_separation_db=35.0)
    _print_result(fm_result)
    print("FMStereoDecoder: PASS")

    pll_result = _run_decoder_checks(PLLStereoDecoder, min_separation_db=45.0)
    _print_result(pll_result)
    print("PLLStereoDecoder: PASS")

    fm_phase_result = _run_phase_noise_checks(
        FMStereoDecoder,
        phase_noise_sigma_rad=0.02,
        phase_noise_pole=0.9995,
        min_noisy_separation_db=24.0,
        max_separation_loss_db=30.0,
    )
    _print_result(fm_phase_result)
    print("FMStereoDecoder phase-noise: PASS")

    pll_phase_result = _run_phase_noise_checks(
        PLLStereoDecoder,
        phase_noise_sigma_rad=0.02,
        phase_noise_pole=0.9995,
        min_noisy_separation_db=50.0,
        max_separation_loss_db=10.0,
    )
    _print_result(pll_phase_result)
    print("PLLStereoDecoder phase-noise: PASS")

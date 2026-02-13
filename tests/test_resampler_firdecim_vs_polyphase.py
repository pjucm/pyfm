#!/usr/bin/env python3
"""
Compare firdecim and polyphase resamplers in the stereo decode path.

This benchmark uses the current pjfm.cfg FIR defaults:
- stereo_resampler_taps
- stereo_resampler_beta
- stereo_lpf_taps
- stereo_lpf_beta
- iq_sample_rate

Metrics:
- Audio SNR at 1 kHz (left channel tone vs in-band residual)
- Stereo separation at 1 kHz (left tone leakage into right channel)
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

AUDIO_SAMPLE_RATE = 48_000
FM_DEVIATION_HZ = 75_000
TEST_DURATION_S = 2.8
TONE_HZ = 1_000.0

# Separate runs to isolate metrics.
RF_SNR_FOR_AUDIO_DB = 30.0
RF_SNR_FOR_SEPARATION_DB = 60.0

# Trim filter/PLL startup transients before analysis.
ANALYSIS_TRIM_S = 0.45


def _load_pjfm_defaults():
    """Load resampler and stereo LPF defaults from pjfm.cfg."""
    defaults = {
        "iq_sample_rate": 480_000,
        "stereo_lpf_taps": 127,
        "stereo_lpf_beta": 6.0,
        "stereo_resampler_mode": "firdecim",
        "stereo_resampler_taps": 127,
        "stereo_resampler_beta": 8.0,
    }

    if not os.path.exists(PJFM_CFG):
        return defaults

    cfg = configparser.ConfigParser()
    cfg.read(PJFM_CFG)
    if not cfg.has_section("radio"):
        return defaults

    radio = cfg["radio"]
    try:
        if "iq_sample_rate" in radio:
            defaults["iq_sample_rate"] = int(radio.get("iq_sample_rate"))
        if "stereo_lpf_taps" in radio:
            defaults["stereo_lpf_taps"] = int(radio.get("stereo_lpf_taps"))
        if "stereo_lpf_beta" in radio:
            defaults["stereo_lpf_beta"] = float(radio.get("stereo_lpf_beta"))
        if "stereo_resampler_mode" in radio:
            defaults["stereo_resampler_mode"] = radio.get("stereo_resampler_mode").strip().lower()
        if "stereo_resampler_taps" in radio:
            defaults["stereo_resampler_taps"] = int(radio.get("stereo_resampler_taps"))
        if "stereo_resampler_beta" in radio:
            defaults["stereo_resampler_beta"] = float(radio.get("stereo_resampler_beta"))
    except ValueError as exc:
        raise AssertionError(f"Invalid value in {PJFM_CFG}: {exc}") from exc

    return defaults


def _synthesize_left_only_fm_iq(iq_sample_rate, rf_snr_db, seed):
    """
    Generate synthetic broadcast-FM IQ with a left-only 1 kHz program tone.
    """
    n = int(TEST_DURATION_S * iq_sample_rate)
    t = np.arange(n, dtype=np.float64) / iq_sample_rate

    left = 0.45 * np.sin(2.0 * np.pi * TONE_HZ * t)
    right = np.zeros_like(left)

    # FM stereo multiplex:
    # - (L+R)/2 baseband
    # - 19 kHz pilot
    # - (L-R)/2 on -cos(38 kHz) DSB-SC subcarrier
    mpx = (
        0.5 * (left + right)
        + 0.09 * np.sin(2.0 * np.pi * 19_000.0 * t)
        - 0.5 * (left - right) * np.cos(2.0 * np.pi * 38_000.0 * t)
    )

    phase = 2.0 * np.pi * FM_DEVIATION_HZ * np.cumsum(mpx) / iq_sample_rate
    iq_clean = np.exp(1j * phase)

    rng = np.random.default_rng(seed)
    noise_power = 10.0 ** (-rf_snr_db / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return (iq_clean + noise).astype(np.complex64)


class _PolyphaseResamplerMixin:
    """
    Swap the decoder's firdecim path with a polyphase reference.

    The filter uses the same tap count, beta, and normalized cutoff as firdecim:
      firwin(taps, 0.45 / decimation, window=("kaiser", beta))
    """

    def _configure_resampler(self):
        ratio_in_out = self.iq_sample_rate / self.audio_sample_rate
        decim = int(round(ratio_in_out))
        integer_ratio = abs(ratio_in_out - decim) < 1e-9 and decim >= 2
        if not integer_ratio:
            raise AssertionError(
                "Polyphase benchmark requires integer iq/audio ratio >= 2 "
                f"(got {ratio_in_out:.8f})"
            )

        self._poly_decimation = decim
        self._poly_taps = signal.firwin(
            self.resampler_taps,
            0.45 / decim,
            window=("kaiser", self.resampler_beta),
        )
        self._resampler_runtime_mode = "polyphase"

        # Keep these members consistent with parent classes.
        self._fir_decimation = decim
        self._fir_decim_l = None
        self._fir_decim_r = None

    def _resample_channels(self, left, right, stereo_allowed):
        ratio_eff = self._nominal_ratio * self._rate_adjust

        left = signal.resample_poly(
            left,
            up=1,
            down=self._poly_decimation,
            window=self._poly_taps,
            padtype="line",
        )
        if stereo_allowed:
            right = signal.resample_poly(
                right,
                up=1,
                down=self._poly_decimation,
                window=self._poly_taps,
                padtype="line",
            )
        else:
            right = left

        # Preserve adaptive-rate behavior for non-unity rate adjust.
        ratio_post = ratio_eff * self._poly_decimation
        if abs(ratio_post - 1.0) > 1e-12:
            if stereo_allowed:
                left, right = self._post_decim_resampler.process(ratio_post, left, right)
            else:
                (left,) = self._post_decim_resampler.process(ratio_post, left)
                right = left

        return left, right


class PolyphasePLLStereoDecoder(_PolyphaseResamplerMixin, PLLStereoDecoder):
    """PLLStereoDecoder with polyphase output resampling."""


def _build_decoder(decoder_cls, settings):
    decoder = decoder_cls(
        iq_sample_rate=settings["iq_sample_rate"],
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        deviation=FM_DEVIATION_HZ,
        stereo_lpf_taps=settings["stereo_lpf_taps"],
        stereo_lpf_beta=settings["stereo_lpf_beta"],
        resampler_mode=settings["stereo_resampler_mode"],
        resampler_taps=settings["stereo_resampler_taps"],
        resampler_beta=settings["stereo_resampler_beta"],
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False
    decoder.stereo_blend_low = -100.0
    decoder.stereo_blend_high = -90.0
    # Synthetic pilot levels in this bench are below the default threshold;
    # lower the threshold so the decoder is evaluated in true stereo mode
    # for the resampler comparison.
    if hasattr(decoder, "pilot_threshold"):
        decoder.pilot_threshold = 0.002
    return decoder


def _analysis_audio(audio):
    trim = int(ANALYSIS_TRIM_S * AUDIO_SAMPLE_RATE)
    if len(audio) <= 2 * trim + 1024:
        raise AssertionError("Not enough decoded audio after analysis trimming")
    return audio[trim:-trim]


def _tone_power(x, tone_hz, sample_rate_hz, half_bins=2):
    n = len(x)
    xw = x * np.hanning(n)
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    power = np.abs(spec) ** 2

    tone_bin = int(np.argmin(np.abs(freqs - tone_hz)))
    lo = max(0, tone_bin - half_bins)
    hi = min(len(power), tone_bin + half_bins + 1)
    tone_mask = np.zeros(len(power), dtype=bool)
    tone_mask[lo:hi] = True
    return power, tone_mask, freqs


def _audio_snr_db(x, tone_hz, sample_rate_hz):
    power, tone_mask, freqs = _tone_power(x, tone_hz, sample_rate_hz)
    inband = freqs <= 15_000.0
    signal_power = np.sum(power[tone_mask & inband])
    noise_power = np.sum(power[(~tone_mask) & inband])
    return 10.0 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20))


def _stereo_separation_db(stereo_audio, tone_hz, sample_rate_hz):
    left_power, left_tone_mask, _ = _tone_power(stereo_audio[:, 0], tone_hz, sample_rate_hz)
    right_power, right_tone_mask, _ = _tone_power(stereo_audio[:, 1], tone_hz, sample_rate_hz)
    left_tone = np.sum(left_power[left_tone_mask])
    right_leak = np.sum(right_power[right_tone_mask])
    return 10.0 * np.log10((left_tone + 1e-20) / (right_leak + 1e-20))


def _run_bench(decoder_name, fir_cls, poly_cls, settings):
    iq_for_snr = _synthesize_left_only_fm_iq(
        settings["iq_sample_rate"], RF_SNR_FOR_AUDIO_DB, seed=20260208
    )
    iq_for_sep = _synthesize_left_only_fm_iq(
        settings["iq_sample_rate"], RF_SNR_FOR_SEPARATION_DB, seed=20260209
    )

    fir = _build_decoder(fir_cls, settings)
    poly = _build_decoder(poly_cls, settings)

    audio_fir_snr = _analysis_audio(fir.demodulate(iq_for_snr))
    audio_poly_snr = _analysis_audio(poly.demodulate(iq_for_snr))

    fir_sep_decoder = _build_decoder(fir_cls, settings)
    poly_sep_decoder = _build_decoder(poly_cls, settings)
    audio_fir_sep = _analysis_audio(fir_sep_decoder.demodulate(iq_for_sep))
    audio_poly_sep = _analysis_audio(poly_sep_decoder.demodulate(iq_for_sep))

    fir_snr = _audio_snr_db(audio_fir_snr[:, 0], TONE_HZ, AUDIO_SAMPLE_RATE)
    poly_snr = _audio_snr_db(audio_poly_snr[:, 0], TONE_HZ, AUDIO_SAMPLE_RATE)
    fir_sep = _stereo_separation_db(audio_fir_sep, TONE_HZ, AUDIO_SAMPLE_RATE)
    poly_sep = _stereo_separation_db(audio_poly_sep, TONE_HZ, AUDIO_SAMPLE_RATE)

    result = {
        "decoder": decoder_name,
        "firdecim_snr_db": float(fir_snr),
        "polyphase_snr_db": float(poly_snr),
        "snr_delta_db": float(poly_snr - fir_snr),
        "firdecim_separation_db": float(fir_sep),
        "polyphase_separation_db": float(poly_sep),
        "separation_delta_db": float(poly_sep - fir_sep),
        "firdecim_runtime_mode": str(getattr(fir, "_resampler_runtime_mode", "unknown")),
        "polyphase_runtime_mode": str(getattr(poly, "_resampler_runtime_mode", "unknown")),
    }

    print(
        f"{decoder_name}: "
        f"SNR firdecim={result['firdecim_snr_db']:.2f} dB, "
        f"polyphase={result['polyphase_snr_db']:.2f} dB, "
        f"delta={result['snr_delta_db']:+.2f} dB | "
        f"Separation firdecim={result['firdecim_separation_db']:.2f} dB, "
        f"polyphase={result['polyphase_separation_db']:.2f} dB, "
        f"delta={result['separation_delta_db']:+.2f} dB"
    )
    return result


def _assert_basic_result_quality(result):
    assert np.isfinite(result["firdecim_snr_db"])
    assert np.isfinite(result["polyphase_snr_db"])
    assert np.isfinite(result["firdecim_separation_db"])
    assert np.isfinite(result["polyphase_separation_db"])
    assert result["firdecim_runtime_mode"] == "firdecim"
    assert result["polyphase_runtime_mode"] == "polyphase"

    # Keep this as a sanity guardrail, not a strict winner-take-all metric.
    assert result["snr_delta_db"] > -1.5, (
        f"Polyphase SNR regressed too far vs firdecim ({result['snr_delta_db']:.2f} dB)"
    )
    assert result["separation_delta_db"] > -4.0, (
        "Polyphase stereo separation regressed too far vs firdecim "
        f"({result['separation_delta_db']:.2f} dB)"
    )


def test_pll_resampler_firdecim_vs_polyphase():
    settings = _load_pjfm_defaults()
    result = _run_bench(
        "PLLStereoDecoder",
        PLLStereoDecoder,
        PolyphasePLLStereoDecoder,
        settings,
    )
    _assert_basic_result_quality(result)


if __name__ == "__main__":
    defaults = _load_pjfm_defaults()
    print(f"Using pjfm defaults from {PJFM_CFG}:")
    print(
        "  iq_sample_rate={iq_sample_rate}, stereo_lpf_taps={stereo_lpf_taps}, "
        "stereo_lpf_beta={stereo_lpf_beta}, stereo_resampler_mode={stereo_resampler_mode}, "
        "stereo_resampler_taps={stereo_resampler_taps}, stereo_resampler_beta={stereo_resampler_beta}".format(
            **defaults
        )
    )

    pll_result = _run_bench(
        "PLLStereoDecoder",
        PLLStereoDecoder,
        PolyphasePLLStereoDecoder,
        defaults,
    )
    _assert_basic_result_quality(pll_result)

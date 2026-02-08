#!/usr/bin/env python3
"""
Small harness to validate L+R/L-R LPF tap-length impact on IHF stereo SNR.

This is a focused experiment script:
- It keeps decoder code unchanged.
- It monkey-patches LPF taps at runtime for A/B comparison.
- It uses the same core IHF-style method as existing tests:
  1. 1 kHz stereo tone (L=R), 19 kHz pilot, 75 kHz FM deviation
  2. Add RF AWGN at fixed SNR (default 120 dB)
  3. Decode full stereo chain
  4. Measure raw SNR and A-weighted IHF SNR via FFT tone-vs-residual

Default profile is "architecture-limited" (very high RF SNR + whole-buffer
decode) so LPF tap improvements are obvious. Use --rf-snr-db 40 to switch to a
receiver-like RF-limited scenario.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal

from demodulator import FMStereoDecoder

try:
    from pll_stereo_decoder import PLLStereoDecoder
except ImportError:
    PLLStereoDecoder = None


FM_DEVIATION_HZ = 75_000.0
DEFAULT_IQ_RATE = 480_000
DEFAULT_AUDIO_RATE = 48_000
DEFAULT_DURATION_S = 2.0
DEFAULT_WARMUP_S = 0.5
DEFAULT_TONE_HZ = 1_000.0
DEFAULT_TONE_AMP = 0.5
DEFAULT_RF_SNR_DB = 120.0
DEFAULT_BLOCK_SIZE = 0
DEFAULT_BETA = 8.0
DEFAULT_TAPS = "127,255,301,511"
DEFAULT_SEED = 42
DEFAULT_RESAMPLER = "both"
DEFAULT_RESAMPLER_TAPS = 127
DEFAULT_RESAMPLER_BETA = 8.0


@dataclass
class SweepRow:
    decoder: str
    requested_resampler: str
    active_resampler: str
    taps: int
    fir_19k_db: float
    raw_snr_db: float
    ihf_snr_db: float
    pilot_leak_dbc: float
    pilot_detected: bool


def parse_taps(value: str) -> list[int]:
    taps = []
    for piece in value.split(","):
        p = piece.strip()
        if not p:
            continue
        taps.append(int(p))
    if not taps:
        raise ValueError("No taps provided")
    for t in taps:
        if t < 3 or t % 2 == 0:
            raise ValueError(f"Invalid tap length {t}; use odd values >= 3")
    return taps


def generate_fm_stereo_multiplex(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: float,
    *,
    pilot_amplitude: float = 0.09,
    sum_gain: float = 0.9,
    diff_gain: float = 0.9,
) -> np.ndarray:
    n = len(left)
    t = np.arange(n, dtype=np.float64) / sample_rate
    lr_sum = (left + right) / 2.0
    lr_diff = (left - right) / 2.0
    pilot = pilot_amplitude * np.sin(2.0 * np.pi * 19_000.0 * t)
    carrier_38k = -np.cos(2.0 * np.pi * 38_000.0 * t)
    return sum_gain * lr_sum + pilot + diff_gain * (lr_diff * carrier_38k)


def fm_modulate(baseband: np.ndarray, sample_rate: float, deviation_hz: float = FM_DEVIATION_HZ) -> np.ndarray:
    phase = 2.0 * np.pi * deviation_hz * np.cumsum(baseband) / sample_rate
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


def add_awgn_at_snr(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(np.abs(iq) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    )
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def a_weighting_ba(fs: float) -> tuple[np.ndarray, np.ndarray]:
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12_194.217
    a1000_db = 1.9997

    nums = np.array([(2.0 * np.pi * f4) ** 2 * 10.0 ** (a1000_db / 20.0), 0.0, 0.0, 0.0, 0.0])
    dens = np.polymul([1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2], [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1.0, 2.0 * np.pi * f3]), [1.0, 2.0 * np.pi * f2])
    return signal.bilinear(nums, dens, fs)


def fft_tone_vs_residual_snr(x: np.ndarray, sample_rate: int, tone_hz: float) -> float:
    n = len(x)
    if n <= 0:
        return float("nan")
    window = np.hanning(n)
    x_w = x * window
    fft_out = np.fft.rfft(x_w)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power = np.abs(fft_out) ** 2 / n

    tone_mask = np.abs(freqs - tone_hz) <= 50.0
    tone_power = float(np.sum(power[tone_mask]))
    residual_power = float(np.sum(power[~tone_mask]))
    return 10.0 * np.log10((tone_power + 1e-30) / (residual_power + 1e-30))


def tone_rms_projection(x: np.ndarray, tone_hz: float, sample_rate: int) -> float:
    n = len(x)
    if n <= 0:
        return 0.0
    t = np.arange(n, dtype=np.float64) / sample_rate
    c = np.cos(2.0 * np.pi * tone_hz * t)
    s = np.sin(2.0 * np.pi * tone_hz * t)
    i = (2.0 / n) * float(np.dot(x, c))
    q = (2.0 / n) * float(np.dot(x, s))
    amp = math.hypot(i, q)
    return amp / math.sqrt(2.0)


def fir_response_at_19k_db(iq_rate: int, taps: int, beta: float) -> float:
    nyq = iq_rate / 2.0
    b = signal.firwin(taps, 15_000.0 / nyq, window=("kaiser", beta))
    w, h = signal.freqz(b, worN=65_536)
    f = w * iq_rate / (2.0 * np.pi)
    idx = int(np.argmin(np.abs(f - 19_000.0)))
    return 20.0 * np.log10(abs(h[idx]) + 1e-20)


def build_decoder(
    kind: str,
    iq_rate: int,
    audio_rate: int,
    *,
    resampler_mode: str,
    resampler_taps: int,
    resampler_beta: float,
) -> object:
    if kind == "pll":
        if PLLStereoDecoder is None:
            raise RuntimeError("PLLStereoDecoder unavailable")
        decoder = PLLStereoDecoder(
            iq_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            deviation=FM_DEVIATION_HZ,
            deemphasis=75e-6,
            resampler_mode=resampler_mode,
            resampler_taps=resampler_taps,
            resampler_beta=resampler_beta,
        )
    else:
        decoder = FMStereoDecoder(
            iq_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            deviation=FM_DEVIATION_HZ,
            deemphasis=75e-6,
            resampler_mode=resampler_mode,
            resampler_taps=resampler_taps,
            resampler_beta=resampler_beta,
        )

    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False
    if hasattr(decoder, "stereo_blend_low"):
        decoder.stereo_blend_low = -120.0
        decoder.stereo_blend_high = -110.0
    return decoder


def active_resampler_mode(decoder: object) -> str:
    mode = getattr(decoder, "_resampler_runtime_mode", None)
    if mode is not None:
        return str(mode)
    configured = getattr(decoder, "resampler_mode", None)
    if configured is not None:
        return str(configured)
    return "n/a"


def apply_runtime_lpf_taps(decoder: object, taps: int, beta: float) -> None:
    """
    Retune LPFs without editing decoder source files.

    Delay compensation does not need changes here because both L+R and L-R
    branches receive the same LPF length; the extra delay term remains set by
    the L-R BPF length.
    """
    nyq = decoder.iq_sample_rate / 2.0
    cutoff = 15_000.0 / nyq
    decoder.lr_sum_lpf = signal.firwin(taps, cutoff, window=("kaiser", beta))
    decoder.lr_sum_lpf_state = signal.lfilter_zi(decoder.lr_sum_lpf, 1.0)
    decoder.lr_diff_lpf = signal.firwin(taps, cutoff, window=("kaiser", beta))
    decoder.lr_diff_lpf_state = signal.lfilter_zi(decoder.lr_diff_lpf, 1.0)
    decoder.reset()


def disable_deemphasis(decoder: object) -> None:
    decoder.deem_b = np.array([1.0])
    decoder.deem_a = np.array([1.0, 0.0])
    decoder.deem_state_l = np.zeros(1)
    decoder.deem_state_r = np.zeros(1)


def decode_blocked(decoder: object, iq: np.ndarray, block_size: int) -> np.ndarray:
    outputs = []
    step = len(iq) if block_size <= 0 else block_size
    for i in range(0, len(iq), step):
        block = iq[i:i + step]
        if len(block) == 0:
            continue
        audio = decoder.demodulate(block)
        if audio is not None and len(audio) > 0:
            outputs.append(audio)
    if not outputs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(outputs)


def measure_one(
    *,
    decoder_kind: str,
    resampler_mode: str,
    resampler_taps: int,
    resampler_beta: float,
    taps: int,
    beta: float,
    iq: np.ndarray,
    iq_rate: int,
    audio_rate: int,
    warmup_s: float,
    tone_hz: float,
    block_size: int,
    a_weight_ba: tuple[np.ndarray, np.ndarray],
) -> SweepRow:
    dec_raw = build_decoder(
        decoder_kind,
        iq_rate,
        audio_rate,
        resampler_mode=resampler_mode,
        resampler_taps=resampler_taps,
        resampler_beta=resampler_beta,
    )
    apply_runtime_lpf_taps(dec_raw, taps, beta)
    disable_deemphasis(dec_raw)
    audio_raw = decode_blocked(dec_raw, iq, block_size)

    dec_ihf = build_decoder(
        decoder_kind,
        iq_rate,
        audio_rate,
        resampler_mode=resampler_mode,
        resampler_taps=resampler_taps,
        resampler_beta=resampler_beta,
    )
    apply_runtime_lpf_taps(dec_ihf, taps, beta)
    audio_ihf = decode_blocked(dec_ihf, iq, block_size)

    trim = int(warmup_s * audio_rate)
    left_raw = audio_raw[trim:, 0] if trim < len(audio_raw) else np.zeros(0, dtype=np.float32)
    left_ihf = audio_ihf[trim:, 0] if trim < len(audio_ihf) else np.zeros(0, dtype=np.float32)

    raw_snr_db = fft_tone_vs_residual_snr(left_raw, audio_rate, tone_hz)
    b_w, a_w = a_weight_ba
    left_ihf_w = signal.lfilter(b_w, a_w, left_ihf)
    ihf_snr_db = fft_tone_vs_residual_snr(left_ihf_w, audio_rate, tone_hz)

    tone_rms = tone_rms_projection(left_raw, tone_hz, audio_rate)
    pilot_rms = tone_rms_projection(left_raw, 19_000.0, audio_rate)
    pilot_leak_dbc = 20.0 * np.log10((pilot_rms + 1e-20) / (tone_rms + 1e-20))

    return SweepRow(
        decoder=decoder_kind,
        requested_resampler=resampler_mode,
        active_resampler=active_resampler_mode(dec_ihf),
        taps=taps,
        fir_19k_db=fir_response_at_19k_db(iq_rate, taps, beta),
        raw_snr_db=raw_snr_db,
        ihf_snr_db=ihf_snr_db,
        pilot_leak_dbc=pilot_leak_dbc,
        pilot_detected=bool(getattr(dec_ihf, "pilot_detected", False)),
    )


def generate_iq(args: argparse.Namespace) -> np.ndarray:
    n = int(args.duration * args.iq_rate)
    t = np.arange(n, dtype=np.float64) / args.iq_rate
    tone = args.tone_amp * np.sin(2.0 * np.pi * args.tone_hz * t)
    mpx = generate_fm_stereo_multiplex(tone, tone, args.iq_rate)
    iq_clean = fm_modulate(mpx, args.iq_rate, FM_DEVIATION_HZ)
    rng = np.random.default_rng(args.seed)
    return add_awgn_at_snr(iq_clean, args.rf_snr_db, rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep LPF taps and report IHF SNR improvement (defaults isolate LPF bottlenecks)."
    )
    parser.add_argument("--decoder", choices=["squaring", "pll", "both"], default="both")
    parser.add_argument(
        "--resampler",
        choices=["interp", "firdecim", "auto", "both"],
        default=DEFAULT_RESAMPLER,
        help="Resampler mode sweep; 'both' compares interp vs firdecim in one run.",
    )
    parser.add_argument(
        "--resampler-taps",
        type=int,
        default=DEFAULT_RESAMPLER_TAPS,
        help=f"FIR decimator taps for firdecim path (default: {DEFAULT_RESAMPLER_TAPS}).",
    )
    parser.add_argument(
        "--resampler-beta",
        type=float,
        default=DEFAULT_RESAMPLER_BETA,
        help=f"Kaiser beta for FIR decimator path (default: {DEFAULT_RESAMPLER_BETA:.1f}).",
    )
    parser.add_argument("--iq-rate", type=int, default=DEFAULT_IQ_RATE)
    parser.add_argument("--audio-rate", type=int, default=DEFAULT_AUDIO_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument("--tone-hz", type=float, default=DEFAULT_TONE_HZ)
    parser.add_argument("--tone-amp", type=float, default=DEFAULT_TONE_AMP)
    parser.add_argument(
        "--rf-snr-db",
        type=float,
        default=DEFAULT_RF_SNR_DB,
        help=f"Injected RF SNR in dB (default {DEFAULT_RF_SNR_DB:.0f}; try 40 for receiver-like conditions).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Decode block size; <=0 means whole-buffer decode (default).",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Kaiser beta for swept LPFs.")
    parser.add_argument("--taps", default=DEFAULT_TAPS, help=f"Comma-separated odd tap lengths (default: {DEFAULT_TAPS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_decoders(value: str) -> Iterable[str]:
    if value == "both":
        return ("squaring", "pll")
    return (value,)


def resolve_resamplers(value: str) -> Iterable[str]:
    if value == "both":
        return ("interp", "firdecim")
    return (value,)


def main() -> int:
    args = parse_args()
    taps_list = parse_taps(args.taps)
    decoders = list(resolve_decoders(args.decoder))
    resamplers = list(resolve_resamplers(args.resampler))
    if "pll" in decoders and PLLStereoDecoder is None:
        print("PLL decoder requested but pll_stereo_decoder.py is unavailable.")
        return 1
    if args.resampler_taps < 3 or (args.resampler_taps % 2) == 0:
        print("--resampler-taps must be an odd integer >= 3")
        return 1

    iq = generate_iq(args)
    a_w = a_weighting_ba(args.audio_rate)

    rows: list[SweepRow] = []
    for decoder_kind in decoders:
        for resampler_mode in resamplers:
            for taps in taps_list:
                rows.append(
                    measure_one(
                        decoder_kind=decoder_kind,
                        resampler_mode=resampler_mode,
                        resampler_taps=args.resampler_taps,
                        resampler_beta=args.resampler_beta,
                        taps=taps,
                        beta=args.beta,
                        iq=iq,
                        iq_rate=args.iq_rate,
                        audio_rate=args.audio_rate,
                        warmup_s=args.warmup,
                        tone_hz=args.tone_hz,
                        block_size=args.block_size,
                        a_weight_ba=a_w,
                    )
                )

    print("\nIHF LPF Sweep Harness")
    print("=" * 120)
    print(f"iq_rate={args.iq_rate} Hz, audio_rate={args.audio_rate} Hz, rf_snr={args.rf_snr_db:.1f} dB")
    print(f"duration={args.duration:.2f}s, warmup={args.warmup:.2f}s, tone={args.tone_hz:.1f} Hz, beta={args.beta:.1f}")
    print(f"resampler(s)={', '.join(resamplers)}, resampler_taps={args.resampler_taps}, resampler_beta={args.resampler_beta:.1f}")
    print()
    print(
        f"{'Decoder':9s} {'ReqRsmp':8s} {'ActRsmp':8s} {'Taps':>5s} {'FIR@19k':>10s} {'Raw SNR':>10s} "
        f"{'IHF SNR':>10s} {'PilotLeak':>10s} {'Pilot':>7s}"
    )
    print("-" * 120)

    for decoder_kind in decoders:
        for resampler_mode in resamplers:
            subset = [r for r in rows if r.decoder == decoder_kind and r.requested_resampler == resampler_mode]
            base = next((r for r in subset if r.taps == taps_list[0]), None)
            for r in subset:
                print(
                    f"{r.decoder:9s} {r.requested_resampler:8s} {r.active_resampler:8s} "
                    f"{r.taps:5d} {r.fir_19k_db:9.1f}dB {r.raw_snr_db:9.1f}dB "
                    f"{r.ihf_snr_db:9.1f}dB {r.pilot_leak_dbc:9.1f}dBc {'yes' if r.pilot_detected else 'no':>7s}"
                )
            if base is not None:
                best = max(subset, key=lambda x: x.ihf_snr_db)
                print(
                    f"  {decoder_kind}/{resampler_mode}: best IHF improvement vs {base.taps} taps = "
                    f"{best.ihf_snr_db - base.ihf_snr_db:+.1f} dB (best at {best.taps} taps)"
                )
            print()

    comparisons = {}
    for r in rows:
        key = (r.decoder, r.taps)
        if key not in comparisons:
            comparisons[key] = {}
        comparisons[key][r.requested_resampler] = r
    ab_rows = []
    for key, by_mode in comparisons.items():
        interp_row = by_mode.get("interp")
        fir_row = by_mode.get("firdecim")
        if interp_row is None or fir_row is None:
            continue
        ab_rows.append(
            (
                key[0],
                key[1],
                fir_row.raw_snr_db - interp_row.raw_snr_db,
                fir_row.ihf_snr_db - interp_row.ihf_snr_db,
                interp_row.ihf_snr_db,
                fir_row.ihf_snr_db,
            )
        )

    if ab_rows:
        print("Resampler A/B (firdecim - interp)")
        print("-" * 120)
        print(
            f"{'Decoder':9s} {'Taps':>5s} {'DeltaRaw':>10s} {'DeltaIHF':>10s} "
            f"{'InterpIHF':>10s} {'FIRDecimIHF':>12s}"
        )
        print("-" * 120)
        for decoder_kind, taps, delta_raw, delta_ihf, interp_ihf, fir_ihf in ab_rows:
            print(
                f"{decoder_kind:9s} {taps:5d} {delta_raw:9.1f}dB {delta_ihf:9.1f}dB "
                f"{interp_ihf:9.1f}dB {fir_ihf:11.1f}dB"
            )
        print()

    print("Notes")
    print("-" * 120)
    print("Use --rf-snr-db 40 to see RF-limited behavior where LPF gains are smaller.")
    print("Use --block-size 8192 to emulate streaming/block-boundary effects from existing tests.")
    print("Use --resampler both for before/after np.interp vs stateful FIR decimator deltas.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

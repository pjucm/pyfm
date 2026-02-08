#!/usr/bin/env python3
"""
IHF/EIA-style FM SNR bench for panadapter.py and pyfm/pjfm decode paths.

This is a standalone synthetic test harness. It is "IHF/EIA-like" rather than
a strict certified bench:
- 1 kHz modulation reference
- de-emphasis in decoder (unless disabled)
- optional post-detection A-weighting
- 30 Hz high-pass + 15 kHz low-pass measurement bandwidth
- stereo mode includes pilot and optional 19 kHz notch in post filter

Supported decode paths:
- panadapter.py: WBFMStereoDemodulator (includes decimation wrapper)
- pyfm/pjfm path: direct decoder class path (PLLStereoDecoder or FMStereoDecoder)

Examples:
    python3 test-ihf-eia.py
    python3 test-ihf-eia.py --backend panadapter --decoder pll
    python3 test-ihf-eia.py --backend pyfm --decoder both --weighting none
    python3 test-ihf-eia.py --backend pyfm --decoder both --resampler both --block-size 8192
    python3 test-ihf-eia.py --rf-snr-db 50 --duration 1.5
    python3 test-ihf-eia.py --rf-snr-db none  # disable RF AWGN
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from demodulator import FMStereoDecoder

try:
    from pll_stereo_decoder import PLLStereoDecoder
except ImportError:
    PLLStereoDecoder = None

try:
    import panadapter
except Exception:
    panadapter = None


FM_DEVIATION_HZ = 75_000.0
DEFAULT_AUDIO_RATE = 48_000
DEFAULT_PANADAPTER_IQ_RATE = 960_000
DEFAULT_PYFM_IQ_RATE = 480_000
DEFAULT_DURATION_S = 1.0
DEFAULT_WARMUP_S = 0.2
DEFAULT_TONE_AMPLITUDE = 0.5
DEFAULT_RF_SNR_DB = 40.0
DEFAULT_WEIGHTING = "a"
DEFAULT_BLOCK_SIZE = 0
DEFAULT_RNG_SEED = 42
DEFAULT_RESAMPLER = "both"
DEFAULT_RESAMPLER_TAPS = 127
DEFAULT_RESAMPLER_BETA = 8.0


@dataclass
class DecodePath:
    backend: str
    requested_decoder: str
    active_decoder: str
    iq_rate: int
    audio_rate: int
    decoder_obj: object
    wrapper_kind: str  # "panadapter" or "direct"


@dataclass
class BenchResult:
    backend: str
    requested_decoder: str
    active_decoder: str
    requested_resampler: str
    active_resampler: str
    mode: str
    weighting: str
    signal_rms: float
    noise_rms: float
    snr_db: float
    pilot_detected_signal: Optional[bool]
    pilot_detected_noise: Optional[bool]
    audio_samples_signal: int
    audio_samples_noise: int


def generate_fm_stereo_multiplex(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: float,
    *,
    pilot_amplitude: float = 0.09,
    include_pilot: bool = True,
    subcarrier_phase: str = "neg_cos",
    sum_gain: float = 1.0,
    diff_gain: float = 1.0,
) -> np.ndarray:
    """Generate FM stereo multiplex baseband."""
    n = len(left)
    t = np.arange(n, dtype=np.float64) / sample_rate

    lr_sum = (left + right) / 2.0
    lr_diff = (left - right) / 2.0

    if include_pilot:
        pilot = pilot_amplitude * np.sin(2.0 * np.pi * 19_000.0 * t)
    else:
        pilot = np.zeros(n, dtype=np.float64)

    if subcarrier_phase == "neg_cos":
        carrier_38k = -np.cos(2.0 * np.pi * 38_000.0 * t)
    elif subcarrier_phase == "cos":
        carrier_38k = np.cos(2.0 * np.pi * 38_000.0 * t)
    elif subcarrier_phase == "sin":
        carrier_38k = np.sin(2.0 * np.pi * 38_000.0 * t)
    else:
        raise ValueError(f"Unknown subcarrier_phase: {subcarrier_phase}")

    return sum_gain * lr_sum + pilot + diff_gain * (lr_diff * carrier_38k)


def fm_modulate(baseband: np.ndarray, sample_rate: float, deviation_hz: float = FM_DEVIATION_HZ) -> np.ndarray:
    """FM-modulate baseband to complex IQ."""
    phase = 2.0 * np.pi * deviation_hz * np.cumsum(baseband) / sample_rate
    iq = np.cos(phase) + 1j * np.sin(phase)
    return iq.astype(np.complex64)


def add_awgn_at_snr(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN at the requested SNR (relative to signal power)."""
    sig_power = np.mean(np.abs(iq) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    )
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def a_weighting_ba(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Digital A-weighting filter (bilinear transform from IEC analog prototype).

    Reference gain is normalized at 1 kHz.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12_194.217
    a1000_db = 1.9997

    nums = np.array([(2.0 * np.pi * f4) ** 2 * 10.0 ** (a1000_db / 20.0), 0.0, 0.0, 0.0, 0.0])
    dens = np.polymul([1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2], [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1.0, 2.0 * np.pi * f3]), [1.0, 2.0 * np.pi * f2])
    return signal.bilinear(nums, dens, fs)


def apply_measurement_filters(x: np.ndarray, sample_rate: int, *, stereo_mode: bool, weighting: str) -> np.ndarray:
    """Apply IHF/EIA-like post-detection measurement filtering."""
    if len(x) == 0:
        return x

    hp_sos = signal.butter(2, 30.0 / (sample_rate / 2.0), btype="high", output="sos")
    lp_sos = signal.butter(6, 15_000.0 / (sample_rate / 2.0), btype="low", output="sos")

    y = signal.sosfilt(hp_sos, x)
    y = signal.sosfilt(lp_sos, y)

    if stereo_mode:
        # Typical stereo SNR benches reject pilot bleed.
        b_notch, a_notch = signal.iirnotch(19_000.0, 30.0, sample_rate)
        y = signal.lfilter(b_notch, a_notch, y)

    if weighting == "a":
        b_w, a_w = a_weighting_ba(sample_rate)
        y = signal.lfilter(b_w, a_w, y)

    return y


def tone_rms(x: np.ndarray, tone_hz: float, sample_rate: int) -> float:
    """Estimate RMS amplitude of a single tone via quadrature projection."""
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


def _disable_deemphasis(decoder_obj: object) -> None:
    """Disable de-emphasis by replacing filter with unity gain path."""
    decoder_obj.deem_b = np.array([1.0])
    decoder_obj.deem_a = np.array([1.0, 0.0])
    decoder_obj.deem_state_l = np.zeros(1)
    decoder_obj.deem_state_r = np.zeros(1)


def _set_resampler(decoder_obj: object, mode: str, taps: int, beta: float) -> None:
    """Apply runtime resampler settings on a decoder instance."""
    if not hasattr(decoder_obj, "resampler_mode"):
        raise RuntimeError("Decoder does not expose resampler controls")
    decoder_obj.resampler_mode = mode
    decoder_obj.resampler_taps = int(taps)
    decoder_obj.resampler_beta = float(beta)
    if hasattr(decoder_obj, "_configure_resampler"):
        decoder_obj._configure_resampler()
    if hasattr(decoder_obj, "reset"):
        decoder_obj.reset()


def _get_stereo_decoder_obj(path: DecodePath) -> object:
    if path.wrapper_kind == "panadapter":
        return path.decoder_obj.stereo_decoder
    return path.decoder_obj


def build_decode_path(
    backend: str,
    decoder_name: str,
    iq_rate: int,
    audio_rate: int,
    *,
    resampler_mode: str,
    resampler_taps: int,
    resampler_beta: float,
    disable_deemphasis: bool,
) -> DecodePath:
    """Build one decode path instance."""
    if backend == "panadapter":
        if panadapter is None:
            raise RuntimeError("panadapter.py dependencies unavailable in this environment")
        demod = panadapter.WBFMStereoDemodulator(
            input_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            stereo_decoder=decoder_name,
        )
        demod.stereo_decoder.bass_boost_enabled = False
        demod.stereo_decoder.treble_boost_enabled = False
        demod.stereo_decoder.stereo_blend_enabled = False
        _set_resampler(demod.stereo_decoder, resampler_mode, resampler_taps, resampler_beta)
        if disable_deemphasis:
            _disable_deemphasis(demod.stereo_decoder)
        return DecodePath(
            backend=backend,
            requested_decoder=decoder_name,
            active_decoder=demod.stereo_decoder_name,
            iq_rate=iq_rate,
            audio_rate=audio_rate,
            decoder_obj=demod,
            wrapper_kind="panadapter",
        )

    if decoder_name == "pll":
        if PLLStereoDecoder is None:
            raise RuntimeError("PLLStereoDecoder unavailable")
        decoder_cls = PLLStereoDecoder
    else:
        decoder_cls = FMStereoDecoder

    dec = decoder_cls(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=FM_DEVIATION_HZ,
        deemphasis=75e-6,
        resampler_mode=resampler_mode,
        resampler_taps=resampler_taps,
        resampler_beta=resampler_beta,
    )
    dec.bass_boost_enabled = False
    dec.treble_boost_enabled = False
    dec.stereo_blend_enabled = False
    _set_resampler(dec, resampler_mode, resampler_taps, resampler_beta)
    if disable_deemphasis:
        _disable_deemphasis(dec)

    return DecodePath(
        backend=backend,
        requested_decoder=decoder_name,
        active_decoder=decoder_name,
        iq_rate=iq_rate,
        audio_rate=audio_rate,
        decoder_obj=dec,
        wrapper_kind="direct",
    )


def decode_iq_stream(path: DecodePath, iq: np.ndarray, block_size: int) -> np.ndarray:
    """Decode IQ samples through either wrapper or direct decoder path."""
    out = []
    step = len(iq) if block_size <= 0 else block_size

    for i in range(0, len(iq), step):
        block = iq[i:i + step]
        if len(block) == 0:
            continue
        if path.wrapper_kind == "panadapter":
            audio = path.decoder_obj.process(block)
            if audio is None:
                continue
        else:
            audio = path.decoder_obj.demodulate(block)
        out.append(audio)

    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(out)


def pilot_state(path: DecodePath) -> Optional[bool]:
    """Return pilot lock/detect state if available."""
    try:
        if path.wrapper_kind == "panadapter":
            return bool(path.decoder_obj.pilot_detected)
        return bool(path.decoder_obj.pilot_detected)
    except Exception:
        return None


def resampler_state(path: DecodePath) -> str:
    """Return active runtime resampler mode if available."""
    try:
        decoder_obj = _get_stereo_decoder_obj(path)
        runtime_mode = getattr(decoder_obj, "_resampler_runtime_mode", None)
        if runtime_mode is not None:
            return str(runtime_mode)
        configured_mode = getattr(decoder_obj, "resampler_mode", None)
        if configured_mode is not None:
            return str(configured_mode)
    except Exception:
        pass
    return "n/a"


def run_mode(
    *,
    backend: str,
    decoder_name: str,
    resampler_mode: str,
    resampler_taps: int,
    resampler_beta: float,
    iq_rate: int,
    audio_rate: int,
    duration_s: float,
    warmup_s: float,
    tone_hz: float,
    tone_amp: float,
    rf_snr_db: Optional[float],
    weighting: str,
    block_size: int,
    disable_deemphasis: bool,
    rng_seed: int,
    mode: str,
) -> BenchResult:
    """Run one mono or stereo IHF/EIA-style SNR measurement."""
    n = int(duration_s * iq_rate)
    t = np.arange(n, dtype=np.float64) / iq_rate
    tone = tone_amp * np.sin(2.0 * np.pi * tone_hz * t)

    if mode == "mono":
        left_sig = tone
        right_sig = tone
        sig_mpx = generate_fm_stereo_multiplex(
            left_sig, right_sig, iq_rate, include_pilot=False, sum_gain=1.0, diff_gain=1.0
        )
        noise_mpx = np.zeros_like(sig_mpx)
        stereo_mode = False
        select_channel = lambda a: 0.5 * (a[:, 0] + a[:, 1])
    elif mode == "stereo":
        left_sig = tone
        right_sig = np.zeros_like(tone)
        sig_mpx = generate_fm_stereo_multiplex(
            left_sig, right_sig, iq_rate, include_pilot=True, sum_gain=1.0, diff_gain=1.0
        )
        noise_mpx = generate_fm_stereo_multiplex(
            np.zeros_like(tone), np.zeros_like(tone), iq_rate, include_pilot=True, sum_gain=1.0, diff_gain=1.0
        )
        stereo_mode = True
        select_channel = lambda a: a[:, 0]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    iq_signal = fm_modulate(sig_mpx, iq_rate, FM_DEVIATION_HZ)
    iq_noise_ref = fm_modulate(noise_mpx, iq_rate, FM_DEVIATION_HZ)

    if rf_snr_db is not None:
        rng = np.random.default_rng(rng_seed)
        iq_signal = add_awgn_at_snr(iq_signal, rf_snr_db, rng)
        iq_noise_ref = add_awgn_at_snr(iq_noise_ref, rf_snr_db, rng)

    # Signal run
    signal_path = build_decode_path(
        backend,
        decoder_name,
        iq_rate,
        audio_rate,
        resampler_mode=resampler_mode,
        resampler_taps=resampler_taps,
        resampler_beta=resampler_beta,
        disable_deemphasis=disable_deemphasis,
    )
    audio_signal = decode_iq_stream(signal_path, iq_signal, block_size)
    pilot_sig = pilot_state(signal_path)
    active_resampler = resampler_state(signal_path)

    # Noise/reference run
    noise_path = build_decode_path(
        backend,
        decoder_name,
        iq_rate,
        audio_rate,
        resampler_mode=resampler_mode,
        resampler_taps=resampler_taps,
        resampler_beta=resampler_beta,
        disable_deemphasis=disable_deemphasis,
    )
    audio_noise = decode_iq_stream(noise_path, iq_noise_ref, block_size)
    pilot_noise = pilot_state(noise_path)

    warmup_samples = int(warmup_s * audio_rate)
    xs = select_channel(audio_signal)
    xn = select_channel(audio_noise)
    if warmup_samples > 0:
        xs = xs[warmup_samples:]
        xn = xn[warmup_samples:]

    xs = apply_measurement_filters(xs, audio_rate, stereo_mode=stereo_mode, weighting=weighting)
    xn = apply_measurement_filters(xn, audio_rate, stereo_mode=stereo_mode, weighting=weighting)

    sig_rms = tone_rms(xs, tone_hz, audio_rate)
    noise_rms = float(np.sqrt(np.mean(xn * xn))) if len(xn) > 0 else 0.0
    snr_db = 20.0 * np.log10((sig_rms + 1e-20) / (noise_rms + 1e-20))

    return BenchResult(
        backend=backend,
        requested_decoder=decoder_name,
        active_decoder=signal_path.active_decoder,
        requested_resampler=resampler_mode,
        active_resampler=active_resampler,
        mode=mode,
        weighting=weighting,
        signal_rms=sig_rms,
        noise_rms=noise_rms,
        snr_db=snr_db,
        pilot_detected_signal=pilot_sig,
        pilot_detected_noise=pilot_noise,
        audio_samples_signal=len(xs),
        audio_samples_noise=len(xn),
    )


def parse_optional_float(value: str) -> Optional[float]:
    """Parse float or keywords that disable the setting."""
    lowered = value.strip().lower()
    if lowered in {"none", "off", "disable", "disabled"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone IHF/EIA-style FM SNR benchmark (datasheet-comparison defaults)."
    )
    parser.add_argument("--backend", choices=["panadapter", "pyfm", "pjfm", "both"], default="both")
    parser.add_argument("--decoder", choices=["pll", "squaring", "both"], default="both")
    parser.add_argument(
        "--resampler",
        choices=["interp", "firdecim", "auto", "both"],
        default=DEFAULT_RESAMPLER,
        help=(
            "Resampler mode under test. "
            "'both' runs interp (np.interp path) and firdecim (stateful FIR decimator)."
        ),
    )
    parser.add_argument(
        "--resampler-taps",
        type=int,
        default=DEFAULT_RESAMPLER_TAPS,
        help=f"FIR decimator tap count for firdecim path (default: {DEFAULT_RESAMPLER_TAPS}).",
    )
    parser.add_argument(
        "--resampler-beta",
        type=float,
        default=DEFAULT_RESAMPLER_BETA,
        help=f"Kaiser beta for FIR decimator (default: {DEFAULT_RESAMPLER_BETA:.1f}).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Signal duration in seconds (default: {DEFAULT_DURATION_S:.1f}).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP_S,
        help=f"Warmup trim in seconds (default: {DEFAULT_WARMUP_S:.1f}).",
    )
    parser.add_argument("--tone-hz", type=float, default=1000.0, help="Reference tone frequency.")
    parser.add_argument(
        "--tone-amp",
        type=float,
        default=DEFAULT_TONE_AMPLITUDE,
        help=f"Reference tone amplitude in multiplex synthesis (default: {DEFAULT_TONE_AMPLITUDE:.2f}).",
    )
    parser.add_argument(
        "--rf-snr-db",
        type=parse_optional_float,
        default=DEFAULT_RF_SNR_DB,
        help=(
            f"RF AWGN SNR in dB (default: {DEFAULT_RF_SNR_DB:.1f}); "
            "use 'none' to disable RF noise injection."
        ),
    )
    parser.add_argument(
        "--weighting",
        choices=["none", "a"],
        default=DEFAULT_WEIGHTING,
        help=f"Post-detection weighting (default: {DEFAULT_WEIGHTING}).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Decode block size; <=0 means whole buffer (default: whole buffer).",
    )
    parser.add_argument("--audio-rate", type=int, default=DEFAULT_AUDIO_RATE)
    parser.add_argument("--panadapter-iq-rate", type=int, default=DEFAULT_PANADAPTER_IQ_RATE)
    parser.add_argument("--pyfm-iq-rate", type=int, default=DEFAULT_PYFM_IQ_RATE)
    parser.add_argument("--disable-deemphasis", action="store_true", help="Disable decoder de-emphasis.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f"RNG seed for optional AWGN (default: {DEFAULT_RNG_SEED}).",
    )
    return parser.parse_args()


def resolve_backends(value: str) -> list[str]:
    if value == "both":
        return ["panadapter", "pyfm"]
    if value == "pjfm":
        return ["pyfm"]
    return [value]


def resolve_decoders(value: str) -> list[str]:
    if value == "both":
        return ["pll", "squaring"]
    return [value]


def resolve_resamplers(value: str) -> list[str]:
    if value == "both":
        return ["interp", "firdecim"]
    return [value]


def format_bool(v: Optional[bool]) -> str:
    if v is None:
        return "n/a"
    return "yes" if v else "no"


def main() -> int:
    args = parse_args()
    backends = resolve_backends(args.backend)
    decoders = resolve_decoders(args.decoder)
    resamplers = resolve_resamplers(args.resampler)
    table_width = 120

    if args.resampler_taps < 3 or (args.resampler_taps % 2) == 0:
        print("--resampler-taps must be an odd integer >= 3")
        return 1

    print("\nIHF/EIA-Style FM SNR Bench")
    print("=" * table_width)
    print("preset defaults: datasheet-comparison")
    print(f"backend(s): {', '.join(backends)}")
    print(f"decoder(s): {', '.join(decoders)}")
    print(f"resampler(s): {', '.join(resamplers)}")
    print(f"duration: {args.duration:.2f}s, warmup: {args.warmup:.2f}s, tone: {args.tone_hz:.1f} Hz")
    print(f"weighting: {args.weighting}, de-emphasis: {'off' if args.disable_deemphasis else 'on'}")
    print(f"rf_snr_db: {'none' if args.rf_snr_db is None else f'{args.rf_snr_db:.1f} dB'}")
    print(f"resampler_taps: {args.resampler_taps}, resampler_beta: {args.resampler_beta:.2f}")
    print()

    results: list[BenchResult] = []
    failures: list[tuple[str, str, str, str, str]] = []

    for backend in backends:
        iq_rate = args.panadapter_iq_rate if backend == "panadapter" else args.pyfm_iq_rate
        for decoder_name in decoders:
            if decoder_name == "pll" and PLLStereoDecoder is None:
                failures.append((backend, decoder_name, "all", "all", "PLLStereoDecoder unavailable"))
                continue

            for resampler_mode in resamplers:
                for mode in ("mono", "stereo"):
                    try:
                        result = run_mode(
                            backend=backend,
                            decoder_name=decoder_name,
                            resampler_mode=resampler_mode,
                            resampler_taps=args.resampler_taps,
                            resampler_beta=args.resampler_beta,
                            iq_rate=iq_rate,
                            audio_rate=args.audio_rate,
                            duration_s=args.duration,
                            warmup_s=args.warmup,
                            tone_hz=args.tone_hz,
                            tone_amp=args.tone_amp,
                            rf_snr_db=args.rf_snr_db,
                            weighting=args.weighting,
                            block_size=args.block_size,
                            disable_deemphasis=args.disable_deemphasis,
                            rng_seed=args.seed + (1000 if mode == "stereo" else 0),
                            mode=mode,
                        )
                        results.append(result)
                    except Exception as exc:
                        failures.append((backend, decoder_name, resampler_mode, mode, str(exc)))

    if results:
        print("Results")
        print("-" * table_width)
        print(
            f"{'Backend':10s} {'ReqDec':8s} {'ActDec':8s} {'ReqRsmp':8s} {'ActRsmp':8s} {'Mode':7s} "
            f"{'SNR(dB)':>9s} {'SignalRMS':>11s} {'NoiseRMS':>11s} {'Pilot(sig/noise)':>17s}"
        )
        print("-" * table_width)
        for r in results:
            pilot_pair = f"{format_bool(r.pilot_detected_signal)}/{format_bool(r.pilot_detected_noise)}"
            print(
                f"{r.backend:10s} {r.requested_decoder:8s} {r.active_decoder:8s} "
                f"{r.requested_resampler:8s} {r.active_resampler:8s} {r.mode:7s} "
                f"{r.snr_db:9.2f} {r.signal_rms:11.6f} {r.noise_rms:11.6f} {pilot_pair:>17s}"
            )

        comparisons = {}
        for r in results:
            key = (r.backend, r.requested_decoder, r.mode)
            if key not in comparisons:
                comparisons[key] = {}
            comparisons[key][r.requested_resampler] = r

        ab_rows = []
        for key, modes in comparisons.items():
            interp_row = modes.get("interp")
            fir_row = modes.get("firdecim")
            if interp_row is None or fir_row is None:
                continue
            delta = fir_row.snr_db - interp_row.snr_db
            ab_rows.append((key, delta, interp_row.snr_db, fir_row.snr_db))

        if ab_rows:
            print("\nResampler A/B (firdecim - interp)")
            print("-" * table_width)
            print(
                f"{'Backend':10s} {'Decoder':8s} {'Mode':7s} "
                f"{'Delta(dB)':>10s} {'Interp':>10s} {'FIRDecim':>10s}"
            )
            print("-" * table_width)
            for (backend, decoder_name, mode), delta, interp_snr, fir_snr in ab_rows:
                print(
                    f"{backend:10s} {decoder_name:8s} {mode:7s} "
                    f"{delta:10.2f} {interp_snr:10.2f} {fir_snr:10.2f}"
                )

    unlocked = [r for r in results if r.mode == "stereo" and r.pilot_detected_signal is False]
    if unlocked:
        print("\nWarnings")
        print("-" * table_width)
        print("Stereo pilot was not detected for some runs; treat those stereo SNR values as non-datasheet.")
        for r in unlocked:
            print(f"{r.backend}/{r.requested_decoder}: pilot_detected=no")

    if failures:
        print("\nFailures")
        print("-" * table_width)
        for backend, decoder_name, resampler_mode, mode, err in failures:
            print(f"{backend}/{decoder_name}/{resampler_mode}/{mode}: {err}")

    print("\nNotes")
    print("-" * table_width)
    print("This is an IHF/EIA-like synthetic DSP bench, not a certified analog RF tuner bench.")
    print("Use this primarily for relative comparisons across decode paths and decoder types.")
    print("Use --resampler both for before/after np.interp vs stateful FIR decimator deltas.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

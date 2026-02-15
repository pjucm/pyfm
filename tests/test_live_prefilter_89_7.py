#!/usr/bin/env python3
"""
Live 89.7 MHz pre-filter A/B test.

Captures live I/Q once, then compares:
  1) Baseline (no I/Q pre-filter)
  2) I/Q low-pass pre-filter at +/- cutoff (default +/-120 kHz)

Metrics reported:
  - RF selectivity proxy: in-channel vs adjacent-channel power ratio
  - Decoder SNR proxy: median/mean PLLStereoDecoder reported SNR
  - Pilot SNR proxy: 19 kHz pilot power vs nearby noise bands

Usage examples:
  python tests/test_live_prefilter_89_7.py
  python tests/test_live_prefilter_89_7.py --capture-seconds 30 --cutoff-khz 120
  python tests/test_live_prefilter_89_7.py --save-iq /tmp/iq_89_7.npz
  python tests/test_live_prefilter_89_7.py --iq-file /tmp/iq_89_7.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import icom_r8600 as r8600
from pll_stereo_decoder import PLLStereoDecoder


FM_DEVIATION_HZ = 75_000.0
EPS = 1e-20


@dataclass
class ConditionMetrics:
    name: str
    rf_main_db: float
    rf_adjacent_db: float
    rf_selectivity_db: float
    decoder_snr_median_db: float
    decoder_snr_mean_db: float
    pilot_proxy_snr_db: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs +/-120 kHz I/Q pre-filter at live 89.7 MHz."
    )
    parser.add_argument("--freq-mhz", type=float, default=89.7, help="Center frequency in MHz (default: 89.7)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=480000,
        help="Requested I/Q sample rate in Hz (default: 480000)",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=20.0,
        help="Capture duration in seconds when using hardware (default: 20)",
    )
    parser.add_argument(
        "--block-samples",
        type=int,
        default=8192,
        help="Fetch/decode block size in samples (default: 8192)",
    )
    parser.add_argument(
        "--cutoff-khz",
        type=float,
        default=120.0,
        help="Pre-filter cutoff in kHz for +/- cutoff LPF (default: 120)",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=255,
        help="Odd FIR tap count for I/Q pre-filter (default: 255)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.7,
        help="Warmup after configure before capture (default: 0.7)",
    )
    parser.add_argument(
        "--use-24bit",
        action="store_true",
        help="Request 24-bit streaming mode (if available at chosen sample rate)",
    )
    parser.add_argument(
        "--iq-file",
        type=str,
        default="",
        help="Load I/Q from .npz file instead of live capture (expects keys: iq, sample_rate, freq_hz)",
    )
    parser.add_argument(
        "--save-iq",
        type=str,
        default="",
        help="Save captured I/Q to .npz file for replay",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON metrics report",
    )
    return parser.parse_args()


def _band_power_two_sided(freqs: np.ndarray, psd: np.ndarray, lo_hz: float, hi_hz: float) -> Tuple[float, float]:
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not np.any(mask):
        return 0.0, 0.0
    f_sel = freqs[mask]
    p_sel = psd[mask]
    if f_sel.size < 2:
        return 0.0, 0.0
    bw_hz = float(f_sel[-1] - f_sel[0])
    df_hz = float(np.mean(np.diff(f_sel)))
    power = float(np.sum(p_sel) * df_hz)
    return power, max(bw_hz, 0.0)


def _band_power_one_sided(freqs: np.ndarray, psd: np.ndarray, lo_hz: float, hi_hz: float) -> Tuple[float, float]:
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not np.any(mask):
        return 0.0, 0.0
    f_sel = freqs[mask]
    p_sel = psd[mask]
    if f_sel.size < 2:
        return 0.0, 0.0
    bw_hz = float(f_sel[-1] - f_sel[0])
    df_hz = float(np.mean(np.diff(f_sel)))
    power = float(np.sum(p_sel) * df_hz)
    return power, max(bw_hz, 0.0)


def _welch_two_sided(iq: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(65536, len(iq))
    if nperseg < 2048:
        raise ValueError("Not enough samples for RF PSD estimate")
    freqs, psd = signal.welch(
        iq.astype(np.complex128),
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freqs)
    return freqs[order], psd[order]


def _welch_one_sided(x: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(65536, len(x))
    if nperseg < 2048:
        raise ValueError("Not enough samples for baseband PSD estimate")
    freqs, psd = signal.welch(
        x.astype(np.float64),
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    return freqs, psd


def fm_demod_baseband(iq: np.ndarray, sample_rate: int) -> np.ndarray:
    product = iq[1:] * np.conj(iq[:-1])
    phase_diff = np.angle(product)
    baseband = phase_diff * (sample_rate / (2.0 * np.pi * FM_DEVIATION_HZ))
    return baseband.astype(np.float64, copy=False)


def design_prefilter(sample_rate: int, cutoff_hz: float, taps: int) -> np.ndarray:
    if taps < 3 or (taps % 2) == 0:
        raise ValueError(f"--taps must be an odd integer >= 3, got {taps}")
    nyq = 0.5 * float(sample_rate)
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Invalid cutoff {cutoff_hz} Hz for sample rate {sample_rate} Hz")
    norm = cutoff_hz / nyq
    return signal.firwin(taps, norm, window=("kaiser", 8.0)).astype(np.float64)


def apply_prefilter(iq: np.ndarray, h: np.ndarray) -> np.ndarray:
    i_f = signal.lfilter(h, [1.0], iq.real.astype(np.float64, copy=False))
    q_f = signal.lfilter(h, [1.0], iq.imag.astype(np.float64, copy=False))
    return (i_f + 1j * q_f).astype(np.complex64)


def decode_snr_series(iq: np.ndarray, sample_rate: int, block_samples: int) -> np.ndarray:
    decoder = PLLStereoDecoder(iq_sample_rate=sample_rate, audio_sample_rate=48_000)
    snr_values = []
    for i in range(0, len(iq), block_samples):
        block = iq[i:i + block_samples]
        if len(block) < 8:
            break
        decoder.demodulate(block)
        snr_values.append(float(decoder.snr_db))
    return np.array(snr_values, dtype=np.float64)


def evaluate_condition(name: str, iq: np.ndarray, sample_rate: int, block_samples: int) -> ConditionMetrics:
    rf_freqs, rf_psd = _welch_two_sided(iq, sample_rate)
    nyq = 0.5 * sample_rate
    adj_lo = min(120_000.0, nyq * 0.75)
    adj_hi = min(220_000.0, nyq * 0.98)
    if adj_hi <= adj_lo + 5_000:
        raise ValueError(
            f"Sample rate {sample_rate} too low for adjacent-band selectivity metric; "
            "use at least 480000 Hz."
        )

    main_p, _ = _band_power_two_sided(rf_freqs, rf_psd, -100_000.0, 100_000.0)
    adj_l_p, _ = _band_power_two_sided(rf_freqs, rf_psd, -adj_hi, -adj_lo)
    adj_u_p, _ = _band_power_two_sided(rf_freqs, rf_psd, adj_lo, adj_hi)
    adj_p = 0.5 * (adj_l_p + adj_u_p)

    rf_main_db = 10.0 * np.log10(main_p + EPS)
    rf_adjacent_db = 10.0 * np.log10(adj_p + EPS)
    rf_selectivity_db = 10.0 * np.log10((main_p + EPS) / (adj_p + EPS))

    snr_series = decode_snr_series(iq, sample_rate, block_samples)
    if snr_series.size == 0:
        decoder_snr_median = float("nan")
        decoder_snr_mean = float("nan")
    else:
        settle = max(3, int(0.1 * snr_series.size))
        steady = snr_series[settle:] if snr_series.size > settle else snr_series
        decoder_snr_median = float(np.median(steady))
        decoder_snr_mean = float(np.mean(steady))

    baseband = fm_demod_baseband(iq, sample_rate)
    bb_freqs, bb_psd = _welch_one_sided(baseband, sample_rate)
    pilot_p, pilot_bw = _band_power_one_sided(bb_freqs, bb_psd, 18_500.0, 19_500.0)
    n1_p, n1_bw = _band_power_one_sided(bb_freqs, bb_psd, 16_000.0, 18_000.0)
    n2_p, n2_bw = _band_power_one_sided(bb_freqs, bb_psd, 20_000.0, 22_000.0)
    noise_p = n1_p + n2_p
    noise_bw = n1_bw + n2_bw
    if pilot_bw <= 0.0 or noise_bw <= 0.0:
        pilot_proxy_snr_db = float("nan")
    else:
        noise_density = noise_p / max(noise_bw, EPS)
        noise_in_pilot_bw = noise_density * pilot_bw
        pilot_proxy_snr_db = 10.0 * np.log10((pilot_p + EPS) / (noise_in_pilot_bw + EPS))

    return ConditionMetrics(
        name=name,
        rf_main_db=rf_main_db,
        rf_adjacent_db=rf_adjacent_db,
        rf_selectivity_db=rf_selectivity_db,
        decoder_snr_median_db=decoder_snr_median,
        decoder_snr_mean_db=decoder_snr_mean,
        pilot_proxy_snr_db=pilot_proxy_snr_db,
    )


def capture_live_iq(
    freq_mhz: float,
    sample_rate: int,
    capture_seconds: float,
    block_samples: int,
    warmup_seconds: float,
    use_24bit: bool,
) -> Tuple[np.ndarray, int]:
    radio = r8600.IcomR8600(use_24bit=use_24bit)
    freq_hz = freq_mhz * 1e6
    try:
        radio.open()
        radio.configure_iq_streaming(freq=freq_hz, sample_rate=sample_rate)
        actual_rate = int(radio.iq_sample_rate)
        print(f"Capture setup: {freq_mhz:.3f} MHz, requested {sample_rate} Hz, actual {actual_rate} Hz")
        time.sleep(max(0.0, warmup_seconds))
        radio.flush_iq()
        time.sleep(0.2)

        target_samples = int(actual_rate * capture_seconds)
        chunks = []
        got = 0
        t0 = time.time()
        next_print = t0 + 1.0
        while got < target_samples:
            n = min(block_samples, target_samples - got)
            iq = radio.fetch_iq(n)
            chunks.append(iq.astype(np.complex64, copy=False))
            got += len(iq)
            now = time.time()
            if now >= next_print:
                pct = 100.0 * got / max(target_samples, 1)
                print(f"  Capture progress: {pct:5.1f}% ({got}/{target_samples} samples)")
                next_print = now + 1.0
        elapsed = time.time() - t0
        print(f"Capture complete: {got} samples in {elapsed:.1f} s")
        return np.concatenate(chunks), actual_rate
    finally:
        try:
            radio.close()
        except Exception:
            pass


def save_iq_npz(path: str, iq: np.ndarray, sample_rate: int, freq_hz: float) -> None:
    np.savez_compressed(
        path,
        iq=iq.astype(np.complex64, copy=False),
        sample_rate=np.int64(sample_rate),
        freq_hz=np.float64(freq_hz),
        captured_unix_s=np.float64(time.time()),
    )


def load_iq_npz(path: str) -> Tuple[np.ndarray, int, float]:
    with np.load(path) as data:
        iq = data["iq"].astype(np.complex64, copy=False)
        sample_rate = int(data["sample_rate"]) if "sample_rate" in data else 480000
        freq_hz = float(data["freq_hz"]) if "freq_hz" in data else 89.7e6
    return iq, sample_rate, freq_hz


def verdict(delta_db: float, threshold_db: float = 1.0) -> str:
    if np.isnan(delta_db):
        return "unknown"
    if delta_db > threshold_db:
        return "improved"
    if delta_db < -threshold_db:
        return "worse"
    return "no meaningful change"


def classify_overall_result(
    d_select: float,
    d_dec_snr: float,
    d_pilot: float,
    threshold_db: float = 1.0,
) -> Tuple[str, str]:
    """Classify overall result with explicit selectivity vs SNR tradeoff logic."""
    selectivity_improved = d_select > threshold_db
    selectivity_worse = d_select < -threshold_db
    snr_improved = (d_dec_snr > threshold_db) or (d_pilot > threshold_db)
    snr_worse = (d_dec_snr < -threshold_db) or (d_pilot < -threshold_db)

    if selectivity_improved and not snr_worse and snr_improved:
        return ("beneficial", "Prefilter improved selectivity and at least one SNR metric without SNR regression.")
    if selectivity_improved and snr_worse:
        return ("mixed", "Tradeoff: selectivity improved, but one or more SNR metrics got worse.")
    if selectivity_worse and snr_worse:
        return ("harmful", "Prefilter reduced selectivity and degraded SNR metrics.")
    if not selectivity_improved and snr_improved:
        return ("mixed", "SNR improved but selectivity did not improve materially.")
    if selectivity_worse and not snr_improved:
        return ("harmful", "Prefilter reduced selectivity without SNR benefit.")
    return ("neutral", "No clear net improvement from this capture.")


def resolve_json_output_path(json_out: str, freq_hz: float, cutoff_khz: float) -> str:
    """
    Resolve --json-out path.

    If a directory is passed (e.g. /tmp/), create a timestamped filename in it.
    """
    out = os.path.expanduser(json_out)
    if os.path.isdir(out):
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        freq_tag = f"{freq_hz / 1e6:.1f}".replace(".", "p")
        cutoff_tag = f"{cutoff_khz:.0f}".replace(".", "p")
        out = os.path.join(out, f"prefilter_{freq_tag}MHz_{cutoff_tag}k_{ts}.json")
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out


def main() -> int:
    args = parse_args()
    cutoff_hz = args.cutoff_khz * 1e3

    if args.iq_file:
        iq, sample_rate, file_freq_hz = load_iq_npz(args.iq_file)
        freq_hz = args.freq_mhz * 1e6
        print(f"Loaded IQ file: {args.iq_file}")
        print(f"  File metadata: {file_freq_hz/1e6:.3f} MHz @ {sample_rate} Hz")
        if abs(freq_hz - file_freq_hz) > 1.0:
            print(f"  NOTE: CLI freq is {freq_hz/1e6:.3f} MHz; using file capture for analysis.")
    else:
        iq, sample_rate = capture_live_iq(
            freq_mhz=args.freq_mhz,
            sample_rate=args.sample_rate,
            capture_seconds=args.capture_seconds,
            block_samples=args.block_samples,
            warmup_seconds=args.warmup_seconds,
            use_24bit=args.use_24bit,
        )
        freq_hz = args.freq_mhz * 1e6
        if args.save_iq:
            save_iq_npz(args.save_iq, iq, sample_rate, freq_hz)
            print(f"Saved IQ capture: {args.save_iq}")

    h = design_prefilter(sample_rate=sample_rate, cutoff_hz=cutoff_hz, taps=args.taps)
    iq_pref = apply_prefilter(iq, h)

    # Trim both paths equally to avoid startup/end transients and keep A/B fair.
    trim = max(args.taps * 2, 10_000)
    if len(iq) <= 2 * trim:
        print("ERROR: Capture too short after transient trim; increase --capture-seconds.")
        return 2
    iq_base = iq[trim:-trim]
    iq_pref = iq_pref[trim:-trim]

    print("\nRunning A/B analysis...")
    baseline = evaluate_condition("baseline", iq_base, sample_rate, args.block_samples)
    filtered = evaluate_condition(f"prefilter_{args.cutoff_khz:.0f}k", iq_pref, sample_rate, args.block_samples)

    d_select = filtered.rf_selectivity_db - baseline.rf_selectivity_db
    d_dec_snr = filtered.decoder_snr_median_db - baseline.decoder_snr_median_db
    d_pilot = filtered.pilot_proxy_snr_db - baseline.pilot_proxy_snr_db

    print("\n" + "=" * 70)
    print("LIVE FM PREFILTER COMPARISON")
    print("=" * 70)
    print(f"Frequency: {freq_hz/1e6:.3f} MHz")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Prefilter: +/-{args.cutoff_khz:.1f} kHz, taps={args.taps}")
    print(f"Analyzed samples per path: {len(iq_base)}")

    print("\nBaseline:")
    print(f"  RF selectivity (main/adjacent): {baseline.rf_selectivity_db:.2f} dB")
    print(f"  Decoder SNR median:             {baseline.decoder_snr_median_db:.2f} dB")
    print(f"  Pilot proxy SNR:                {baseline.pilot_proxy_snr_db:.2f} dB")

    print("\nPrefiltered:")
    print(f"  RF selectivity (main/adjacent): {filtered.rf_selectivity_db:.2f} dB")
    print(f"  Decoder SNR median:             {filtered.decoder_snr_median_db:.2f} dB")
    print(f"  Pilot proxy SNR:                {filtered.pilot_proxy_snr_db:.2f} dB")

    print("\nDelta (prefilter - baseline):")
    print(f"  Selectivity delta: {d_select:+.2f} dB -> {verdict(d_select)}")
    print(f"  Decoder SNR delta: {d_dec_snr:+.2f} dB -> {verdict(d_dec_snr)}")
    print(f"  Pilot SNR delta:   {d_pilot:+.2f} dB -> {verdict(d_pilot)}")

    overall_label, overall_reason = classify_overall_result(d_select, d_dec_snr, d_pilot, threshold_db=1.0)
    print("\nConclusion:")
    if overall_label == "beneficial":
        print("  +/-120 kHz pre-filter appears beneficial on this capture.")
    elif overall_label == "harmful":
        print("  +/-120 kHz pre-filter appears harmful on this capture.")
    elif overall_label == "mixed":
        print("  Mixed result: selectivity/SNR tradeoff detected on this capture.")
    else:
        print("  Neutral/mixed result; repeat multiple captures at different times for confidence.")
    print(f"  Detail: {overall_reason}")

    if args.json_out:
        json_path = resolve_json_output_path(args.json_out, freq_hz, args.cutoff_khz)
        report = {
            "timestamp_unix_s": time.time(),
            "frequency_hz": freq_hz,
            "sample_rate_hz": sample_rate,
            "prefilter_cutoff_hz": cutoff_hz,
            "prefilter_taps": args.taps,
            "baseline": asdict(baseline),
            "prefiltered": asdict(filtered),
            "delta": {
                "rf_selectivity_db": d_select,
                "decoder_snr_median_db": d_dec_snr,
                "pilot_proxy_snr_db": d_pilot,
            },
            "overall": {
                "label": overall_label,
                "reason": overall_reason,
            },
        }
        with open(json_path, "w", encoding="ascii") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

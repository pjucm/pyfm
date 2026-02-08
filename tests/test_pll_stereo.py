#!/usr/bin/env python3
"""
PLL vs Pilot-Squaring FM Stereo Decoder Comparison

Generates synthetic FM stereo IQ signals at various SNR levels and measures:
- Stereo separation (dB)
- THD+N (%)
- Audio SNR (dB)
- PLL lock time (ms)
- PLL phase error (°RMS)

Usage:
    python test_pll_stereo.py           # Full sweep with blend OFF/ON tables
    python test_pll_stereo.py --quick   # Quick check (3 SNR points, 3 trials)
    python test_pll_stereo.py --lock-test  # PLL acquisition time analysis
    python test_pll_stereo.py --pe-lpf-test  # Compare PLL PE LPF cutoff before/after
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy import signal as sp_signal

from demodulator import FMStereoDecoder
from pll_stereo_decoder import PLLStereoDecoder


def set_pll_pe_lpf_cutoff(decoder, cutoff_hz):
    """Set PLL phase-error LPF cutoff for a PLLStereoDecoder instance."""
    cutoff_hz = max(1.0, float(cutoff_hz))
    alpha = 2 * np.pi * cutoff_hz / (decoder.iq_sample_rate + 2 * np.pi * cutoff_hz)
    decoder._pll_pe_alpha = alpha
    return cutoff_hz


# ── Synthetic FM Stereo Signal Generator ──────────────────────────────────────

def generate_fm_stereo_iq(duration_s=0.5, fs_iq=480000, deviation=75000,
                          left_tones=None, right_tones=None,
                          pilot_level=0.09, snr_db=40.0, seed=42):
    """
    Generate synthetic FM stereo IQ samples.

    Args:
        duration_s: Duration in seconds
        fs_iq: IQ sample rate in Hz
        deviation: FM deviation in Hz
        left_tones: List of (freq_hz, amplitude) for left channel
        right_tones: List of (freq_hz, amplitude) for right channel
        pilot_level: Pilot tone amplitude (0.09 = 9% of deviation, standard)
        snr_db: RF SNR in dB (noise added to IQ)
        seed: RNG seed for reproducibility

    Returns:
        Complex64 numpy array of IQ samples
    """
    if left_tones is None:
        left_tones = [(1000, 0.3)]
    if right_tones is None:
        right_tones = [(1000, 0.3)]

    rng = np.random.default_rng(seed)
    n = int(duration_s * fs_iq)
    t = np.arange(n) / fs_iq

    # Build L and R audio signals
    left = np.zeros(n, dtype=np.float64)
    for freq, amp in left_tones:
        left += amp * np.sin(2 * np.pi * freq * t)

    right = np.zeros(n, dtype=np.float64)
    for freq, amp in right_tones:
        right += amp * np.sin(2 * np.pi * freq * t)

    # FM stereo multiplex
    lr_sum = (left + right) / 2    # L+R (mono)
    lr_diff = (left - right) / 2   # L-R

    # 19 kHz pilot
    pilot = pilot_level * np.cos(2 * np.pi * 19000 * t)

    # 38 kHz DSB-SC carrier for L-R
    carrier_38k = np.cos(2 * np.pi * 38000 * t)
    lr_diff_mod = lr_diff * carrier_38k

    # Composite multiplex signal
    multiplex = lr_sum + pilot + lr_diff_mod

    # FM modulate: iq = exp(j * 2π * deviation * ∫multiplex dt)
    # Integrate multiplex (cumulative sum scaled by 1/fs)
    phase = 2 * np.pi * deviation * np.cumsum(multiplex) / fs_iq
    iq = np.exp(1j * phase)

    # Add AWGN at specified RF SNR
    if snr_db < 100:
        signal_power = np.mean(np.abs(iq) ** 2)  # Should be ~1.0
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power / 2), n) + \
                1j * rng.normal(0, np.sqrt(noise_power / 2), n)
        iq = iq + noise

    return iq.astype(np.complex64)


# ── Metric Functions ──────────────────────────────────────────────────────────

def measure_stereo_separation(decoder, fs_iq=480000, fs_audio=48000,
                              snr_db=40.0, seed=42, blend_override=None):
    """
    Measure stereo separation: 1 kHz on L only, measure leakage into R.

    Returns separation in dB (higher is better).
    """
    decoder.reset()
    old_blend = None
    if blend_override is not None:
        old_blend = decoder.stereo_blend_enabled
        decoder.stereo_blend_enabled = bool(blend_override)

    try:
        # Generate 1 kHz on left only
        iq = generate_fm_stereo_iq(
            duration_s=0.5, fs_iq=fs_iq, deviation=75000,
            left_tones=[(1000, 0.4)], right_tones=[],
            snr_db=snr_db, seed=seed
        )

        # Process in blocks (like real-time)
        block_size = 2048
        all_audio = []
        for i in range(0, len(iq), block_size):
            block = iq[i:i + block_size]
            if len(block) == 0:
                break
            audio = decoder.demodulate(block)
            all_audio.append(audio)

        audio = np.concatenate(all_audio, axis=0)

        # Skip first 100 ms for filter settling
        skip = int(0.1 * fs_audio)
        if len(audio) <= skip + 1024:
            return 0.0
        audio = audio[skip:]

        left = audio[:, 0]
        right = audio[:, 1]

        # Measure via FFT
        n_fft = min(len(left), 16384)
        left_fft = np.abs(np.fft.rfft(left[:n_fft] * np.hanning(n_fft)))
        right_fft = np.abs(np.fft.rfft(right[:n_fft] * np.hanning(n_fft)))

        # Find 1 kHz bin
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs_audio)
        bin_1k = np.argmin(np.abs(freqs - 1000))

        # Peak power in a ±3 bin window around 1 kHz
        window = 3
        left_power = np.max(left_fft[max(0, bin_1k - window):bin_1k + window + 1]) ** 2
        right_power = np.max(right_fft[max(0, bin_1k - window):bin_1k + window + 1]) ** 2

        if right_power < 1e-20:
            return 80.0  # Effectively infinite separation
        separation = 10 * np.log10(left_power / right_power)
        return separation
    finally:
        if old_blend is not None:
            decoder.stereo_blend_enabled = old_blend


def measure_thd_n(decoder, fs_iq=480000, fs_audio=48000,
                  snr_db=40.0, seed=42):
    """
    Measure THD+N: 1 kHz mono tone, total power minus fundamental / total.

    Returns THD+N as percentage.
    """
    decoder.reset()

    # Generate 1 kHz mono (equal on both channels)
    iq = generate_fm_stereo_iq(
        duration_s=0.5, fs_iq=fs_iq, deviation=75000,
        left_tones=[(1000, 0.3)], right_tones=[(1000, 0.3)],
        snr_db=snr_db, seed=seed
    )

    block_size = 2048
    all_audio = []
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) == 0:
            break
        audio = decoder.demodulate(block)
        all_audio.append(audio)

    audio = np.concatenate(all_audio, axis=0)

    # Skip first 100 ms
    skip = int(0.1 * fs_audio)
    if len(audio) <= skip + 1024:
        return 100.0
    audio = audio[skip:]

    # Use left channel (mono signal)
    mono = audio[:, 0]
    n_fft = min(len(mono), 16384)
    spectrum = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft))) ** 2

    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs_audio)
    bin_1k = np.argmin(np.abs(freqs - 1000))

    # Fundamental power (±3 bins)
    window = 3
    fund_power = np.sum(spectrum[max(0, bin_1k - window):bin_1k + window + 1])

    # Total power
    total_power = np.sum(spectrum)

    if total_power < 1e-20:
        return 100.0

    thd_n = 100.0 * np.sqrt((total_power - fund_power) / total_power)
    return thd_n


def measure_audio_snr(decoder, fs_iq=480000, fs_audio=48000,
                      snr_db=40.0, seed=42):
    """
    Measure audio SNR: 1 kHz mono tone with RF noise.

    Returns audio SNR in dB.
    """
    decoder.reset()

    iq = generate_fm_stereo_iq(
        duration_s=0.5, fs_iq=fs_iq, deviation=75000,
        left_tones=[(1000, 0.3)], right_tones=[(1000, 0.3)],
        snr_db=snr_db, seed=seed
    )

    block_size = 2048
    all_audio = []
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) == 0:
            break
        audio = decoder.demodulate(block)
        all_audio.append(audio)

    audio = np.concatenate(all_audio, axis=0)

    skip = int(0.1 * fs_audio)
    if len(audio) <= skip + 1024:
        return 0.0
    audio = audio[skip:]

    mono = audio[:, 0]
    n_fft = min(len(mono), 16384)
    spectrum = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft))) ** 2

    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs_audio)
    bin_1k = np.argmin(np.abs(freqs - 1000))

    window = 3
    fund_power = np.sum(spectrum[max(0, bin_1k - window):bin_1k + window + 1])
    noise_power = np.sum(spectrum) - fund_power

    if noise_power < 1e-20:
        return 80.0
    return 10 * np.log10(fund_power / noise_power)


def measure_pll_lock_time(fs_iq=480000, snr_db=40.0, seed=42, pe_lpf_cutoff_hz=5000.0):
    """
    Measure PLL acquisition time in milliseconds.

    Processes signal in 256-sample blocks and checks lock state after each.
    Returns time to first lock in ms, or None if never locked.
    """
    decoder = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=48000)
    decoder.stereo_blend_enabled = False
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    set_pll_pe_lpf_cutoff(decoder, pe_lpf_cutoff_hz)

    # Generate stereo signal
    iq = generate_fm_stereo_iq(
        duration_s=1.0, fs_iq=fs_iq, deviation=75000,
        left_tones=[(1000, 0.3)], right_tones=[(1000, 0.3)],
        snr_db=snr_db, seed=seed
    )

    block_size = 256
    samples_processed = 0
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) == 0:
            break
        decoder.demodulate(block)
        samples_processed += len(block)
        if decoder.pll_locked:
            return (samples_processed / fs_iq) * 1000  # ms

    return None  # Never locked


def measure_pll_phase_error(fs_iq=480000, snr_db=40.0, seed=42, pe_lpf_cutoff_hz=5000.0):
    """
    Measure steady-state PLL phase error RMS in degrees.

    Runs signal through PLL and reads phase error after settling.
    """
    decoder = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=48000)
    decoder.stereo_blend_enabled = False
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    set_pll_pe_lpf_cutoff(decoder, pe_lpf_cutoff_hz)

    iq = generate_fm_stereo_iq(
        duration_s=0.5, fs_iq=fs_iq, deviation=75000,
        left_tones=[(1000, 0.3)], right_tones=[(1000, 0.3)],
        snr_db=snr_db, seed=seed
    )

    block_size = 2048
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) == 0:
            break
        decoder.demodulate(block)

    if hasattr(decoder, 'pll_phase_error_deg'):
        return decoder.pll_phase_error_deg
    return decoder.pll_phase_error_rms


# ── Main Test Harness ─────────────────────────────────────────────────────────

def run_comparison(snr_levels, n_trials, fs_iq=480000, fs_audio=48000):
    """Run side-by-side comparison of PLL vs pilot-squaring decoders."""
    def mean(vals):
        valid = [v for v in vals if not np.isnan(v)]
        return np.mean(valid) if valid else float('nan')

    # Lock/phase metrics do not depend on stereo blend mode. Compute once.
    lock_phase = {}
    for snr_db in snr_levels:
        pll_lock_times = []
        pll_phase_errs = []
        for trial in range(n_trials):
            seed = 1000 + trial * 137
            lock_t = measure_pll_lock_time(fs_iq, snr_db, seed)
            pll_lock_times.append(lock_t if lock_t is not None else float('nan'))
            pll_phase_errs.append(measure_pll_phase_error(fs_iq, snr_db, seed + 3))
        lock_phase[snr_db] = (mean(pll_lock_times), mean(pll_phase_errs))

    # Print header
    print()
    print(f"FM Stereo Decoder Comparison: PLL vs Pilot-Squaring")
    print(f"IQ rate: {fs_iq/1000:.0f} kHz, Audio rate: {fs_audio/1000:.0f} kHz, Trials: {n_trials}")
    print()

    for blend_enabled in (False, True):
        scenario = "Blend OFF (raw decoder path)" if not blend_enabled else "Blend ON (adaptive mode)"
        print(scenario)
        print(f"{'SNR':>4s}  │ {'--- PLL Decoder ---':^39s} │ {'--- Pilot-Squaring ---':^28s}")
        print(f"{'(dB)':>4s}  │ {'Sep(dB)':>7s}  {'THD+N%':>6s}  {'ASNR':>6s}  {'Lock ms':>7s}  {'PhErr°':>6s} │ {'Sep(dB)':>7s}  {'THD+N%':>6s}  {'ASNR':>6s}")
        print(f"{'─' * 5}─┼─{'─' * 39}─┼─{'─' * 28}")

        for snr_db in snr_levels:
            # Accumulators
            pll_sep, pll_thd, pll_asnr = [], [], []
            ps_sep, ps_thd, ps_asnr = [], [], []

            for trial in range(n_trials):
                seed = 1000 + trial * 137  # Different but reproducible seeds

                # Create fresh decoders for each trial
                pll_dec = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=fs_audio)
                pll_dec.stereo_blend_enabled = blend_enabled
                pll_dec.bass_boost_enabled = False
                pll_dec.treble_boost_enabled = False

                ps_dec = FMStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=fs_audio)
                ps_dec.stereo_blend_enabled = blend_enabled
                ps_dec.bass_boost_enabled = False
                ps_dec.treble_boost_enabled = False

                # Separation
                pll_sep.append(measure_stereo_separation(
                    pll_dec, fs_iq, fs_audio, snr_db, seed,
                    blend_override=blend_enabled
                ))
                ps_sep.append(measure_stereo_separation(
                    ps_dec, fs_iq, fs_audio, snr_db, seed,
                    blend_override=blend_enabled
                ))

                # THD+N
                pll_thd.append(measure_thd_n(pll_dec, fs_iq, fs_audio, snr_db, seed + 1))
                ps_thd.append(measure_thd_n(ps_dec, fs_iq, fs_audio, snr_db, seed + 1))

                # Audio SNR
                pll_asnr.append(measure_audio_snr(pll_dec, fs_iq, fs_audio, snr_db, seed + 2))
                ps_asnr.append(measure_audio_snr(ps_dec, fs_iq, fs_audio, snr_db, seed + 2))

            ps_row = f"{mean(ps_sep):7.1f}  {mean(ps_thd):6.2f}  {mean(ps_asnr):6.1f}"
            pll_row = f"{mean(pll_sep):7.1f}  {mean(pll_thd):6.2f}  {mean(pll_asnr):6.1f}"

            lock_ms, phase_err = lock_phase[snr_db]
            lock_str = f"{lock_ms:7.1f}" if not np.isnan(lock_ms) else "  never"
            phase_str = f"{phase_err:6.2f}" if not np.isnan(phase_err) else "   N/A"

            print(f"{snr_db:4.0f}  │ {pll_row}  {lock_str}  {phase_str} │ {ps_row}")

        print()


def run_pe_lpf_before_after_test(snr_levels, n_trials, fs_iq=480000, fs_audio=48000,
                                 cutoff_before_hz=5000.0, cutoff_after_hz=1000.0):
    """Compare PLL performance before/after changing PE LPF cutoff."""
    cutoffs = [('Before', cutoff_before_hz), ('After', cutoff_after_hz)]

    def mean(vals):
        valid = [v for v in vals if not np.isnan(v)]
        return np.mean(valid) if valid else float('nan')

    print()
    print("PLL PE LPF Cutoff Comparison (Blend OFF / raw decode)")
    print(f"IQ rate: {fs_iq/1000:.0f} kHz, Audio rate: {fs_audio/1000:.0f} kHz, Trials: {n_trials}")
    print(f"Before cutoff: {cutoff_before_hz:.0f} Hz, After cutoff: {cutoff_after_hz:.0f} Hz")
    print()
    print(f"{'SNR':>4s}  {'Case':>7s}  {'Cutoff':>8s}  {'Sep(dB)':>7s}  {'THD+N%':>6s}  {'ASNR':>6s}  {'Lock ms':>7s}  {'PhErr°':>6s}")
    print(f"{'─' * 4}  {'─' * 7}  {'─' * 8}  {'─' * 7}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 6}")

    for snr_db in snr_levels:
        row = {}
        for label, cutoff_hz in cutoffs:
            pll_sep, pll_thd, pll_asnr = [], [], []
            pll_lock_times, pll_phase_errs = [], []

            for trial in range(n_trials):
                seed = 5000 + trial * 137

                dec = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=fs_audio)
                dec.stereo_blend_enabled = False
                dec.bass_boost_enabled = False
                dec.treble_boost_enabled = False
                set_pll_pe_lpf_cutoff(dec, cutoff_hz)

                pll_sep.append(measure_stereo_separation(
                    dec, fs_iq, fs_audio, snr_db, seed, blend_override=False
                ))
                pll_thd.append(measure_thd_n(dec, fs_iq, fs_audio, snr_db, seed + 1))
                pll_asnr.append(measure_audio_snr(dec, fs_iq, fs_audio, snr_db, seed + 2))

                lock_t = measure_pll_lock_time(
                    fs_iq=fs_iq, snr_db=snr_db, seed=seed + 3,
                    pe_lpf_cutoff_hz=cutoff_hz
                )
                pll_lock_times.append(lock_t if lock_t is not None else float('nan'))
                pll_phase_errs.append(measure_pll_phase_error(
                    fs_iq=fs_iq, snr_db=snr_db, seed=seed + 4,
                    pe_lpf_cutoff_hz=cutoff_hz
                ))

            sep = mean(pll_sep)
            thd = mean(pll_thd)
            asnr = mean(pll_asnr)
            lock_ms = mean(pll_lock_times)
            phase_err = mean(pll_phase_errs)
            row[label] = (sep, thd, asnr, lock_ms, phase_err)

            lock_str = f"{lock_ms:7.1f}" if not np.isnan(lock_ms) else "  never"
            phase_str = f"{phase_err:6.2f}" if not np.isnan(phase_err) else "   N/A"
            print(f"{snr_db:4.0f}  {label:>7s}  {cutoff_hz:8.0f}  {sep:7.1f}  {thd:6.2f}  {asnr:6.1f}  {lock_str}  {phase_str}")

        b = row['Before']
        a = row['After']
        print(f"{'':4s}  {'ΔA-B':>7s}  {'':8s}  {a[0]-b[0]:+7.1f}  {a[1]-b[1]:+6.2f}  {a[2]-b[2]:+6.1f}  {a[3]-b[3]:+7.2f}  {a[4]-b[4]:+6.2f}")
    print()


def run_lock_test(fs_iq=480000):
    """Detailed PLL lock acquisition analysis."""
    print()
    print("PLL Lock Acquisition Analysis")
    print(f"IQ rate: {fs_iq/1000:.0f} kHz")
    print()
    print(f"{'SNR(dB)':>7s}  {'Lock Time (ms)':>14s}  {'Phase Err (°)':>13s}  {'Freq Offset (Hz)':>16s}  {'Status':>8s}")
    print(f"{'─' * 7}  {'─' * 14}  {'─' * 13}  {'─' * 16}  {'─' * 8}")

    snr_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    n_trials = 5

    for snr_db in snr_levels:
        lock_times = []
        phase_errs = []
        freq_offsets = []

        for trial in range(n_trials):
            seed = 2000 + trial * 137

            decoder = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=48000)
            decoder.stereo_blend_enabled = False
            decoder.bass_boost_enabled = False
            decoder.treble_boost_enabled = False

            iq = generate_fm_stereo_iq(
                duration_s=1.0, fs_iq=fs_iq, deviation=75000,
                left_tones=[(1000, 0.3)], right_tones=[(1000, 0.3)],
                snr_db=snr_db, seed=seed
            )

            block_size = 256
            samples_processed = 0
            locked_time = None
            for i in range(0, len(iq), block_size):
                block = iq[i:i + block_size]
                if len(block) == 0:
                    break
                decoder.demodulate(block)
                samples_processed += len(block)
                if decoder.pll_locked and locked_time is None:
                    locked_time = (samples_processed / fs_iq) * 1000

            lock_times.append(locked_time if locked_time is not None else float('nan'))
            phase_errs.append(decoder.pll_phase_error_rms)
            freq_offsets.append(decoder.pll_frequency_offset)

        avg_lock = np.nanmean(lock_times)
        avg_phase = np.mean(phase_errs)
        avg_freq = np.mean(freq_offsets)
        locked_count = sum(1 for t in lock_times if not np.isnan(t))

        lock_str = f"{avg_lock:14.1f}" if not np.isnan(avg_lock) else "         never"
        status = f"{locked_count}/{n_trials}"

        print(f"{snr_db:7.0f}  {lock_str}  {avg_phase:13.2f}  {avg_freq:16.3f}  {status:>8s}")

    print()


def main():
    parser = argparse.ArgumentParser(description="PLL vs Pilot-Squaring FM Stereo Comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 SNR points, 3 trials")
    parser.add_argument("--lock-test", action="store_true",
                        help="PLL lock acquisition analysis")
    parser.add_argument("--pe-lpf-test", action="store_true",
                        help="PLL PE LPF before/after comparison")
    parser.add_argument("--pe-before", type=float, default=5000.0,
                        help="Before PE LPF cutoff in Hz (default: 5000)")
    parser.add_argument("--pe-after", type=float, default=1000.0,
                        help="After PE LPF cutoff in Hz (default: 1000)")
    parser.add_argument("--fs-iq", type=int, default=480000,
                        help="IQ sample rate (default: 480000)")
    args = parser.parse_args()

    t_start = time.time()

    if args.lock_test:
        run_lock_test(fs_iq=args.fs_iq)
    elif args.pe_lpf_test:
        if args.quick:
            snr_levels = [15, 25, 40]
            n_trials = 3
        else:
            snr_levels = list(range(0, 45, 5))
            n_trials = 5
        run_pe_lpf_before_after_test(
            snr_levels, n_trials, fs_iq=args.fs_iq,
            cutoff_before_hz=args.pe_before, cutoff_after_hz=args.pe_after
        )
    else:
        if args.quick:
            snr_levels = [15, 25, 40]
            n_trials = 3
        else:
            snr_levels = list(range(0, 45, 5))  # 0, 5, 10, ..., 40
            n_trials = 5

        run_comparison(snr_levels, n_trials, fs_iq=args.fs_iq)

    elapsed = time.time() - t_start
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
L+R / L-R LPF Tap Count Test Harness

Sweeps different FIR filter lengths for the 15 kHz L+R and L-R lowpass filters
in PLLStereoDecoder.  Measures:

  - Pilot rejection at 19 kHz (the bottleneck identified in IHF_SNR_ANALYSIS.txt)
  - IHF/EIA SNR (A-weighted, de-emphasized)
  - Stereo separation (1 kHz left-only signal)
  - Processing throughput (blocks/sec and real-time multiple)
  - Per-filter CPU time breakdown

Usage:
    python3 test-taps-claude.py                    # Default sweep
    python3 test-taps-claude.py 127 255 511        # Specific tap counts
    python3 test-taps-claude.py --rate 960000      # Custom IQ rate
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# A-weighting filter (IEC 61672)
# ---------------------------------------------------------------------------

def design_a_weighting(fs):
    z = [0, 0, 0, 0]
    p = [-2 * np.pi * 20.598997, -2 * np.pi * 20.598997,
         -2 * np.pi * 107.65265, -2 * np.pi * 737.86223,
         -2 * np.pi * 12194.217, -2 * np.pi * 12194.217]
    k = (2 * np.pi * 12194.217) ** 2
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    w, h = signal.sosfreqz(sos, worN=[1000], fs=fs)
    sos[0, :3] /= np.abs(h[0])
    return sos


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_stereo_iq(iq_rate, duration, left_amp=0.5, right_amp=0.5,
                       left_freq=1000, right_freq=1000,
                       rf_snr_db=None, rng=None):
    """Generate FM-modulated stereo IQ with pilot, optional AWGN."""
    n = int(duration * iq_rate)
    t = np.arange(n) / iq_rate

    left = left_amp * np.sin(2 * np.pi * left_freq * t)
    right = right_amp * np.sin(2 * np.pi * right_freq * t)

    lr_sum = (left + right) / 2
    lr_diff = (left - right) / 2

    pilot = 0.09 * np.sin(2 * np.pi * 19000 * t)
    carrier = -np.cos(2 * np.pi * 38000 * t)

    multiplex = lr_sum * 0.9 + pilot + lr_diff * carrier * 0.9

    dt = 1.0 / iq_rate
    phase = 2 * np.pi * 75000 * np.cumsum(multiplex) * dt
    iq = (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)

    if rf_snr_db is not None:
        if rng is None:
            rng = np.random.default_rng(42)
        sig_power = np.mean(np.abs(iq) ** 2)
        noise_power = sig_power / (10 ** (rf_snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))
        iq = iq + noise.astype(np.complex64)

    return iq


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_snr_fft(x, tone_freq, fs, bw=100):
    """Tone-to-residual SNR via windowed FFT."""
    n = len(x)
    w = np.hanning(n)
    fft = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    ps = np.abs(fft) ** 2 / n
    mask = np.abs(freqs - tone_freq) <= bw / 2
    tone_power = np.sum(ps[mask])
    noise_power = np.sum(ps[~mask])
    if noise_power <= 0:
        return np.inf
    return 10 * np.log10(tone_power / noise_power)


def measure_separation(left_out, right_out, freq, fs):
    """Channel separation: power of freq in left vs right."""
    n = min(len(left_out), len(right_out))
    w = np.hanning(n)

    fft_l = np.fft.rfft(left_out[:n] * w)
    fft_r = np.fft.rfft(right_out[:n] * w)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask = np.abs(freqs - freq) <= 50

    lp = np.sum(np.abs(fft_l[mask]) ** 2)
    rp = np.sum(np.abs(fft_r[mask]) ** 2)
    if rp <= 0:
        return np.inf
    return 10 * np.log10(lp / rp)


def lpf_rejection_at(taps, beta, cutoff_hz, test_hz, fs):
    """Compute LPF rejection at a specific frequency."""
    lpf = signal.firwin(taps, cutoff_hz, fs=fs, window=('kaiser', beta))
    w, h = signal.freqz(lpf, worN=[2 * np.pi * test_hz / fs])
    return 20 * np.log10(np.abs(h[0]) + 1e-30)


# ---------------------------------------------------------------------------
# Monkey-patch decoder to use custom tap counts
# ---------------------------------------------------------------------------

def patch_decoder(decoder, lpf_taps, beta=6.0):
    """Replace L+R and L-R LPF filters with custom tap count."""
    nyq = decoder.iq_sample_rate / 2
    cutoff = 15000 / nyq

    decoder.lr_sum_lpf = signal.firwin(lpf_taps, cutoff, window=('kaiser', beta))
    decoder.lr_sum_lpf_state = signal.lfilter_zi(decoder.lr_sum_lpf, 1.0)

    decoder.lr_diff_lpf = signal.firwin(lpf_taps, cutoff, window=('kaiser', beta))
    decoder.lr_diff_lpf_state = signal.lfilter_zi(decoder.lr_diff_lpf, 1.0)

    # Update group delay compensation to match new BPF + LPF delay budget.
    # The L-R path delay = BPF center delay + new LPF center delay.
    # The L+R path must match: new LPF center delay + delay_buf = L-R path delay.
    # So delay_buf = BPF center delay = (len(lr_diff_bpf) - 1) // 2 (unchanged).
    # The BPF hasn't changed, so the existing delay buffer size is still correct.


def demodulate_blocks(decoder, iq, block_size=8192):
    """Process IQ in blocks, return audio and elapsed time."""
    chunks = []
    t_start = time.perf_counter()
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) > 0:
            chunks.append(decoder.demodulate(block))
    elapsed = time.perf_counter() - t_start
    audio = np.vstack(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)
    return audio, elapsed


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(lpf_taps, iq_rate, audio_rate, a_sos, iq_mono, iq_stereo,
             duration, beta=6.0):
    """Run full test suite for one tap count. Returns result dict."""
    from pll_stereo_decoder import PLLStereoDecoder

    result = {'taps': lpf_taps}

    # --- Filter-only measurement ---
    result['rej_19k'] = lpf_rejection_at(lpf_taps, beta, 15000, 19000, iq_rate)

    skip = int(0.1 * audio_rate)

    # --- IHF SNR (mono: identical L+R) ---
    dec = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                          deviation=75000, deemphasis=75e-6, force_mono=True)
    dec.bass_boost_enabled = False
    dec.treble_boost_enabled = False
    dec.stereo_blend_enabled = False
    patch_decoder(dec, lpf_taps, beta)

    audio_m, _ = demodulate_blocks(dec, iq_mono)
    left_m = audio_m[skip:-skip, 0]
    left_m_aw = signal.sosfilt(a_sos, left_m)
    result['ihf_mono'] = measure_snr_fft(left_m_aw, 1000, audio_rate)

    # --- IHF SNR (stereo: identical L+R with pilot) ---
    dec_s = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                            deviation=75000, deemphasis=75e-6)
    dec_s.bass_boost_enabled = False
    dec_s.treble_boost_enabled = False
    dec_s.stereo_blend_enabled = False
    patch_decoder(dec_s, lpf_taps, beta)

    audio_s, _ = demodulate_blocks(dec_s, iq_mono)
    left_s = audio_s[skip:-skip, 0]
    left_s_aw = signal.sosfilt(a_sos, left_s)
    result['ihf_stereo'] = measure_snr_fft(left_s_aw, 1000, audio_rate)

    # --- Raw SNR (stereo, no A-weighting, no de-emphasis) ---
    dec_raw = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                              deviation=75000, deemphasis=1e-9)
    dec_raw.bass_boost_enabled = False
    dec_raw.treble_boost_enabled = False
    dec_raw.stereo_blend_enabled = False
    patch_decoder(dec_raw, lpf_taps, beta)

    audio_raw, _ = demodulate_blocks(dec_raw, iq_mono)
    left_raw = audio_raw[skip:-skip, 0]
    result['raw_stereo'] = measure_snr_fft(left_raw, 1000, audio_rate)

    # --- Stereo separation (1 kHz left only) ---
    dec_sep = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                              deviation=75000, deemphasis=1e-9)
    dec_sep.bass_boost_enabled = False
    dec_sep.treble_boost_enabled = False
    dec_sep.stereo_blend_enabled = False
    patch_decoder(dec_sep, lpf_taps, beta)

    audio_sep, _ = demodulate_blocks(dec_sep, iq_stereo)
    half = len(audio_sep) // 2
    result['separation'] = measure_separation(
        audio_sep[half:, 0], audio_sep[half:, 1], 1000, audio_rate)

    # --- Throughput benchmark (5 runs, best of 3) ---
    dec_bench = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                                deviation=75000, deemphasis=75e-6)
    dec_bench.bass_boost_enabled = False
    dec_bench.treble_boost_enabled = False
    dec_bench.stereo_blend_enabled = False
    patch_decoder(dec_bench, lpf_taps, beta)

    # Warm up
    demodulate_blocks(dec_bench, iq_mono[:iq_rate // 10])
    dec_bench.reset()
    patch_decoder(dec_bench, lpf_taps, beta)

    timings = []
    for _ in range(5):
        dec_bench.reset()
        patch_decoder(dec_bench, lpf_taps, beta)
        _, elapsed = demodulate_blocks(dec_bench, iq_mono)
        timings.append(elapsed)

    timings.sort()
    best3 = np.mean(timings[:3])
    result['time_sec'] = best3
    result['realtime_x'] = duration / best3

    # --- Per-filter CPU time (single run with profiling) ---
    dec_prof = PLLStereoDecoder(iq_sample_rate=iq_rate, audio_sample_rate=audio_rate,
                               deviation=75000, deemphasis=75e-6)
    dec_prof.bass_boost_enabled = False
    dec_prof.treble_boost_enabled = False
    dec_prof.stereo_blend_enabled = False
    dec_prof.profile_enabled = True
    patch_decoder(dec_prof, lpf_taps, beta)
    demodulate_blocks(dec_prof, iq_mono)
    result['profile'] = dict(dec_prof._profile)

    return result


def main():
    # Parse arguments
    iq_rate = 480000
    audio_rate = 48000
    duration = 2.0
    beta = 6.0
    rf_snr_db = None

    tap_values = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--rate' and i + 1 < len(args):
            iq_rate = int(args[i + 1])
            i += 2
        elif args[i] == '--beta' and i + 1 < len(args):
            beta = float(args[i + 1])
            i += 2
        elif args[i] == '--duration' and i + 1 < len(args):
            duration = float(args[i + 1])
            i += 2
        elif args[i] == '--rf-snr' and i + 1 < len(args):
            rf_snr_db = float(args[i + 1])
            i += 2
        elif args[i].isdigit():
            tap_values.append(int(args[i]))
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    if not tap_values:
        tap_values = [127, 201, 255, 301, 401, 511, 701, 1023]

    # Ensure odd tap counts (FIR type I)
    tap_values = [t | 1 for t in tap_values]

    print("=" * 78)
    print("L+R / L-R LPF Tap Count Sweep")
    print("=" * 78)
    print(f"  IQ sample rate:    {iq_rate / 1000:.0f} kHz")
    print(f"  Audio sample rate: {audio_rate / 1000:.0f} kHz")
    print(f"  Kaiser beta:       {beta}")
    print(f"  RF SNR:            {rf_snr_db:.0f} dB" if rf_snr_db else
          f"  RF SNR:            clean (no noise)")
    print(f"  Test duration:     {duration:.1f} s")
    print(f"  Tap values:        {tap_values}")
    print()

    # Pre-generate test signals (shared across all tap values)
    print("Generating test signals...", end="", flush=True)
    a_sos = design_a_weighting(audio_rate)
    rng = np.random.default_rng(42)

    # Mono IQ (identical L+R with pilot) — for IHF SNR
    iq_mono = generate_stereo_iq(iq_rate, duration,
                                 left_freq=1000, right_freq=1000,
                                 rf_snr_db=rf_snr_db, rng=rng)

    # Stereo IQ (left-only) — for separation measurement
    iq_stereo = generate_stereo_iq(iq_rate, duration,
                                   left_amp=0.5, right_amp=0.0,
                                   left_freq=1000, right_freq=1000,
                                   rf_snr_db=rf_snr_db, rng=rng)
    print(" done.\n")

    # Run tests
    results = []
    for taps in tap_values:
        print(f"--- Testing {taps} taps ---")
        r = run_test(taps, iq_rate, audio_rate, a_sos, iq_mono, iq_stereo,
                     duration, beta)
        results.append(r)

        print(f"  Pilot rejection @ 19 kHz:  {r['rej_19k']:.1f} dB")
        print(f"  Raw stereo SNR:            {r['raw_stereo']:.1f} dB")
        print(f"  IHF stereo SNR:            {r['ihf_stereo']:.1f} dB")
        print(f"  IHF mono SNR:              {r['ihf_mono']:.1f} dB")
        print(f"  Stereo separation @ 1 kHz: {r['separation']:.1f} dB")
        print(f"  Processing time:           {r['time_sec']*1000:.1f} ms"
              f"  ({r['realtime_x']:.1f}x real-time)")

        # Show top CPU consumers
        prof = r['profile']
        top = sorted(prof.items(), key=lambda x: -x[1])[:4]
        parts = [f"{k}={v:.0f}us" for k, v in top if v > 0]
        if parts:
            print(f"  CPU breakdown (EMA us):    {', '.join(parts)}")
        print()

    # Summary table
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    hdr = (f"  {'Taps':>6s}  {'19kHz rej':>9s}  {'Raw Stereo':>10s}  "
           f"{'IHF Stereo':>10s}  {'IHF Mono':>10s}  {'Separation':>10s}  "
           f"{'Time(ms)':>8s}  {'RT mult':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        print(f"  {r['taps']:6d}  {r['rej_19k']:+8.1f} dB"
              f"  {r['raw_stereo']:8.1f} dB"
              f"  {r['ihf_stereo']:8.1f} dB"
              f"  {r['ihf_mono']:8.1f} dB"
              f"  {r['separation']:8.1f} dB"
              f"  {r['time_sec']*1000:8.1f}"
              f"  {r['realtime_x']:6.1f}x")

    print()

    # Find sweet spot: best IHF stereo SNR that still runs > 10x real-time
    viable = [r for r in results if r['realtime_x'] > 10]
    if viable:
        best = max(viable, key=lambda r: r['ihf_stereo'])
        print(f"  Recommended: {best['taps']} taps"
              f" ({best['ihf_stereo']:.1f} dB IHF stereo,"
              f" {best['realtime_x']:.1f}x real-time)")
    else:
        print("  WARNING: No tap count achieves >10x real-time.")
        best = max(results, key=lambda r: r['realtime_x'])
        print(f"  Fastest: {best['taps']} taps at {best['realtime_x']:.1f}x real-time")

    # Show diminishing returns
    if len(results) >= 2:
        print()
        print("  Diminishing returns analysis:")
        prev = results[0]
        for r in results[1:]:
            snr_gain = r['ihf_stereo'] - prev['ihf_stereo']
            tap_cost = r['taps'] - prev['taps']
            speed_ratio = r['realtime_x'] / prev['realtime_x'] if prev['realtime_x'] > 0 else 0
            print(f"    {prev['taps']:4d} -> {r['taps']:4d}:"
                  f"  +{snr_gain:.1f} dB IHF,"
                  f"  {speed_ratio:.2f}x speed")
            prev = r


if __name__ == "__main__":
    main()

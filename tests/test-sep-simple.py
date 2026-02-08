#!/usr/bin/env python3
"""
Simplified stereo separation test - generates signal directly at baseband rate.
"""

import numpy as np
from scipy import signal

from demodulator import FMStereoDecoder
from pll_stereo_decoder import PLLStereoDecoder


def generate_simple_composite(test_freq, duration, fs_baseband):
    """
    Generate FM stereo composite directly at baseband rate.

    Test signal: pure tone at test_freq in LEFT channel only.

    Returns: composite baseband signal
    """
    n_samples = int(duration * fs_baseband)
    t = np.arange(n_samples) / fs_baseband

    # Generate test tone (bandlimited to 15 kHz)
    test_tone = np.sin(2 * np.pi * test_freq * t)

    # Apply 15 kHz lowpass to simulate broadcast audio bandwidth
    nyq = fs_baseband / 2
    lpf = signal.firwin(201, 15000 / nyq, window='hamming')
    test_tone = signal.filtfilt(lpf, 1, test_tone)

    # Scale to typical modulation level
    test_tone = test_tone * 0.4

    # Generate pilot and carrier with correct phase relationship
    # FM stereo standard: 38 kHz carrier is derived by doubling 19 kHz pilot
    # Using cos(2θ) = 2cos²(θ) - 1 identity:
    pilot_19k = np.cos(2 * np.pi * 19000 * t)
    carrier_38k = 2 * np.cos(2 * np.pi * 19000 * t) ** 2 - 1  # = cos(38000*t)

    # Verify phase relationship
    carrier_38k_direct = np.cos(2 * np.pi * 38000 * t)
    phase_error = np.sqrt(np.mean((carrier_38k - carrier_38k_direct) ** 2))
    print(f"  Phase relationship check: RMS error = {phase_error:.9f} (should be ~0)")

    # Left = test_tone, Right = 0
    left = test_tone
    right = np.zeros_like(test_tone)

    lr_sum = left + right  # = test_tone
    lr_diff = left - right  # = test_tone

    # FM stereo composite
    composite = lr_sum + 0.09 * pilot_19k + lr_diff * carrier_38k

    print(f"Generated composite:")
    print(f"  Samples: {len(composite)}")
    print(f"  Test tone freq: {test_freq} Hz")
    print(f"  L+R RMS: {np.sqrt(np.mean(lr_sum**2)):.6f}")
    print(f"  L-R RMS: {np.sqrt(np.mean(lr_diff**2)):.6f}")
    print(f"  Pilot RMS: {np.sqrt(np.mean(pilot_19k**2)):.6f}")
    print(f"  Composite RMS: {np.sqrt(np.mean(composite**2)):.6f}")

    return composite


def fm_modulate(baseband, fs, deviation=75000):
    """Simple FM modulator."""
    phase = 2 * np.pi * deviation * np.cumsum(baseband) / fs
    return np.exp(1j * phase).astype(np.complex64)


def test_decoder(decoder_class, iq_samples, fs_iq, fs_audio, test_freq):
    """Test a decoder and measure separation."""
    print(f"\nTesting {decoder_class.__name__}:")

    decoder = decoder_class(
        iq_sample_rate=fs_iq,
        audio_sample_rate=fs_audio,
        force_mono=False
    )

    # Decode in blocks
    block_size = 8192
    warmup_blocks = 5

    left_chunks = []
    right_chunks = []

    for i in range(0, len(iq_samples), block_size):
        block = iq_samples[i:i+block_size]
        if len(block) < block_size // 2:
            break

        audio = decoder.demodulate(block)

        # Skip warmup
        if i >= warmup_blocks * block_size:
            left_chunks.append(audio[:, 0])
            right_chunks.append(audio[:, 1])

    left = np.concatenate(left_chunks)
    right = np.concatenate(right_chunks)

    # Print decoder state
    print(f"  Decoder state:")
    print(f"    SNR: {decoder.snr_db:.1f} dB")
    print(f"    Stereo blend: {decoder.stereo_blend_factor:.3f}")
    if hasattr(decoder, 'pll_locked'):
        print(f"    PLL locked: {decoder.pll_locked}")
    else:
        print(f"    Pilot detected: {decoder.pilot_detected}")

    # Print audio levels
    print(f"  Decoded audio:")
    print(f"    Left RMS: {np.sqrt(np.mean(left**2)):.6f}")
    print(f"    Right RMS: {np.sqrt(np.mean(right**2)):.6f}")
    print(f"    Left peak: {np.max(np.abs(left)):.6f}")
    print(f"    Right peak: {np.max(np.abs(right)):.6f}")

    # Measure separation using FFT
    n_fft = len(left)
    window = np.hanning(n_fft)

    left_fft = np.fft.rfft(left * window)
    right_fft = np.fft.rfft(right * window)
    freqs = np.fft.rfftfreq(n_fft, 1/fs_audio)

    test_bin = np.argmin(np.abs(freqs - test_freq))

    # Sum power in test bin ±2 bins
    left_power = np.sum(np.abs(left_fft[test_bin-2:test_bin+3]) ** 2)
    right_power = np.sum(np.abs(right_fft[test_bin-2:test_bin+3]) ** 2)

    if right_power > 0:
        separation_db = 10 * np.log10(left_power / right_power)
    else:
        separation_db = float('inf')

    print(f"  Separation measurement:")
    print(f"    Test bin: {test_bin} ({freqs[test_bin]:.1f} Hz)")
    print(f"    Left power: {left_power:.3e}")
    print(f"    Right power: {right_power:.3e}")
    print(f"    Separation: {separation_db:.1f} dB")

    return separation_db


def main():
    print("=" * 70)
    print("SIMPLIFIED STEREO SEPARATION TEST")
    print("=" * 70)

    # Test parameters
    fs_baseband = 480000
    fs_audio = 48000
    duration = 1.0
    test_freq = 1000  # Hz

    print(f"\nParameters:")
    print(f"  Baseband rate: {fs_baseband} Hz")
    print(f"  Audio rate: {fs_audio} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Test frequency: {test_freq} Hz in LEFT channel only")
    print()

    # Generate test signal
    composite = generate_simple_composite(test_freq, duration, fs_baseband)

    # FM modulate
    print(f"\nFM modulating...")
    iq_samples = fm_modulate(composite, fs_baseband)
    print(f"  IQ samples: {len(iq_samples)}")

    # Test both decoders
    print("\n" + "=" * 70)
    sep1 = test_decoder(FMStereoDecoder, iq_samples, fs_baseband, fs_audio, test_freq)

    print("\n" + "=" * 70)
    sep2 = test_decoder(PLLStereoDecoder, iq_samples, fs_baseband, fs_audio, test_freq)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"FMStereoDecoder (pilot-squaring): {sep1:.1f} dB")
    print(f"PLLStereoDecoder (PLL):           {sep2:.1f} dB")
    print()
    print("Expected: >40 dB for good stereo separation")
    print("=" * 70)


if __name__ == '__main__':
    main()

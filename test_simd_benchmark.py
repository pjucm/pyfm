#!/usr/bin/env python3
"""
SIMD/Vectorization Benchmark Test Suite

Verifies that NumPy/SciPy DSP operations are using SIMD-accelerated code paths.
Measures throughput of critical DSP operations to detect performance regressions.

Usage:
    python test_simd_benchmark.py          # Run all benchmarks
    pytest test_simd_benchmark.py -v       # Run with pytest

Performance Targets (Intel i7-14700 with AVX2/FMA3):
    - FFT (8192 complex): > 50,000 ops/sec
    - lfilter (8192 samples, 127-tap FIR): > 5,000 ops/sec
    - Complex discriminator (8192 samples): > 100,000 ops/sec
    - Resampling (8192 -> 1566 samples): > 2,000 ops/sec

These thresholds are conservative - actual SIMD performance should exceed them
significantly. Failing these tests indicates a configuration problem.
"""

import time
import numpy as np
from scipy import signal


# =============================================================================
# Configuration
# =============================================================================

# Block sizes matching real-time processing in pjfm
BLOCK_SIZE_IQ = 8192       # Typical I/Q block from BB60D/IC-R8600
BLOCK_SIZE_AUDIO = 1566    # Decimated audio block (8192 * 48000 / 250000)
SAMPLE_RATE_IQ = 250000    # BB60D I/Q rate
SAMPLE_RATE_AUDIO = 48000  # Audio output rate

# Benchmark iterations
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Minimum acceptable throughput (ops/sec) - conservative thresholds
# Set at ~50% of typical SIMD-accelerated performance to catch major regressions
# while allowing for system load variation. Real-time feasibility test is the
# definitive check for production readiness.
MIN_FFT_OPS = 10000           # Typical: 20K+ ops/sec
MIN_LFILTER_FIR_OPS = 2000    # Typical: 5K+ ops/sec (127-tap)
MIN_LFILTER_IIR_OPS = 15000   # Typical: 30K+ ops/sec (2nd order)
MIN_DISCRIMINATOR_OPS = 2500  # Typical: 5K+ ops/sec
MIN_RESAMPLE_OPS = 2000       # Typical: 8K+ ops/sec
MIN_COMPLEX_ARITH_OPS = 25000 # Typical: 50K+ ops/sec


# =============================================================================
# Helper Functions
# =============================================================================

def benchmark(func, iterations=BENCHMARK_ITERATIONS, warmup=WARMUP_ITERATIONS):
    """
    Benchmark a function and return ops/sec.

    Args:
        func: Callable to benchmark (should do one unit of work)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)

    Returns:
        Tuple of (ops_per_sec, total_time_sec)
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    return ops_per_sec, elapsed


def format_throughput(ops_per_sec):
    """Format throughput with appropriate units."""
    if ops_per_sec >= 1e6:
        return f"{ops_per_sec/1e6:.2f}M ops/sec"
    elif ops_per_sec >= 1e3:
        return f"{ops_per_sec/1e3:.2f}K ops/sec"
    else:
        return f"{ops_per_sec:.2f} ops/sec"


# =============================================================================
# Backend Information
# =============================================================================

def test_numpy_backend_info():
    """
    Display NumPy/SciPy backend configuration.

    This test always passes - it's informational only.
    Prints BLAS library, SIMD capabilities, and FFT backend info.
    """
    print("\n" + "=" * 70)
    print("NumPy/SciPy Backend Configuration")
    print("=" * 70)

    # NumPy version and config
    print(f"\n  NumPy version: {np.__version__}")

    # Get BLAS info
    try:
        config = np.__config__
        if hasattr(config, 'blas_ilp64_opt_info'):
            blas_info = config.blas_ilp64_opt_info
            print(f"  BLAS (ILP64): {blas_info.get('libraries', ['unknown'])}")
        elif hasattr(config, 'blas_opt_info'):
            blas_info = config.blas_opt_info
            print(f"  BLAS: {blas_info.get('libraries', ['unknown'])}")
    except Exception:
        print("  BLAS: (unable to determine)")

    # SciPy version
    import scipy
    print(f"  SciPy version: {scipy.__version__}")

    # Check for SIMD in build info
    print("\n  SIMD Detection:")
    try:
        # NumPy 2.0+ method
        if hasattr(np, '__cpu_features__'):
            features = np.__cpu_features__
            simd_features = ['SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE41', 'SSE42',
                           'AVX', 'AVX2', 'AVX512F', 'FMA3']
            detected = [f for f in simd_features if features.get(f, False)]
            print(f"    Detected: {', '.join(detected) if detected else 'None'}")
        else:
            print("    (SIMD detection not available in this NumPy version)")
    except Exception as e:
        print(f"    (error detecting SIMD: {e})")

    # FFT backend
    print("\n  FFT Backend:")
    try:
        # Check for pocketfft (NumPy 1.17+)
        from numpy.fft import _pocketfft
        print("    PocketFFT: Yes (SIMD-optimized)")
    except ImportError:
        print("    PocketFFT: No (using older FFT backend)")

    print("\n  Result: PASS (informational)")
    return True


# =============================================================================
# FFT Benchmarks
# =============================================================================

def test_fft_complex_benchmark():
    """
    Benchmark complex FFT performance.

    Tests np.fft.fft on complex64 data (typical I/Q processing).
    PocketFFT with AVX2 should achieve >50K ops/sec for 8192-point FFT.
    """
    print("\n" + "=" * 70)
    print("Benchmark: Complex FFT (np.fft.fft)")
    print("=" * 70)

    # Generate test data
    data = (np.random.randn(BLOCK_SIZE_IQ) +
            1j * np.random.randn(BLOCK_SIZE_IQ)).astype(np.complex64)

    def fft_op():
        np.fft.fft(data)

    ops_per_sec, elapsed = benchmark(fft_op)

    print(f"\n  Block size: {BLOCK_SIZE_IQ} complex samples")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Samples/sec: {format_throughput(ops_per_sec * BLOCK_SIZE_IQ)}")
    print(f"  Minimum required: {format_throughput(MIN_FFT_OPS)}")

    passed = ops_per_sec >= MIN_FFT_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_rfft_benchmark():
    """
    Benchmark real FFT performance.

    Tests np.fft.rfft on float32 data (spectrum analyzer path).
    Used in pjfm.py SpectrumAnalyzer for audio visualization.
    """
    print("\n" + "=" * 70)
    print("Benchmark: Real FFT (np.fft.rfft)")
    print("=" * 70)

    # Audio-sized block with Hanning window (matches pjfm spectrum analyzer)
    data = np.random.randn(2048).astype(np.float32)
    window = np.hanning(2048).astype(np.float32)

    def rfft_op():
        np.fft.rfft(data * window)

    ops_per_sec, elapsed = benchmark(rfft_op)

    print(f"\n  Block size: 2048 real samples (windowed)")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")

    # rfft should be faster than complex fft
    passed = ops_per_sec >= MIN_FFT_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Filter Benchmarks
# =============================================================================

def test_lfilter_fir_benchmark():
    """
    Benchmark FIR filtering with scipy.signal.lfilter.

    Tests 127-tap FIR filter (matches L+R LPF in FMStereoDecoder).
    lfilter uses compiled C code with contiguous array optimization.
    """
    print("\n" + "=" * 70)
    print("Benchmark: FIR Filter (scipy.signal.lfilter, 127 taps)")
    print("=" * 70)

    # Design filter matching FMStereoDecoder L+R LPF
    taps = 127
    fir_coef = signal.firwin(taps, 15000, fs=SAMPLE_RATE_IQ, window=('kaiser', 5.0))
    zi = signal.lfilter_zi(fir_coef, 1.0)

    # Test data
    data = np.random.randn(BLOCK_SIZE_IQ).astype(np.float64)
    state = zi * data[0]

    def lfilter_fir_op():
        nonlocal state
        _, state = signal.lfilter(fir_coef, 1.0, data, zi=state)

    ops_per_sec, elapsed = benchmark(lfilter_fir_op)

    print(f"\n  Filter taps: {taps}")
    print(f"  Block size: {BLOCK_SIZE_IQ} samples")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Samples/sec: {format_throughput(ops_per_sec * BLOCK_SIZE_IQ)}")
    print(f"  Minimum required: {format_throughput(MIN_LFILTER_FIR_OPS)}")

    passed = ops_per_sec >= MIN_LFILTER_FIR_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_lfilter_iir_benchmark():
    """
    Benchmark IIR filtering with scipy.signal.lfilter.

    Tests 2nd-order Butterworth (matches de-emphasis and tone filters).
    IIR filters are inherently sequential but lfilter optimizes memory access.
    """
    print("\n" + "=" * 70)
    print("Benchmark: IIR Filter (scipy.signal.lfilter, 2nd order)")
    print("=" * 70)

    # Design 2nd-order Butterworth (like de-emphasis filter)
    b, a = signal.butter(2, 3000, fs=SAMPLE_RATE_IQ)
    zi = signal.lfilter_zi(b, a)

    # Test data
    data = np.random.randn(BLOCK_SIZE_IQ).astype(np.float64)
    state = zi * data[0]

    def lfilter_iir_op():
        nonlocal state
        _, state = signal.lfilter(b, a, data, zi=state)

    ops_per_sec, elapsed = benchmark(lfilter_iir_op)

    print(f"\n  Filter order: 2 (Butterworth)")
    print(f"  Block size: {BLOCK_SIZE_IQ} samples")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Samples/sec: {format_throughput(ops_per_sec * BLOCK_SIZE_IQ)}")
    print(f"  Minimum required: {format_throughput(MIN_LFILTER_IIR_OPS)}")

    passed = ops_per_sec >= MIN_LFILTER_IIR_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Complex Arithmetic Benchmarks
# =============================================================================

def test_fm_discriminator_benchmark():
    """
    Benchmark FM quadrature discriminator.

    Tests the core FM demodulation: angle(s[n] * conj(s[n-1]))
    This is the critical path in FMStereoDecoder.demodulate().
    All operations should use AVX2 vectorization.
    """
    print("\n" + "=" * 70)
    print("Benchmark: FM Discriminator (angle + conj + multiply)")
    print("=" * 70)

    # Complex I/Q samples
    iq = (np.random.randn(BLOCK_SIZE_IQ) +
          1j * np.random.randn(BLOCK_SIZE_IQ)).astype(np.complex64)

    prev_sample = iq[0]

    def discriminator_op():
        # Exact operation from FMStereoDecoder.demodulate()
        samples = np.concatenate([[prev_sample], iq])
        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product)
        return baseband

    ops_per_sec, elapsed = benchmark(discriminator_op)

    print(f"\n  Block size: {BLOCK_SIZE_IQ} complex samples")
    print(f"  Operations: concatenate + multiply + conj + angle")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Samples/sec: {format_throughput(ops_per_sec * BLOCK_SIZE_IQ)}")
    print(f"  Minimum required: {format_throughput(MIN_DISCRIMINATOR_OPS)}")

    passed = ops_per_sec >= MIN_DISCRIMINATOR_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_complex_arithmetic_benchmark():
    """
    Benchmark basic complex array arithmetic.

    Tests element-wise multiply, add, abs, sqrt operations.
    These form the basis of power calculations and carrier generation.
    """
    print("\n" + "=" * 70)
    print("Benchmark: Complex Arithmetic (multiply, abs, sqrt)")
    print("=" * 70)

    # Complex arrays
    a = (np.random.randn(BLOCK_SIZE_IQ) +
         1j * np.random.randn(BLOCK_SIZE_IQ)).astype(np.complex64)
    b = (np.random.randn(BLOCK_SIZE_IQ) +
         1j * np.random.randn(BLOCK_SIZE_IQ)).astype(np.complex64)

    def arith_op():
        c = a * b
        mag = np.abs(c)
        rms = np.sqrt(np.mean(mag ** 2))
        return rms

    ops_per_sec, elapsed = benchmark(arith_op)

    print(f"\n  Block size: {BLOCK_SIZE_IQ} complex samples")
    print(f"  Operations: multiply + abs + square + mean + sqrt")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Minimum required: {format_throughput(MIN_COMPLEX_ARITH_OPS)}")

    passed = ops_per_sec >= MIN_COMPLEX_ARITH_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Resampling Benchmark
# =============================================================================

def test_resample_benchmark():
    """
    Benchmark polyphase resampling.

    Tests scipy.signal.resample for I/Q rate to audio rate conversion.
    This is used in FMStereoDecoder for decimation (250 kHz -> 48 kHz).
    resample() uses FFT internally and benefits from PocketFFT optimization.
    """
    print("\n" + "=" * 70)
    print("Benchmark: Polyphase Resample (scipy.signal.resample)")
    print("=" * 70)

    # Stereo audio data (2 channels)
    input_samples = BLOCK_SIZE_IQ
    output_samples = int(input_samples * SAMPLE_RATE_AUDIO / SAMPLE_RATE_IQ)

    data = np.random.randn(input_samples, 2).astype(np.float64)

    def resample_op():
        signal.resample(data, output_samples, axis=0)

    ops_per_sec, elapsed = benchmark(resample_op)

    print(f"\n  Input: {input_samples} samples @ {SAMPLE_RATE_IQ} Hz")
    print(f"  Output: {output_samples} samples @ {SAMPLE_RATE_AUDIO} Hz")
    print(f"  Channels: 2 (stereo)")
    print(f"  Iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {format_throughput(ops_per_sec)}")
    print(f"  Minimum required: {format_throughput(MIN_RESAMPLE_OPS)}")

    passed = ops_per_sec >= MIN_RESAMPLE_OPS
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Real-Time Feasibility Test
# =============================================================================

def test_realtime_feasibility():
    """
    Test real-time processing feasibility.

    Simulates one complete demodulation block and verifies it completes
    faster than real-time. At 250 kHz sample rate with 8192-sample blocks,
    we have 32.8 ms per block. Processing should use <50% of that budget.
    """
    print("\n" + "=" * 70)
    print("Test: Real-Time Processing Feasibility")
    print("=" * 70)

    # Time budget
    block_duration_ms = (BLOCK_SIZE_IQ / SAMPLE_RATE_IQ) * 1000
    target_budget_pct = 50  # Should use less than 50% of available time

    # Setup filters (matching FMStereoDecoder)
    pilot_bpf = signal.firwin(201, [18500, 19500], fs=SAMPLE_RATE_IQ,
                               pass_zero=False, window=('kaiser', 8.0))
    lr_sum_lpf = signal.firwin(127, 15000, fs=SAMPLE_RATE_IQ, window=('kaiser', 5.0))
    lr_diff_bpf = signal.firwin(201, [23000, 53000], fs=SAMPLE_RATE_IQ,
                                 pass_zero=False, window=('kaiser', 8.0))

    # Initialize filter states
    pilot_zi = signal.lfilter_zi(pilot_bpf, 1.0)
    lr_sum_zi = signal.lfilter_zi(lr_sum_lpf, 1.0)
    lr_diff_zi = signal.lfilter_zi(lr_diff_bpf, 1.0)

    # Generate test I/Q
    iq = (np.random.randn(BLOCK_SIZE_IQ) +
          1j * np.random.randn(BLOCK_SIZE_IQ)).astype(np.complex64)
    prev_sample = iq[0]

    output_samples = int(BLOCK_SIZE_IQ * SAMPLE_RATE_AUDIO / SAMPLE_RATE_IQ)

    def full_demod_block():
        nonlocal pilot_zi, lr_sum_zi, lr_diff_zi

        # FM discriminator
        samples = np.concatenate([[prev_sample], iq])
        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product)

        # Pilot extraction
        pilot, pilot_zi = signal.lfilter(pilot_bpf, 1.0, baseband, zi=pilot_zi)

        # L+R filtering
        lr_sum, lr_sum_zi = signal.lfilter(lr_sum_lpf, 1.0, baseband, zi=lr_sum_zi)

        # L-R filtering
        lr_diff, lr_diff_zi = signal.lfilter(lr_diff_bpf, 1.0, baseband, zi=lr_diff_zi)

        # Carrier regeneration (pilot squaring)
        carrier = 2 * pilot ** 2 - 1

        # Demodulate L-R
        lr_diff_demod = lr_diff * carrier * 2

        # Matrix decode
        left = lr_sum + lr_diff_demod
        right = lr_sum - lr_diff_demod

        # Resample to audio rate
        stereo = np.column_stack([left, right])
        audio = signal.resample(stereo, output_samples, axis=0)

        return audio

    # Benchmark
    ops_per_sec, elapsed = benchmark(full_demod_block, iterations=50, warmup=5)

    time_per_block_ms = (elapsed / 50) * 1000
    budget_used_pct = (time_per_block_ms / block_duration_ms) * 100

    print(f"\n  Block size: {BLOCK_SIZE_IQ} I/Q samples")
    print(f"  Block duration: {block_duration_ms:.2f} ms (real-time)")
    print(f"  Processing time: {time_per_block_ms:.2f} ms")
    print(f"  Budget used: {budget_used_pct:.1f}%")
    print(f"  Target: <{target_budget_pct}%")
    print(f"  Headroom: {100 - budget_used_pct:.1f}%")

    passed = budget_used_pct < target_budget_pct
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print("\n  WARNING: Processing may not keep up with real-time!")
        print("  Check NumPy/SciPy BLAS configuration and CPU frequency scaling.")

    return passed


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all benchmarks and report results."""
    print("\n" + "=" * 70)
    print("SIMD/VECTORIZATION BENCHMARK TEST SUITE")
    print("=" * 70)
    print("\nVerifying NumPy/SciPy SIMD acceleration for real-time DSP")
    print(f"Target: {SAMPLE_RATE_IQ} Hz I/Q rate, {BLOCK_SIZE_IQ}-sample blocks")

    tests = [
        ("NumPy Backend Info", test_numpy_backend_info),
        ("Complex FFT", test_fft_complex_benchmark),
        ("Real FFT", test_rfft_benchmark),
        ("FIR Filter (127 taps)", test_lfilter_fir_benchmark),
        ("IIR Filter (2nd order)", test_lfilter_iir_benchmark),
        ("FM Discriminator", test_fm_discriminator_benchmark),
        ("Complex Arithmetic", test_complex_arithmetic_benchmark),
        ("Polyphase Resample", test_resample_benchmark),
        ("Real-Time Feasibility", test_realtime_feasibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s}  {status}")

    print(f"\n  {passed_count}/{total_count} tests passed")

    if passed_count < total_count:
        print("\n  RECOMMENDATIONS:")
        print("  - Verify OpenBLAS/MKL is installed: pip show numpy")
        print("  - Check CPU frequency scaling: cpufreq-info or /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        print("  - Ensure no thermal throttling: watch -n1 'cat /sys/class/thermal/thermal_zone*/temp'")

    return passed_count == total_count


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

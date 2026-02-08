# FM Stereo Decoder Test Suite

## Overview

`test_pjfm.py` is a comprehensive test suite for the FM stereo decoder in `demodulator.py`. It verifies mathematically correct mono and stereo FM decoding with textbook accuracy and documents phase/delay relationships throughout the decode chain.

## Running the Tests

```bash
# Run all tests with detailed output
python test_pjfm.py

# Run with pytest (if installed)
pytest test_pjfm.py -v
```

## Test Results Summary

**10/10 tests pass** (as of 2026-02-02)

| Test | Result | Notes |
|------|--------|-------|
| FM Demodulation Accuracy | PASS | Correlation 1.000000, 0% amplitude error |
| Audio SNR (Clean) | PASS | 35.3 dB (target: >35 dB) |
| THD+N | PASS | -35.3 dB / 1.72% (target: <-35 dB) |
| Mono Decoding | PASS | L/R correlation 1.000000 |
| Stereo Decoding | PASS | 70-73 dB separation |
| Stereo Separation | PASS | 67-100 dB across 100 Hz - 12 kHz |
| Subcarrier Phase Sensitivity | PASS | Correct phase behavior verified |
| Group Delay Alignment | PASS | 0 samples L/R difference |
| Frequency Response | PASS | Flat ±0.00 dB below 1 kHz |
| SNR with Noise | PASS | Graceful degradation verified |

## Validation Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| FM demod correlation | > 0.999 | 1.000000 |
| Audio SNR (clean) | > 35 dB | 35.3 dB |
| THD+N | < -35 dB | -35.3 dB (1.72%) |
| Stereo separation | > 30 dB | 67-100 dB |
| L/R timing | < 5 samples | 0 samples |
| Frequency response | ±3 dB (<1 kHz) | ±0.00 dB |

## Testing Methodology

### Synthetic Signal Generation

Tests use mathematically correct synthetic FM stereo signals to enable precise accuracy measurements without real-world RF variables:

```python
# FM Multiplex generation (per FM broadcast standard)
lr_sum = (left + right) / 2          # Mono-compatible sum
lr_diff = (left - right) / 2          # Stereo difference
pilot = 0.09 * sin(2*pi*19000*t)      # 19 kHz pilot at 9%
carrier_38k = -cos(2*pi*38000*t)      # DSB-SC subcarrier
lr_diff_mod = lr_diff * carrier_38k   # Modulated difference
multiplex = lr_sum*0.9 + pilot + lr_diff_mod*0.9

# FM Modulation (75 kHz deviation for broadcast FM)
phase = 2*pi*75000 * cumsum(multiplex) * dt
iq = cos(phase) + j*sin(phase)
```

### Demodulation Method

**Quadrature Discriminator** (`FMStereoDecoder`):
```python
# Instantaneous frequency from phase difference
product = samples[n] * conj(samples[n-1])
baseband = angle(product) * (sample_rate / (2*pi*deviation))
```

This is the standard FM demodulation technique that extracts instantaneous frequency from the phase difference between consecutive samples.

### Measurement Techniques

**Goertzel Algorithm**: Single-frequency power measurement for stereo separation tests. More efficient than FFT for measuring specific frequencies.

**SNR Measurement**: FFT-based with Hanning window. Signal power measured in narrow band around test frequency, noise power from remainder of spectrum.

**THD+N**: Measures fundamental power vs. harmonics + noise floor. Accounts for harmonics up to Nyquist.

**Step Response**: Used for group delay alignment measurement. Measures 50% crossing point on each channel.

## Signal Flow Documentation

### FM Stereo Multiplex Structure

```
0-15 kHz:   L+R (mono compatible)
19 kHz:     Pilot tone (9% of max deviation)
23-53 kHz:  L-R on 38 kHz DSB-SC carrier
57 kHz:     RDS subcarrier (not tested here)
```

### Decoder Signal Flow

```
IQ samples (250 kHz)
    |
    v
FM Demodulation (quadrature discriminator)
    |
    +---> Pilot BPF (201 taps, 18.5-19.5 kHz)
    |         |
    |         v
    |     Pilot-squaring: 2*sin^2(x)-1 = -cos(2x)
    |         |
    |         v
    |     38 kHz carrier regeneration
    |
    +---> L+R LPF (127 taps, 15 kHz) + 100 sample delay buffer
    |         |
    |         v
    |     L+R signal (163 samples total delay)
    |
    +---> L-R BPF (201 taps, 23-53 kHz) --> x carrier x 2 --> L-R LPF
              |
              v
          L-R signal (163 samples total delay)

Matrix Decode:
    L = L+R + L-R
    R = L+R - L-R

Post-processing:
    - Resample to 48 kHz
    - De-emphasis (75 us)
    - Tone controls (optional bass/treble boost)
    - Soft limiter (tanh)
```

### Group Delay Alignment

Both L+R and L-R paths are carefully aligned to 163 samples total delay at IQ rate:

| Path | Components | Delay |
|------|------------|-------|
| L+R | LPF (63 samples) + compensation buffer (100 samples) | 163 samples |
| L-R | BPF (100 samples) + LPF (63 samples) | 163 samples |

This alignment is critical for stereo separation at high audio frequencies. Misalignment causes frequency-dependent L/R crosstalk.

## Test Descriptions

### test_fm_demod_accuracy
Verifies the CPU quadrature discriminator correctly recovers FM baseband.
- Generates known 1 kHz baseband, FM modulates to IQ, demodulates
- Compares demodulated output to original input
- **Target**: Correlation > 0.999, amplitude error < 1%
- **Result**: Correlation 1.000000, 0% error

### test_audio_snr
Measures signal-to-noise ratio of decoded audio with clean synthetic input.
- Full stereo processing chain including resampling and limiting
- 1 kHz stereo test tone
- **Target**: > 35 dB
- **Result**: 35.3 dB

### test_thd_n
Measures Total Harmonic Distortion + Noise.
- Pure 1 kHz tone through full processing chain
- Measures fundamental vs. harmonics + noise floor
- **Target**: < -35 dB (< 2% distortion)
- **Result**: -35.3 dB (1.72%)

### test_mono_decode
Verifies mono decoding when no pilot tone is present.
- Generates FM signal without 19 kHz pilot
- L and R channels should be identical
- Pilot detection should report false
- **Target**: L/R correlation > 0.999
- **Result**: Correlation 1.000000

### test_stereo_decode
Tests stereo separation with different tones on L and R channels.
- 1 kHz on left channel only
- 2 kHz on right channel only
- Measures crosstalk between channels
- **Target**: > 20 dB channel separation
- **Result**: 73 dB (left), 71 dB (right)

### test_stereo_separation
Tests stereo separation across the audio frequency range.
- Left-only signal at 100 Hz, 1 kHz, 5 kHz, 10 kHz, 12 kHz
- Measures power ratio between channels at each frequency
- **Target**: > 30 dB at all frequencies
- **Result**: 67-100 dB (exceeds target by 37+ dB)

### test_subcarrier_phase_sensitivity
Documents decoder sensitivity to transmitter 38 kHz subcarrier phase.

The pilot-squaring method produces a specific carrier phase:
```
Pilot: sin(wt)
Squared: 2*sin^2(wt) - 1 = -cos(2wt)
```

| TX Subcarrier | RX Carrier | Result |
|---------------|------------|--------|
| `-cos(2wt)` | `-cos(2wt)` | CORRECT - proper stereo |
| `cos(2wt)` | `-cos(2wt)` | INVERTED - L/R swapped |
| `sin(2wt)` | `-cos(2wt)` | FAILED - 90 deg phase error |

**Result**: Phase behavior matches theoretical prediction

### test_group_delay_alignment
Verifies L/R timing alignment using step response measurement.
- Generates step function in both channels
- Measures 50% crossing time for each channel
- **Target**: Difference < 5 samples at 48 kHz
- **Result**: 0.00 samples difference

### test_frequency_response
Measures frequency response from 100 Hz to 14 kHz.
- Mono signal at each test frequency
- Normalized to 1 kHz reference
- **Target**: ±3 dB below 1 kHz
- **Result**: ±0.00 dB (flat response below 1 kHz)

| Frequency | Response |
|-----------|----------|
| 100 Hz | +0.00 dB |
| 200 Hz | +0.00 dB |
| 500 Hz | -0.00 dB |
| 1000 Hz | 0.00 dB (ref) |
| 2000 Hz | -0.01 dB |
| 5000 Hz | -0.05 dB |
| 8000 Hz | -0.13 dB |
| 10000 Hz | -0.21 dB |
| 12000 Hz | -0.44 dB |
| 14000 Hz | -2.95 dB |

### test_snr_with_noise
Tests decoder behavior with AWGN-corrupted input.
- Tests at 40, 30, 20, 10 dB input SNR
- Verifies graceful degradation
- Checks stereo blend engages at low SNR

| Input SNR | Output SNR | Pilot | Blend |
|-----------|------------|-------|-------|
| 40 dB | 35.3 dB | Yes | 0.10 |
| 30 dB | 35.2 dB | Yes | 0.10 |
| 20 dB | 34.5 dB | Yes | 0.10 |
| 10 dB | 30.1 dB | Yes | 0.02 |

**Result**: Decoder maintains >30 dB output SNR even with 10 dB input

## Implementation Notes

### Carrier Regeneration
The decoder uses pilot-squaring for 38 kHz carrier regeneration:
```
carrier = 2 * (pilot_normalized)^2 - 1
```
This exploits the trig identity `2*sin^2(x) - 1 = -cos(2x)` to double the pilot frequency without a PLL.

### Filter Design
All FIR filters use Kaiser windows for steep rolloff:
- Pilot BPF: 201 taps, beta=7.0 (~70 dB stopband)
- L+R/L-R LPF: 127 taps, beta=6.0 (~60 dB stopband)
- L-R BPF: 201 taps, beta=7.0 (~70 dB stopband)

### Stereo Blend
At low SNR, the decoder blends toward mono to reduce noise:
- Below 8 dB pilot SNR: Full mono
- Above 20 dB pilot SNR: Full stereo
- Linear blend between thresholds

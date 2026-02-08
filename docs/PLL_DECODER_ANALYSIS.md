# PLL Stereo Decoder Architecture Analysis

## Executive Summary

The PLLStereoDecoder achieves **excellent low-frequency performance** (50 dB separation at 1 kHz) but suffers **severe high-frequency degradation** (12 dB at 5 kHz, negative at 15 kHz). The primary issue is **frequency-dependent group delay mismatch** between the L+R and L-R signal paths.

## Current Performance

| Metric | Low Freq (1 kHz) | Mid Freq (5 kHz) | High Freq (15 kHz) |
|--------|------------------|------------------|--------------------|
| **Separation** | 50 dB (excellent) | 12 dB (poor) | -3 dB (inverted!) |
| **SNR Tracking** | ✅ Accurate | ✅ Accurate | ✅ Accurate |
| **Phase Noise** | ✅ Robust | ✅ Robust | ✅ Robust |

## Architecture Overview

### PLL Design (Second-Order Type 2)
```
Pilot (19 kHz) → BPF → Phase Detector → Loop Filter → NCO → 38 kHz Carrier
                           ↑                                      |
                           └──────────────────────────────────────┘
```

**Parameters:**
- Loop bandwidth (Bn): 30 Hz
- Damping (ζ): 0.707 (critically damped)
- Phase error filter: 5 kHz LPF (1st order IIR)
- Lock detection: EMA of squared phase error

**Strengths:**
- ✅ Narrow bandwidth rejects pilot amplitude noise
- ✅ Phase-coherent carrier tracking
- ✅ Explicit lock detection with hysteresis
- ✅ Filters 38 kHz component from phase error

## Critical Issue: High-Frequency Separation Degradation

### Problem Analysis

At 5 kHz audio, separation drops from 50 dB → 12 dB (**38 dB loss!**)

**Root Cause:** Group delay mismatch between L+R and L-R paths.

#### Calculation:
For 12 dB separation at 5 kHz:
- Requires phase error φ = 28°
- Time delay: Δt = φ/(2πf) = 15.5 μs
- **At 480 kHz: 7.4 samples of mismatch**

Current compensation: **Fixed 100-sample delay** on L+R path.

### Why Fixed Delay Fails

**L-R Signal Path:**
1. Baseband → **23-53 kHz BPF (201 taps)** → L-R modulated signal
2. Multiply by 38 kHz carrier → Demodulated L-R
3. **15 kHz LPF (127 taps)** → Clean L-R

**L+R Signal Path:**
1. Baseband → **15 kHz LPF (127 taps)** → L+R
2. **Fixed 100-sample delay** → Aligned L+R

**The Issue:**
- Bandpass filters have **frequency-dependent group delay**
- Peak delay at center frequency (38 kHz)
- **Reduced delay at band edges** (23 kHz, 53 kHz)
- At 15 kHz audio → DSB sidebands at 23 and 53 kHz (band edges!)
- Group delay varies by ~7-8 samples across passband
- Fixed compensation cannot track this variation

## Improvement Opportunities

### 🔴 PRIORITY 1: High-Frequency Separation (CRITICAL)

**Impact:** Would improve 5 kHz separation from 12 dB → 40+ dB

#### Option 1A: All-Pass Delay Equalizer (Recommended)
Design an all-pass filter that matches the L-R BPF's frequency-dependent group delay.

```python
# Measure L-R path group delay vs. frequency
# Design complementary all-pass for L+R path
lr_diff_bpf_delay = measure_group_delay(self.lr_diff_bpf, freqs)
allpass_coeffs = design_allpass_equalizer(lr_diff_bpf_delay, freqs)
```

**Pros:**
- Precise frequency-dependent compensation
- No amplitude distortion
- Can be pre-computed offline

**Cons:**
- More complex implementation
- Adds computational cost

**Estimated Improvement:** +25 dB at 5 kHz, +30 dB at 10 kHz

#### Option 1B: Wider L-R BPF with Flatter Group Delay
Widen BPF passband to 20-56 kHz to move critical frequencies away from band edges.

```python
# Current: 23-53 kHz (tight around 38 kHz ± 15 kHz)
# Proposed: 20-56 kHz (more margin)
lr_diff_low = 20000 / nyq
lr_diff_high = min(56000 / nyq, 0.95)
```

**Pros:**
- Simple to implement
- Flatter group delay in audio band
- No extra computation

**Cons:**
- More noise passes through
- Requires higher sample rate for margin

**Estimated Improvement:** +15 dB at 5 kHz, +20 dB at 10 kHz

#### Option 1C: Higher Sample Rate
Operate at 960 kHz instead of 480 kHz.

**Pros:**
- Moves filters further from Nyquist (better performance)
- Reduces quantization effects
- More samples for delay matching

**Cons:**
- 2× computational cost
- Requires 960 kHz I/Q input

**Estimated Improvement:** +10 dB at 10 kHz, +20 dB at 15 kHz

---

### 🟡 PRIORITY 2: Phase Error Filter Enhancement

**Current:** 1st-order IIR at 5 kHz provides only **-18 dB rejection** at 38 kHz.

**Issue:** Some 38 kHz ripple leaks through, creating small integrator bias.

#### Solution: 2nd-Order Butterworth Lowpass
```python
# Current: Single-pole IIR
self._pll_pe_alpha = 2 * np.pi * 5000 / (iq_sample_rate + 2 * np.pi * 5000)

# Proposed: 2nd-order Butterworth
from scipy.signal import butter, lfilter_zi
b, a = butter(2, 5000 / (iq_sample_rate / 2), 'low')
self._pll_pe_b = b
self._pll_pe_a = a
self._pll_pe_state = lfilter_zi(b, a)
```

**Benefits:**
- **-36 dB rejection** at 38 kHz (2× better)
- Negligible additional phase lag at 30 Hz loop bandwidth
- Cleaner phase error signal → tighter lock

**Estimated Improvement:** +2-3 dB separation across all frequencies

---

### 🟡 PRIORITY 3: Adaptive Loop Bandwidth

**Current:** Fixed 30 Hz bandwidth (conservative).

**Issue:** Slow acquisition after frequency changes or brief signal loss.

#### Solution: Three-State Bandwidth Adaptation
```python
# State 1: ACQUISITION (PLL unlocked)
Bn = 100 Hz  # Wide bandwidth for fast pull-in

# State 2: TRACKING (PLL locked, SNR > 20 dB)
Bn = 30 Hz   # Current nominal bandwidth

# State 3: PRECISION (PLL locked, SNR > 35 dB)
Bn = 10 Hz   # Ultra-narrow for maximum noise rejection
```

**Implementation:**
```python
if not self._pll_locked:
    # Acquisition mode
    omega_n = 2 * np.pi * 100 / (0.707 + 1 / (4 * 0.707))
elif self._snr_db > 35:
    # Precision mode
    omega_n = 2 * np.pi * 10 / (0.707 + 1 / (4 * 0.707))
else:
    # Tracking mode (current)
    omega_n = 2 * np.pi * 30 / (0.707 + 1 / (4 * 0.707))

# Recompute gains
self._pll_Kp = 2 * 0.707 * omega_n / iq_sample_rate
self._pll_Ki = (omega_n ** 2) / (iq_sample_rate ** 2)
```

**Benefits:**
- Faster tune/seek (3× faster acquisition)
- Better noise rejection in precision mode (+5 dB effective SNR)
- No downside (adapts to conditions)

**Estimated Improvement:**
- Tune time: 0.3s → 0.1s
- SNR at high RF SNR: +3 dB

---

### 🟢 PRIORITY 4: L-R Gain Calibration Refinement

**Current:** Fixed empirical gain of 1.0029.

**Issue:** Single gain value may not be optimal across all sample rates and conditions.

#### Solution: Frequency-Dependent Calibration
```python
# Measure L+R and L-R path gains at multiple frequencies
# Apply frequency-dependent correction via EQ filter
calib_freqs = [100, 1000, 5000, 10000, 15000]  # Hz
calib_gains = measure_path_gains(calib_freqs)
self.lr_diff_eq = design_eq_filter(calib_freqs, calib_gains, fs_audio)
```

**Alternative:** Auto-calibration using pilot amplitude reference.

**Estimated Improvement:** +1-2 dB separation (minor refinement)

---

### 🟢 PRIORITY 5: Enhanced Lock Detection

**Current:** Squared phase error EMA with fixed thresholds.

**Issues:**
- Not robust to rapid pilot amplitude changes
- No frequency error indication

#### Solution: Multi-Metric Lock Detector
```python
# Metric 1: Phase error magnitude (existing)
phase_error_metric = self._pll_pe_avg < self._pll_lock_threshold

# Metric 2: Frequency error magnitude (new)
freq_error = abs(self._pll_integrator)
freq_error_metric = freq_error < 0.1  # < 0.1 rad/sample error

# Metric 3: Pilot correlation (new)
correlation = sqrt(self._pll_i_lp**2 + self._pll_q_lp**2)
correlation_metric = correlation > 0.5  # Strong correlation

# Combined lock decision
self._pll_locked = (phase_error_metric and
                    freq_error_metric and
                    correlation_metric)
```

**Benefits:**
- More reliable lock indication
- Faster unlock detection on signal loss
- Diagnostic info for debugging

**Estimated Improvement:** Better stereo engage/disengage behavior

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Widen L-R BPF** (Option 1B) - Immediate +15 dB at 5 kHz
2. ✅ **2nd-order phase error filter** - Clean up PLL tracking (+2-3 dB)

### Phase 2: Major Improvements (1 day)
3. ✅ **All-pass delay equalizer** (Option 1A) - Fix high-frequency separation (+25 dB)
4. ✅ **Adaptive loop bandwidth** - Faster acquisition, better noise rejection

### Phase 3: Refinements (2-3 hours)
5. ✅ **Enhanced lock detector** - More robust stereo switching
6. ✅ **Frequency-dependent gain calibration** - Final polish

### Phase 4: Future (requires infrastructure)
7. 🔄 **Higher sample rate support** (960 kHz) - Requires BB60D/R8600 config

---

## Performance Projections

### After Phase 1 (Quick Wins):
| Freq | Current | Projected | Improvement |
|------|---------|-----------|-------------|
| 1 kHz | 50 dB | 52 dB | +2 dB |
| 5 kHz | 12 dB | 29 dB | **+17 dB** |
| 10 kHz | 10 dB | 32 dB | **+22 dB** |
| 15 kHz | -3 dB | 15 dB | **+18 dB** |

### After Phase 2 (Major Improvements):
| Freq | Current | Projected | Improvement |
|------|---------|-----------|-------------|
| 1 kHz | 50 dB | 54 dB | +4 dB |
| 5 kHz | 12 dB | **45 dB** | **+33 dB** |
| 10 kHz | 10 dB | **40 dB** | **+30 dB** |
| 15 kHz | -3 dB | **30 dB** | **+33 dB** |

This would bring the decoder to **professional broadcast-grade performance** across the entire audio spectrum.

---

## Comparison to Pilot-Squaring Decoder

**Why PLL is fundamentally better:**

| Aspect | Pilot-Squaring | PLL |
|--------|----------------|-----|
| **Pilot noise** | Squares noise (2× penalty) | Filters noise (30 Hz BW) |
| **Carrier phase** | Fixed by pilot^2 identity | Tracks with zero steady-state error |
| **Phase coherence** | Dependent on pilot amplitude | Independent of pilot level |
| **Best case separation** | ~25 dB (pilot AM noise limit) | **50 dB** (limited by delay matching) |

The PLL's phase-coherent carrier regeneration is the key advantage. Once group delay matching is fixed, it should achieve 40-50 dB separation across 100 Hz - 15 kHz.

---

## References

- Current test results: `test-snr-stereo-opus.py`
- Phase noise analysis: `test-phase-noise-debug.py`
- Simplified bench: `test-sep-simple.py`

## Implementation Notes

All proposed changes are **backward-compatible** and can be enabled/disabled via config flags for A/B testing.

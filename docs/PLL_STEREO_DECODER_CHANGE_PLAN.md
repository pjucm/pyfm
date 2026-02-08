# PLL Stereo Decoder Change Plan

## Goal
Improve `PLLStereoDecoder` in three areas without destabilizing core decode behavior:

1. Stereo separation (especially high audio frequencies near 15 kHz)
2. Phase-noise robustness
3. Stereo quality gating (SNR/lock decisions)

This plan is staged so each change can be validated independently and reverted safely.

## Non-Goals
- No major refactor of the full FM demod pipeline in one step
- No dependency changes outside existing NumPy/SciPy stack
- No immediate behavior changes for unrelated decoders

## Baseline (Before Changes)
Record baseline metrics with the existing bench scripts and fixed seeds:

- 1 kHz L-only separation at RF SNR 15/20/30/45 dB
- 15 kHz L-only separation at RF SNR 15/20/30/45 dB
- Phase-noise sweep for:
  - RF carrier jitter mode
  - TX common pilot/subcarrier jitter mode
- Stereo lock/blend behavior with:
  - weak pilot
  - no pilot
  - low IQ sample rates (192 kHz, 160 kHz)

Store results in a dated markdown report per run.

## Phase 1 (Highest Priority): Fix High-Frequency Stereo Separation
### Problem
Current L-R extraction bandpass (`23–53 kHz`) clips stereo sideband edges (`38 +/- 15 kHz`), hurting separation near 15 kHz.

### Change
Implement one of these options:

1. Minimal-risk: widen L-R BPF to approximately `21–55 kHz` (or `20–56 kHz` after sweep).
2. Preferred architecture: shift to complex 38 kHz mixing (I/Q demod) + lowpass, reducing sensitivity to BPF edge placement.

### Acceptance Criteria
- 15 kHz separation improves by at least 20 dB versus baseline at RF SNR >= 20 dB.
- 1 kHz separation does not degrade by more than 1 dB.
- No regressions in lock stability or CPU budget > 15%.

### Risks
- Wider BPF may admit extra noise/interference.
- Mitigation: compare multiple candidate passbands and keep a conservative default.

## Phase 2: Make PLL Lock Detection Amplitude-Aware
### Problem
Current lock detector can report `pll_locked=True` with very weak/no pilot because phase-error magnitude is not normalized by pilot amplitude.

### Change
Gate lock on both:

1. normalized phase quality (existing low-passed pilot I/Q metric),
2. pilot amplitude floor (derived from same I/Q envelope).

Use hysteresis for both lock and unlock thresholds.

### Acceptance Criteria
- With pilot removed, decoder must report unlocked after settling.
- With nominal pilot, lock acquisition and hold remain stable.
- False-lock rate in weak/no-pilot test cases drops to near zero.

### Risks
- Overly strict threshold may delay lock on weak stations.
- Mitigation: tune with sweeps over pilot amplitudes and RF SNR.

## Phase 3: Redesign Stereo Quality Metric (SNR/Gating)
### Problem
Current `90–100 kHz` noise-band metric can be biased by out-of-band content and is unavailable at lower IQ rates, forcing mono unexpectedly.

### Change
Add a composite stereo-quality estimator:

1. pilot-to-nearband-noise estimate around pilot sidebands (rate-safe),
2. PLL phase-quality term (from normalized phase error),
3. fallback path when high-frequency noise band is unavailable.

Use this quality metric for stereo gating/blend decisions.

### Acceptance Criteria
- At 192 kHz IQ, stereo decode can engage when pilot lock/quality is good.
- External 90–100 kHz tones no longer collapse blend unless true quality drops.
- Blend transitions remain smooth and artifact-free.

### Risks
- Re-tuning blend thresholds required.
- Mitigation: keep old metric behind feature flag during validation.

## Phase 4: Adaptive PLL Loop Bandwidth
### Problem
Fixed ~30 Hz loop bandwidth is not optimal across all jitter/noise regimes.

### Change
Implement two-state or continuous adaptation:

- Narrow BW in steady state (noise rejection)
- Wider BW during acquisition/high phase error (tracking)

Clamp gains and add dwell timers to avoid oscillation.

### Acceptance Criteria
- Under TX common phase jitter, separation improves versus fixed 30 Hz.
- Under clean conditions, separation/SNR do not regress measurably.
- No lock-chatter under edge cases.

### Risks
- Loop instability if adaptation is too aggressive.
- Mitigation: bounded gain schedule + extensive seeded sweeps.

## Phase 5: Calibration and Defaults
### Change
- Revisit fixed `lr_diff_gain` calibration after filter/loop updates.
- Tune default blend thresholds using new quality metric.
- Add versioned tuning constants to ease future retuning.

### Acceptance Criteria
- Gains and thresholds are reproducible from documented bench scripts.
- Defaults perform well across representative RF SNR and sample rates.

## Test and Validation Matrix
Run seeded sweeps for:

- RF SNR: 10, 15, 20, 25, 30, 35, 40 dB
- Audio tones: 100, 1000, 5000, 10000, 15000 Hz
- IQ rates: 160k, 192k, 200k, 250k, 480k
- Phase noise:
  - RF jitter mode: 0.0, 0.01, 0.05, 0.1, 0.2 rad RMS
  - TX common mode: 0.0, 0.01, 0.05, 0.1, 0.2 rad RMS
- Pilot amplitude: 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.09

Collect:

- Separation (L-only and R-only)
- Decoder-reported quality/SNR
- Lock state, phase error, blend factor
- Processing-time profile when enabled

## Rollout Strategy
1. Ship Phase 1 behind a feature flag or constructor option.
2. Validate, then make Phase 1 default.
3. Ship Phase 2 + Phase 3 together (quality/lock coherence).
4. Ship Phase 4 adaptation with conservative defaults.
5. Final calibration pass (Phase 5) and remove temporary flags.

## Deliverables
- Code changes in `pll_stereo_decoder.py` (staged by phase)
- Updated or new deterministic bench tests
- One markdown report per phase with before/after metrics
- Final tuning summary with default constants and rationale

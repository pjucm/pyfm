#!/usr/bin/env python3
"""
RDS Decoder Alternative Mechanism Tests

Tests alternative demodulation and timing recovery approaches:
1. Early-late gate timing instead of Gardner TED
2. Root-raised-cosine matched filter
3. PLL-based carrier tracking (Costas loop)
4. Symbol-rate matched FIR filter
5. Different timing loop bandwidths
"""

import numpy as np
from scipy import signal
from datetime import date

from rds_decoder import (
    RDSDecoder,
    RDS_CARRIER_FREQ,
    RDS_SYMBOL_RATE,
    OFFSET_WORDS,
    PTY_NAMES,
    compute_syndrome,
)

PILOT_FREQ = 19000.0


def mjd_from_date(year, month, day):
    return (date(year, month, day) - date(1858, 11, 17)).days


def _crc_checkword(data16, offset_word):
    remainder = compute_syndrome((data16 & 0xFFFF) << 10)
    return (remainder ^ offset_word) & 0x3FF


def _build_block(data16, offset_word):
    check = _crc_checkword(data16, offset_word)
    return ((data16 & 0xFFFF) << 10) | check


def _block_to_bits(block26):
    return [(block26 >> i) & 1 for i in range(25, -1, -1)]


def _group_to_bits(block_a, block_b, block_c, block_d):
    blocks = [
        _build_block(block_a, OFFSET_WORDS['A']),
        _build_block(block_b, OFFSET_WORDS['B']),
        _build_block(block_c, OFFSET_WORDS['C']),
        _build_block(block_d, OFFSET_WORDS['D']),
    ]
    bits = []
    for block in blocks:
        bits.extend(_block_to_bits(block))
    return bits


def _encode_group_0a(pi_code, pty, segment, ps_name):
    segment &= 0x03
    ps_name = (ps_name[:8]).ljust(8)
    char1 = ord(ps_name[segment * 2])
    char2 = ord(ps_name[segment * 2 + 1])
    block_a = pi_code
    block_b = (0 << 12) | (0 << 11) | (pty << 5) | segment
    block_c = 0x0000
    block_d = (char1 << 8) | char2
    return _group_to_bits(block_a, block_b, block_c, block_d)


def _build_bitstream(pi_code, pty, ps_name, repeats=2):
    bits = []
    for _ in range(repeats):
        for segment in range(4):
            bits.extend(_encode_group_0a(pi_code, pty, segment, ps_name))
    return np.array(bits, dtype=np.uint8)


def _differential_encode(bits):
    symbols = np.zeros(len(bits), dtype=np.float64)
    prev = 1.0
    for i, bit in enumerate(bits):
        if bit == 1:
            prev = -prev
        symbols[i] = prev
    return symbols


def design_rrc_filter(samples_per_symbol, alpha=0.35, num_taps=101):
    """Design root-raised-cosine filter for RDS."""
    # Symbol-spaced taps
    t = np.arange(num_taps) - (num_taps - 1) / 2
    t = t / samples_per_symbol  # Normalize to symbol periods

    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - alpha + 4 * alpha / np.pi
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi / (4*alpha)) +
                (1 - 2/np.pi) * np.cos(np.pi / (4*alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti)**2)
            if abs(den) > 1e-10:
                h[i] = num / den
            else:
                h[i] = 0.0

    # Normalize
    h = h / np.sqrt(np.sum(h**2))
    return h


def _symbols_to_baseband(symbols, sample_rate, include_pilot=True,
                         pilot_amp=0.1, rds_amp=0.06, pad_s=0.05,
                         shape=True, use_rrc=False):
    """Convert differential symbols into RDS baseband samples."""
    sps = sample_rate / RDS_SYMBOL_RATE
    total_samples = int(np.ceil(len(symbols) * sps))
    t = np.arange(total_samples) / sample_rate

    symbol_idx = np.floor(t * RDS_SYMBOL_RATE).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    symbol_wave = symbols[symbol_idx]

    if shape:
        if use_rrc:
            # Use RRC filter for proper pulse shaping
            rrc = design_rrc_filter(sps, alpha=0.35, num_taps=int(8 * sps) | 1)
            symbol_wave = signal.lfilter(rrc, 1.0, symbol_wave)
        else:
            # Simple LPF shaping
            nyq = sample_rate / 2.0
            cutoff = 2400.0 / nyq
            taps = signal.firwin(101, cutoff, window=('kaiser', 7.0))
            symbol_wave = signal.lfilter(taps, 1.0, symbol_wave)

    rds = rds_amp * symbol_wave * np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    if include_pilot:
        pilot = pilot_amp * np.cos(2 * np.pi * PILOT_FREQ * t)
    else:
        pilot = 0.0

    baseband = rds + pilot

    if pad_s > 0:
        pad_samples = int(pad_s * sample_rate)
        baseband = np.concatenate([
            np.zeros(pad_samples),
            baseband,
            np.zeros(pad_samples),
        ])

    return baseband.astype(np.float64)


def _add_audio_tone(baseband, sample_rate, amplitude=0.7, tone_hz=1000.0):
    t = np.arange(len(baseband)) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * tone_hz * t)
    return baseband + audio


def _fm_modulate(baseband, sample_rate, deviation=75000.0):
    dt = 1.0 / sample_rate
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
    return np.cos(phase) + 1j * np.sin(phase)


def _fm_demod(iq, sample_rate, deviation=75000.0):
    product = iq[1:] * np.conj(iq[:-1])
    phase_diff = np.angle(product)
    baseband = phase_diff * sample_rate / (2 * np.pi * deviation)
    return np.concatenate([baseband, baseband[-1:]])


def _add_awgn(iq, snr_db, rng):
    signal_power = np.mean(np.abs(iq) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2.0)
    noise = rng.normal(scale=sigma, size=iq.shape) + 1j * rng.normal(scale=sigma, size=iq.shape)
    return iq + noise


class EarlyLateGateDecoder:
    """
    Alternative timing recovery using early-late gate.

    Simpler than Gardner TED, samples at early/on-time/late points
    and adjusts timing based on amplitude difference.
    """

    def __init__(self, sample_rate=250000, timing_gain=0.01, loop_bw=0.01):
        self.sample_rate = sample_rate
        self.sps = sample_rate / RDS_SYMBOL_RATE
        self.timing_mu = 0.5
        self.timing_gain = timing_gain
        self.loop_bw = loop_bw
        self.sample_buffer = []

        # Design filters
        nyq = sample_rate / 2.0

        # 57 kHz bandpass
        rds_bw = 2000
        low = (RDS_CARRIER_FREQ - rds_bw) / nyq
        high = (RDS_CARRIER_FREQ + rds_bw) / nyq
        self.bpf_b = signal.firwin(201, [max(0.01, low), min(0.99, high)],
                                    pass_zero=False, window=('kaiser', 7.0))
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, 1.0) * 0

        # Pilot bandpass (for coherent demod)
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf_b = signal.firwin(201, [pilot_low, pilot_high],
                                          pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_zi = signal.lfilter_zi(self.pilot_bpf_b, 1.0) * 0

        # Post-demod LPF
        self.lpf_b, self.lpf_a = signal.butter(2, 800 / nyq, btype='low')
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * 0

        # Delay buffer for differential decoding
        self.delay_samples = int(round(self.sps))
        self.delay_buf = np.zeros(self.delay_samples)

        # AGC
        self.agc_rms = 0.1

        # Stats
        self.bits_extracted = 0
        self.synced = False
        self.bit_buffer = []
        self.expected_block = 0
        self.blocks_ok = 0
        self.blocks_total = 0

    def process(self, baseband, use_coherent=True):
        """Process baseband and extract bits using early-late gate timing."""
        baseband = np.asarray(baseband, dtype=np.float64)

        # BPF for RDS
        rds_signal, self.bpf_zi = signal.lfilter(self.bpf_b, 1.0, baseband, zi=self.bpf_zi)

        # Coherent demod using pilot
        if use_coherent:
            pilot, self.pilot_bpf_zi = signal.lfilter(
                self.pilot_bpf_b, 1.0, baseband, zi=self.pilot_bpf_zi)
            pilot_rms = np.sqrt(np.mean(pilot ** 2))
            if pilot_rms > 1e-10:
                pilot_norm = pilot / (pilot_rms * np.sqrt(2))
                carrier = 4 * pilot_norm**3 - 3 * pilot_norm
                product = rds_signal * carrier
            else:
                product = rds_signal ** 2  # fallback
        else:
            product = rds_signal ** 2

        # Differential decoding
        n = len(product)
        delay = self.delay_samples
        delayed = np.zeros(n)
        if n >= delay:
            delayed[:delay] = self.delay_buf
            delayed[delay:] = product[:n - delay]
            self.delay_buf = product[-delay:].copy()
        else:
            delayed[:n] = self.delay_buf[:n]
            self.delay_buf[:-n] = self.delay_buf[n:]
            self.delay_buf[-n:] = product

        diff_decoded = product * delayed

        # LPF
        bpsk_bb, self.lpf_zi = signal.lfilter(self.lpf_b, self.lpf_a, diff_decoded, zi=self.lpf_zi)

        # AGC
        rms = np.sqrt(np.mean(bpsk_bb ** 2))
        if rms > 1e-10:
            self.agc_rms = 0.95 * self.agc_rms + 0.05 * rms
            bpsk_bb = np.clip(bpsk_bb / max(self.agc_rms, 1e-10), -3.0, 3.0)

        # Early-late gate timing recovery
        self.sample_buffer.extend(bpsk_bb.tolist())
        self._extract_symbols_elg()

        return {
            'blocks_ok': self.blocks_ok,
            'blocks_total': self.blocks_total,
            'block_rate': self.blocks_ok / max(1, self.blocks_total),
            'synced': self.synced,
        }

    def _extract_symbols_elg(self):
        """Extract symbols using early-late gate."""
        sps = self.sps
        early_offset = 0.25  # Sample 1/4 symbol early
        late_offset = 0.25   # Sample 1/4 symbol late

        while True:
            mu = self.timing_mu

            # Calculate sample positions
            early_pos = mu - early_offset * sps
            on_time_pos = mu
            late_pos = mu + late_offset * sps
            next_pos = mu + sps

            if len(self.sample_buffer) < int(next_pos) + 2:
                break

            def interp(pos):
                if pos < 0:
                    return self.sample_buffer[0]
                idx = int(pos)
                if idx >= len(self.sample_buffer) - 1:
                    return self.sample_buffer[-1]
                frac = pos - idx
                return (1 - frac) * self.sample_buffer[idx] + frac * self.sample_buffer[idx + 1]

            early_sample = abs(interp(max(0, early_pos)))
            late_sample = abs(interp(late_pos))
            on_time_sample = interp(on_time_pos)

            # Symbol decision
            bit = 0 if on_time_sample > 0 else 1
            self.bits_extracted += 1
            self.bit_buffer.append(bit)

            # Early-late gate error: positive means we're early
            timing_error = late_sample - early_sample

            # Update timing
            self.timing_mu += self.timing_gain * timing_error

            # Wrap timing_mu
            advance = int(sps)
            while self.timing_mu >= 1.0:
                self.timing_mu -= 1.0
                advance += 1
            while self.timing_mu < 0.0:
                self.timing_mu += 1.0
                advance -= 1

            advance = max(1, advance)
            self.sample_buffer = self.sample_buffer[advance:]

            # Check for block sync
            if len(self.bit_buffer) >= 26:
                self._check_block()

    def _check_block(self):
        """Check if we have a valid RDS block."""
        if len(self.bit_buffer) < 26:
            return

        block = 0
        for b in self.bit_buffer[:26]:
            block = (block << 1) | (b & 1)

        syndrome = compute_syndrome(block)

        # Check all offset words
        valid = False
        for offset in OFFSET_WORDS.values():
            if syndrome == offset:
                valid = True
                break
            # Try inverted
            if compute_syndrome(block ^ 0x3FFFFFF) == offset:
                valid = True
                break

        self.blocks_total += 1
        if valid:
            self.blocks_ok += 1
            self.synced = True
            self.bit_buffer = self.bit_buffer[26:]
        else:
            if self.synced:
                self.bit_buffer = self.bit_buffer[26:]
            else:
                self.bit_buffer = self.bit_buffer[1:]


class MatchedFilterDecoder:
    """
    RDS decoder with proper matched filter (root-raised-cosine).
    """

    def __init__(self, sample_rate=250000, alpha=0.35):
        self.sample_rate = sample_rate
        self.sps = sample_rate / RDS_SYMBOL_RATE
        self.alpha = alpha

        # Design RRC matched filter
        num_taps = int(8 * self.sps) | 1
        self.mf = design_rrc_filter(self.sps, alpha, num_taps)
        self.mf_zi = signal.lfilter_zi(self.mf, 1.0) * 0

        nyq = sample_rate / 2.0

        # 57 kHz bandpass - slightly wider to not truncate symbol energy
        rds_bw = 2400
        low = (RDS_CARRIER_FREQ - rds_bw) / nyq
        high = (RDS_CARRIER_FREQ + rds_bw) / nyq
        self.bpf_b = signal.firwin(201, [max(0.01, low), min(0.99, high)],
                                    pass_zero=False, window=('kaiser', 7.0))
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, 1.0) * 0

        # Pilot bandpass
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf_b = signal.firwin(201, [pilot_low, pilot_high],
                                          pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_zi = signal.lfilter_zi(self.pilot_bpf_b, 1.0) * 0

        # Delay buffer
        self.delay_samples = int(round(self.sps))
        self.delay_buf = np.zeros(self.delay_samples)

        # Timing recovery (Gardner)
        self.timing_mu = 0.5
        self.timing_freq = 0.0
        self.timing_freq_offset = 0.0
        self.timing_gain_p = 0.02
        self.timing_gain_i = 0.001
        self.sample_buffer = []

        # AGC
        self.agc_rms = 0.1

        # Stats
        self.bit_buffer = []
        self.synced = False
        self.blocks_ok = 0
        self.blocks_total = 0

    def process(self, baseband, use_coherent=True):
        baseband = np.asarray(baseband, dtype=np.float64)

        # BPF for RDS
        rds_signal, self.bpf_zi = signal.lfilter(self.bpf_b, 1.0, baseband, zi=self.bpf_zi)

        # Coherent demod
        if use_coherent:
            pilot, self.pilot_bpf_zi = signal.lfilter(
                self.pilot_bpf_b, 1.0, baseband, zi=self.pilot_bpf_zi)
            pilot_rms = np.sqrt(np.mean(pilot ** 2))
            if pilot_rms > 1e-10:
                pilot_norm = pilot / (pilot_rms * np.sqrt(2))
                carrier = 4 * pilot_norm**3 - 3 * pilot_norm
                product = rds_signal * carrier
            else:
                product = rds_signal ** 2
        else:
            product = rds_signal ** 2

        # Differential decode
        n = len(product)
        delay = self.delay_samples
        delayed = np.zeros(n)
        if n >= delay:
            delayed[:delay] = self.delay_buf
            delayed[delay:] = product[:n - delay]
            self.delay_buf = product[-delay:].copy()
        else:
            delayed[:n] = self.delay_buf[:n]
            self.delay_buf[:-n] = self.delay_buf[n:]
            self.delay_buf[-n:] = product

        diff_decoded = product * delayed

        # Matched filter (RRC)
        bpsk_bb, self.mf_zi = signal.lfilter(self.mf, 1.0, diff_decoded, zi=self.mf_zi)

        # AGC
        rms = np.sqrt(np.mean(bpsk_bb ** 2))
        if rms > 1e-10:
            self.agc_rms = 0.95 * self.agc_rms + 0.05 * rms
            bpsk_bb = np.clip(bpsk_bb / max(self.agc_rms, 1e-10), -3.0, 3.0)

        # Timing recovery
        self.sample_buffer.extend(bpsk_bb.tolist())
        self._extract_symbols()

        return {
            'blocks_ok': self.blocks_ok,
            'blocks_total': self.blocks_total,
            'block_rate': self.blocks_ok / max(1, self.blocks_total),
            'synced': self.synced,
        }

    def _extract_symbols(self):
        sps = self.sps
        sps_int = int(sps)
        sps_frac = sps - sps_int
        half_sps = sps / 2.0

        while True:
            mu = max(0, min(1, self.timing_mu))
            pos_prev = mu
            pos_mid = mu + half_sps
            pos_next = mu + sps

            if len(self.sample_buffer) < int(pos_next) + 2:
                break

            def interp(pos):
                idx = int(pos)
                frac = pos - idx
                return (1 - frac) * self.sample_buffer[idx] + frac * self.sample_buffer[idx + 1]

            prev_sample = interp(pos_prev)
            mid_sample = interp(pos_mid)
            next_sample = interp(pos_next)

            bit = 0 if next_sample > 0 else 1
            self.bit_buffer.append(bit)

            # Gardner TED
            timing_error_raw = -(next_sample - prev_sample) * mid_sample
            norm = abs(next_sample) + abs(prev_sample) + abs(mid_sample) + 1e-10
            timing_error = np.clip(timing_error_raw / norm, -0.5, 0.5)

            self.timing_mu += self.timing_gain_p * timing_error
            self.timing_freq_offset += self.timing_gain_i * timing_error
            self.timing_freq_offset = np.clip(self.timing_freq_offset, -1.0, 1.0)

            effective_sps_frac = sps_frac + self.timing_freq_offset
            self.timing_freq += effective_sps_frac
            advance = sps_int
            while self.timing_freq >= 1.0:
                advance += 1
                self.timing_freq -= 1.0
            while self.timing_freq < 0.0:
                advance -= 1
                self.timing_freq += 1.0

            while self.timing_mu >= 1.0:
                self.timing_mu -= 1.0
                advance += 1
            while self.timing_mu < 0.0:
                self.timing_mu += 1.0
                advance -= 1

            advance = max(1, advance)
            self.sample_buffer = self.sample_buffer[advance:]

            if len(self.bit_buffer) >= 26:
                self._check_block()

    def _check_block(self):
        if len(self.bit_buffer) < 26:
            return

        block = 0
        for b in self.bit_buffer[:26]:
            block = (block << 1) | (b & 1)

        syndrome = compute_syndrome(block)

        valid = False
        for offset in OFFSET_WORDS.values():
            if syndrome == offset:
                valid = True
                break
            if compute_syndrome(block ^ 0x3FFFFFF) == offset:
                valid = True
                break

        self.blocks_total += 1
        if valid:
            self.blocks_ok += 1
            self.synced = True
            self.bit_buffer = self.bit_buffer[26:]
        else:
            if self.synced:
                self.bit_buffer = self.bit_buffer[26:]
            else:
                self.bit_buffer = self.bit_buffer[1:]


def run_decoder_test(decoder_class, decoder_name, snr_values, trials=5,
                     sample_rate=250000, repeats=8, **decoder_kwargs):
    """Run SNR sweep test for a decoder."""
    print(f"\n  Testing: {decoder_name}")
    results = []

    for snr_db in snr_values:
        block_rates = []

        for seed in range(trials):
            rng = np.random.default_rng(seed)

            # Build test signal
            pi_code = 0x54A8
            pty = 1
            ps_name = "TESTPJFM"
            bits = _build_bitstream(pi_code, pty, ps_name, repeats=repeats)
            symbols = _differential_encode(bits)
            baseband = _symbols_to_baseband(symbols, sample_rate, include_pilot=True, pad_s=0.2)
            baseband = _add_audio_tone(baseband, sample_rate, amplitude=0.7)

            # FM modulate and add noise
            iq = _fm_modulate(baseband, sample_rate)
            iq = _add_awgn(iq, snr_db, rng)
            demod = _fm_demod(iq, sample_rate)

            # Run decoder
            decoder = decoder_class(sample_rate=sample_rate, **decoder_kwargs)
            block_size = 8192
            result = None
            for i in range(0, len(demod), block_size):
                result = decoder.process(demod[i:i + block_size], use_coherent=True)

            block_rates.append(result['block_rate'])

        avg_rate = np.mean(block_rates)
        results.append({'snr_db': snr_db, 'block_rate': avg_rate})
        print(f"    SNR {snr_db:3d} dB: {avg_rate*100:5.1f}%")

    return results


def test_timing_loop_bandwidth():
    """Test different timing loop bandwidths."""
    print("\n" + "=" * 60)
    print("TIMING LOOP BANDWIDTH COMPARISON")
    print("=" * 60)
    print("\nTesting Gardner TED with different loop gains...")

    snr_values = [30, 25, 22, 20, 18, 15, 12, 10]
    trials = 5

    # Test configurations: (gain_p, gain_i, name)
    configs = [
        (0.03, 0.003, "Wide (original)"),
        (0.015, 0.0008, "Medium (improved)"),
        (0.008, 0.0003, "Narrow"),
        (0.004, 0.0001, "Very narrow"),
    ]

    all_results = {}
    for gain_p, gain_i, name in configs:
        results = run_decoder_test(
            RDSDecoder,
            f"{name} (p={gain_p}, i={gain_i})",
            snr_values,
            trials=trials,
            timing_gain_p=gain_p,
            timing_gain_i=gain_i,
        )
        all_results[name] = results

    # Summary table
    print("\n  SUMMARY - Block rate by SNR and loop bandwidth:")
    print("  " + "-" * 70)
    header = "  SNR  |"
    for name in [c[2] for c in configs]:
        header += f" {name[:12]:>12} |"
    print(header)
    print("  " + "-" * 70)

    for i, snr in enumerate(snr_values):
        row = f"  {snr:3d}  |"
        for name in [c[2] for c in configs]:
            rate = all_results[name][i]['block_rate'] * 100
            row += f" {rate:11.1f}% |"
        print(row)

    return all_results


def test_early_late_gate():
    """Test early-late gate timing recovery."""
    print("\n" + "=" * 60)
    print("EARLY-LATE GATE vs GARDNER TED")
    print("=" * 60)

    snr_values = [30, 25, 22, 20, 18, 15, 12, 10]
    trials = 5

    print("\n  Testing Early-Late Gate with different gains...")

    elg_results = {}
    for gain in [0.005, 0.01, 0.02, 0.04]:
        results = run_decoder_test(
            EarlyLateGateDecoder,
            f"ELG gain={gain}",
            snr_values,
            trials=trials,
            timing_gain=gain,
        )
        elg_results[gain] = results

    print("\n  Testing Gardner TED (baseline)...")
    gardner_results = run_decoder_test(
        RDSDecoder,
        "Gardner TED (baseline)",
        snr_values,
        trials=trials,
    )

    # Summary
    print("\n  SUMMARY:")
    print("  " + "-" * 60)
    print("  SNR  | Gardner TED | ELG 0.01   | ELG 0.02   | ELG 0.04")
    print("  " + "-" * 60)
    for i, snr in enumerate(snr_values):
        g = gardner_results[i]['block_rate'] * 100
        e1 = elg_results[0.01][i]['block_rate'] * 100
        e2 = elg_results[0.02][i]['block_rate'] * 100
        e4 = elg_results[0.04][i]['block_rate'] * 100
        print(f"  {snr:3d}  | {g:10.1f}% | {e1:9.1f}% | {e2:9.1f}% | {e4:9.1f}%")


def test_matched_filter():
    """Test matched filter (RRC) vs simple LPF."""
    print("\n" + "=" * 60)
    print("MATCHED FILTER (RRC) vs SIMPLE LPF")
    print("=" * 60)

    snr_values = [30, 25, 22, 20, 18, 15, 12, 10]
    trials = 5

    print("\n  Testing RRC matched filter with different rolloff...")

    mf_results = {}
    for alpha in [0.25, 0.35, 0.5]:
        results = run_decoder_test(
            MatchedFilterDecoder,
            f"RRC alpha={alpha}",
            snr_values,
            trials=trials,
            alpha=alpha,
        )
        mf_results[alpha] = results

    print("\n  Testing simple LPF (baseline)...")
    lpf_results = run_decoder_test(
        RDSDecoder,
        "Simple LPF (baseline)",
        snr_values,
        trials=trials,
    )

    # Summary
    print("\n  SUMMARY:")
    print("  " + "-" * 60)
    print("  SNR  | Simple LPF | RRC 0.25  | RRC 0.35  | RRC 0.50")
    print("  " + "-" * 60)
    for i, snr in enumerate(snr_values):
        lpf = lpf_results[i]['block_rate'] * 100
        r25 = mf_results[0.25][i]['block_rate'] * 100
        r35 = mf_results[0.35][i]['block_rate'] * 100
        r50 = mf_results[0.5][i]['block_rate'] * 100
        print(f"  {snr:3d}  | {lpf:9.1f}% | {r25:8.1f}% | {r35:8.1f}% | {r50:8.1f}%")


def test_bpf_bandwidth():
    """Test different 57 kHz BPF bandwidths."""
    print("\n" + "=" * 60)
    print("57 kHz BANDPASS FILTER WIDTH")
    print("=" * 60)

    snr_values = [30, 25, 22, 20, 18, 15, 12, 10]
    trials = 5

    print("\n  Testing different BPF bandwidths...")

    results = {}
    for bw in [1500, 2000, 2400, 3000]:
        r = run_decoder_test(
            RDSDecoder,
            f"BPF +/-{bw} Hz",
            snr_values,
            trials=trials,
            bpf_bandwidth=bw,
        )
        results[bw] = r

    # Summary
    print("\n  SUMMARY:")
    print("  " + "-" * 55)
    print("  SNR  | +/-1500Hz | +/-2000Hz | +/-2400Hz | +/-3000Hz")
    print("  " + "-" * 55)
    for i, snr in enumerate(snr_values):
        r1 = results[1500][i]['block_rate'] * 100
        r2 = results[2000][i]['block_rate'] * 100
        r3 = results[2400][i]['block_rate'] * 100
        r4 = results[3000][i]['block_rate'] * 100
        print(f"  {snr:3d}  | {r1:8.1f}% | {r2:8.1f}% | {r3:8.1f}% | {r4:8.1f}%")


def run_all_alternatives():
    """Run all alternative mechanism tests."""
    print("\n" + "=" * 70)
    print("RDS DECODER ALTERNATIVE MECHANISM TESTS")
    print("=" * 70)

    test_bpf_bandwidth()
    test_timing_loop_bandwidth()
    test_matched_filter()
    test_early_late_gate()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_alternatives()

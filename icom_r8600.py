#!/usr/bin/env python3
"""
IC-R8600 I/Q Streaming Interface for pyfm

Provides a BB60D-compatible interface for the Icom IC-R8600's USB I/Q output.
Uses the protocol discovered via USB traffic capture analysis.

Requires: pip install pyusb
Linux: sudo access or udev rules for USB device
"""

import numpy as np
import usb.core
import usb.util
import threading
import time

# Icom USB IDs
ICOM_VID = 0x0c26
R8600_PID_READY = 0x0023     # Device ready for I/Q streaming
R8600_PID_BOOTLOADER = 0x0022  # Device needs firmware upload

# Cypress FX2 CPUCS register address
CPUCS_ADDR = 0xE600

# USB Endpoints
EP_CMD_OUT = 0x02   # CI-V commands PC -> R8600
EP_IQ_IN = 0x86     # I/Q data R8600 -> PC
EP_RESP_IN = 0x88   # CI-V responses R8600 -> PC

# CI-V Protocol
CIV_PREAMBLE = 0xFE
CIV_TERMINATOR = 0xFD
CIV_OK = 0xFB
CIV_NG = 0xFA
R8600_ADDR = 0x96
PC_ADDR = 0xE0

# Sample rate codes for I/Q output (1A 13 01 01 BD SR)
# BD = bit depth (00=16-bit, 01=24-bit)
# SR = sample rate code
# Note: 5.12 MSPS only supports 16-bit; other rates support both
# Using 16-bit by default for stability; 24-bit available but needs more testing
SAMPLE_RATES = {
    5120000: (0x00, 0x01),  # 5.12 MSPS, 16-bit only (hardware limitation)
    3840000: (0x00, 0x02),  # 3.84 MSPS, 16-bit
    1920000: (0x00, 0x03),  # 1.92 MSPS, 16-bit
    960000:  (0x00, 0x04),  # 960 KSPS, 16-bit
    480000:  (0x00, 0x05),  # 480 KSPS, 16-bit
    240000:  (0x00, 0x06),  # 240 KSPS, 16-bit
}

# Alternative 24-bit sample rates (for future use when 24-bit is debugged)
SAMPLE_RATES_24BIT = {
    3840000: (0x01, 0x02),  # 3.84 MSPS, 24-bit
    1920000: (0x01, 0x03),  # 1.92 MSPS, 24-bit
    960000:  (0x01, 0x04),  # 960 KSPS, 24-bit
    480000:  (0x01, 0x05),  # 480 KSPS, 24-bit
    240000:  (0x01, 0x06),  # 240 KSPS, 24-bit
}


def _build_civ_command(cmd_data):
    """Build a CI-V command packet with even-length padding."""
    packet = bytes([CIV_PREAMBLE, CIV_PREAMBLE, R8600_ADDR, PC_ADDR]) + bytes(cmd_data) + bytes([CIV_TERMINATOR])
    if len(packet) % 2 != 0:
        packet += bytes([0xFF])
    return packet


def _freq_to_bcd(freq_hz):
    """Convert frequency in Hz to BCD format (LSB first, 5 bytes)."""
    freq_str = f"{int(freq_hz):010d}"
    bcd = []
    for i in range(8, -2, -2):
        high = int(freq_str[i])
        low = int(freq_str[i + 1])
        bcd.append((high << 4) | low)
    return bytes(bcd)


def _load_hex_file(filename):
    """Parse Intel HEX file and return list of (addr, data) tuples."""
    segments = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith(':'):
                continue
            rec = bytes.fromhex(line[1:])
            length, addr_hi, addr_lo, rec_type = rec[0], rec[1], rec[2], rec[3]
            addr = (addr_hi << 8) | addr_lo
            data = rec[4:4+length]
            if rec_type == 0:
                segments.append((addr, data))
            elif rec_type == 1:
                break
    return segments


def _upload_firmware_stage(dev, segments):
    """Upload a firmware stage to Cypress FX2 device."""
    # Stop CPU
    dev.ctrl_transfer(0x40, 0xA0, CPUCS_ADDR, 0, bytes([0x01]), timeout=1000)

    # Upload segments
    for addr, data in segments:
        dev.ctrl_transfer(0x40, 0xA0, addr, 0, data, timeout=1000)

    # Start CPU
    dev.ctrl_transfer(0x40, 0xA0, CPUCS_ADDR, 0, bytes([0x00]), timeout=1000)


def _find_firmware_file():
    """Find the IC-R8600 firmware .spt file."""
    import os
    search_paths = [
        os.path.join(os.path.dirname(__file__), "IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/dev/IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/dev/r8600start/IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/.local/share/pyfm/IC-R8600_usb_iq.spt"),
        "/usr/share/pyfm/IC-R8600_usb_iq.spt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def _extract_firmware_from_spt(spt_path):
    """Extract firmware stages from Cypress .spt script file."""
    import struct

    with open(spt_path, 'rb') as f:
        data = f.read()

    stages = []
    current_stage = []
    cpucs_state = None
    pos = 0

    while pos < len(data):
        if data[pos:pos+4] != b"CSPT":
            break

        length = struct.unpack("<L", data[pos+4:pos+8])[0]
        if length < 32 or pos + length > len(data):
            break

        # Parse chunk header
        fmt = "<LLLLBBHLLL"
        hdr_len = struct.calcsize(fmt)
        (_, _, _, _, request, _, addr, _, _, data_len) = struct.unpack(fmt, data[pos:pos+hdr_len])
        chunk_data = data[pos+hdr_len:pos+hdr_len+data_len]

        if request == 0xA0:  # Firmware upload request
            if addr == CPUCS_ADDR and len(chunk_data) == 1:
                # CPUCS write - controls CPU reset
                new_state = chunk_data[0] & 1
                if cpucs_state == 1 and new_state == 0:
                    # CPU started - end of stage
                    if current_stage:
                        stages.append(current_stage)
                        current_stage = []
                cpucs_state = new_state
            else:
                current_stage.append((addr, chunk_data))

        pos += length

    return stages


def _switch_to_iq_mode(bootloader_dev):
    """Upload firmware to switch IC-R8600 from bootloader (0x0022) to I/Q mode (0x0023)."""
    print("IC-R8600 detected in bootloader mode (PID 0x0022)")
    print("Uploading firmware to enable I/Q streaming...")

    # Find and extract firmware
    spt_path = _find_firmware_file()
    if not spt_path:
        raise RuntimeError(
            "IC-R8600 firmware file not found.\n"
            "Please copy IC-R8600_usb_iq.spt to ~/dev/ or ~/.local/share/pyfm/\n"
            "This file is from the Icom USB I/Q Package for HDSDR."
        )

    stages = _extract_firmware_from_spt(spt_path)
    if len(stages) < 2:
        raise RuntimeError(f"Invalid firmware file: expected 2 stages, got {len(stages)}")

    # Detach kernel driver
    try:
        if bootloader_dev.is_kernel_driver_active(0):
            bootloader_dev.detach_kernel_driver(0)
    except (usb.core.USBError, NotImplementedError):
        pass

    # Upload both firmware stages
    print(f"  Loading stage 1 ({sum(len(d) for _, d in stages[0])} bytes)...")
    _upload_firmware_stage(bootloader_dev, stages[0])
    time.sleep(0.5)

    # Re-find device (might have re-enumerated)
    dev = usb.core.find(idVendor=ICOM_VID)
    if dev is None:
        raise RuntimeError("Device lost after stage 1 firmware upload")

    if dev.idProduct == R8600_PID_READY:
        print("  Device switched to I/Q mode!")
        return dev

    # Need stage 2
    try:
        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)
    except (usb.core.USBError, NotImplementedError):
        pass

    print(f"  Loading stage 2 ({sum(len(d) for _, d in stages[1])} bytes)...")
    _upload_firmware_stage(dev, stages[1])

    # Wait for device to re-enumerate as PID 0x0023
    # This can take a few seconds as USB re-enumerates and udev applies rules
    print("  Waiting for device to initialize...")
    ready_dev = None
    for attempt in range(10):
        time.sleep(0.5)
        ready_dev = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
        if ready_dev is not None:
            break

    if ready_dev is None:
        raise RuntimeError("Firmware upload completed but device did not switch to I/Q mode")

    # Extra delay for device to fully initialize and udev rules to apply
    time.sleep(0.5)

    print("  Device switched to I/Q mode!")
    return ready_dev


class IcomR8600:
    """
    IC-R8600 I/Q streaming interface compatible with BB60D API.

    Provides the same methods as BB60D for use with pyfm's FMRadio class.
    """

    # Frequency limits - R8600 covers 10 kHz to 3 GHz
    MIN_FREQ = 10e3
    MAX_FREQ = 3000e6

    # FM broadcast band
    FM_MIN_FREQ = 88.0e6
    FM_MAX_FREQ = 108.0e6
    FM_STEP = 100e3

    DEFAULT_FREQ = 89.9e6

    def __init__(self):
        self.device = None
        self.frequency = self.DEFAULT_FREQ
        self.streaming_mode = None
        self.total_sample_loss = 0
        self.recent_sample_loss = 0

        # I/Q streaming state
        self.iq_sample_rate = 0
        self.iq_bandwidth = 0
        self._bit_depth = 16
        self._bytes_per_sample = 4  # 16-bit: 2 bytes I + 2 bytes Q

        # I/Q gain - applied to normalized samples
        # For FM demodulation, this should be 1.0 since phase detection is
        # amplitude-independent. Higher gain only needed if downstream
        # processing requires it.
        self._iq_gain = 1.0

        # DC offset tracking - per Icom I/Q Reference Guide, there's a DC
        # component in the I/Q data that varies with sample rate. We use
        # an exponential moving average to track and remove it.
        self._dc_offset = 0.0 + 0.0j  # Complex DC offset estimate
        self._dc_alpha = 0.001  # EMA smoothing factor (lower = slower tracking)

        # I/Q data buffer and thread
        self._iq_buffer = []
        self._iq_lock = threading.Lock()
        self._iq_thread = None
        self._running = False

    def open(self):
        """Open connection to IC-R8600 I/Q USB port."""
        firmware_uploaded = False

        # First try to find device in ready state (PID 0x0023)
        self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)

        if self.device is None:
            # Check for bootloader mode (PID 0x0022) - needs firmware upload
            bootloader_dev = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_BOOTLOADER)
            if bootloader_dev is not None:
                self.device = _switch_to_iq_mode(bootloader_dev)
                firmware_uploaded = True
            else:
                raise RuntimeError(
                    f"IC-R8600 not found.\n"
                    "Make sure the radio is connected via the USB I/Q port (not the main USB port)."
                )

        # After firmware upload, the device handle may be stale - re-find it
        if firmware_uploaded:
            time.sleep(0.5)
            self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
            if self.device is None:
                raise RuntimeError("Device lost after firmware upload")

        # Retry loop for USB initialization (device may need time to settle)
        last_error = None
        for attempt in range(3):
            try:
                # Detach kernel driver if needed
                try:
                    if self.device.is_kernel_driver_active(0):
                        self.device.detach_kernel_driver(0)
                except (usb.core.USBError, NotImplementedError):
                    pass

                # Set configuration
                self.device.set_configuration()

                # Claim interface 0 for communication
                usb.util.claim_interface(self.device, 0)

                # Success
                last_error = None
                break

            except usb.core.USBError as e:
                last_error = e
                if attempt < 2:
                    time.sleep(0.5)
                    # Re-find device in case it re-enumerated
                    self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
                    if self.device is None:
                        raise RuntimeError("Device lost during initialization")

        if last_error is not None:
            raise RuntimeError(f"Failed to initialize USB device: {last_error}")

        # Get device info
        try:
            manufacturer = usb.util.get_string(self.device, self.device.iManufacturer)
            product = usb.util.get_string(self.device, self.device.iProduct)
            print(f"Connected to {manufacturer} {product}")
        except usb.core.USBError:
            print(f"Connected to IC-R8600 (VID={ICOM_VID:04x} PID={R8600_PID_READY:04x})")

        return self.device

    def close(self):
        """Close connection and disable I/Q mode."""
        self._running = False
        if self._iq_thread:
            self._iq_thread.join(timeout=1.0)
            self._iq_thread = None

        if self.device:
            try:
                # Disable I/Q output first, with delay for processing
                self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=1000)
                time.sleep(0.2)
                # Disable I/Q mode (turn off REMOTE LED)
                self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x00]), timeout=1000)
                time.sleep(0.2)
            except (usb.core.USBError, RuntimeError):
                pass

            try:
                usb.util.release_interface(self.device, 0)
            except usb.core.USBError:
                pass

            try:
                usb.util.dispose_resources(self.device)
            except usb.core.USBError:
                pass
            self.device = None

    def _send_command(self, cmd_bytes, timeout=2000):
        """Send CI-V command and return response."""
        try:
            self.device.write(EP_CMD_OUT, cmd_bytes, timeout=timeout)
        except usb.core.USBError as e:
            raise RuntimeError(f"Failed to send command: {e}")

        try:
            response = self.device.read(EP_RESP_IN, 64, timeout=timeout)
            return bytes(response)
        except usb.core.USBTimeoutError:
            return None
        except usb.core.USBError as e:
            raise RuntimeError(f"Failed to read response: {e}")

    def configure_iq_streaming(self, freq=None, sample_rate=250000):
        """
        Configure IC-R8600 for I/Q streaming.

        Args:
            freq: Center frequency in Hz
            sample_rate: Desired sample rate (will use nearest available)
        """
        if freq is not None:
            self.frequency = freq

        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, self.frequency))

        # Find nearest available sample rate
        # For FM, we need at least 250 kHz, so 480 KSPS is minimum
        available_rates = sorted(SAMPLE_RATES.keys())
        chosen_rate = available_rates[0]
        for rate in available_rates:
            if rate >= sample_rate:
                chosen_rate = rate
                break
        else:
            chosen_rate = available_rates[-1]  # Use highest if requested is higher

        bit_depth, rate_code = SAMPLE_RATES[chosen_rate]
        self._bit_depth = 16 if bit_depth == 0x00 else 24
        self._bytes_per_sample = 4 if self._bit_depth == 16 else 6

        # First, ensure clean state by disabling any existing I/Q output
        # These may timeout if already disabled - that's OK
        try:
            self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=500)
        except RuntimeError:
            pass
        time.sleep(0.1)

        # Enable I/Q mode (REMOTE LED on)
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x01]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q mode")

        # Set frequency
        bcd_freq = _freq_to_bcd(self.frequency)
        resp = self._send_command(_build_civ_command([0x05] + list(bcd_freq)))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to set frequency")

        # Query and display current RF settings (don't override user's radio settings)
        rf_settings = self.query_rf_settings()
        if rf_settings:
            print(f"RF Settings: Gain={rf_settings.get('rf_gain', '?')}, "
                  f"Att={rf_settings.get('attenuator', '?')}dB, "
                  f"Preamp={'ON' if rf_settings.get('preamp') else 'OFF'}")

        # Enable HF BPF (for HF frequencies)
        if self.frequency < 30e6:
            self._send_command(_build_civ_command([0x1A, 0x13, 0x02, 0x01]))

        # Enable I/Q output with chosen sample rate
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x01, bit_depth, rate_code]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q output")

        self.iq_sample_rate = chosen_rate
        # Bandwidth is approximately 87% of sample rate for R8600
        self.iq_bandwidth = chosen_rate * 0.87
        self.streaming_mode = "iq"

        # Start I/Q reader thread
        self._running = True
        self._iq_buffer = []
        self._iq_thread = threading.Thread(target=self._iq_reader_loop, daemon=True)
        self._iq_thread.start()

        print(f"IQ Streaming: {self.iq_sample_rate/1e6:.3f} MS/s ({self._bit_depth}-bit), BW: {self.iq_bandwidth/1e3:.1f} kHz, Gain: {self._iq_gain:.0f}x")

    def _iq_reader_loop(self):
        """Background thread to continuously read I/Q data from USB."""
        while self._running:
            try:
                # Read large chunks for efficiency
                data = self.device.read(EP_IQ_IN, 65536, timeout=1000)
                if data:
                    with self._iq_lock:
                        self._iq_buffer.append(bytes(data))
            except usb.core.USBTimeoutError:
                continue
            except usb.core.USBError:
                if self._running:
                    self.recent_sample_loss += 1
                    self.total_sample_loss += 1
                break

    def fetch_iq(self, num_samples=8192):
        """
        Fetch IQ samples for software demodulation.

        Args:
            num_samples: Number of IQ samples to fetch

        Returns:
            numpy array of complex64 IQ samples
        """
        if self.streaming_mode != "iq":
            raise RuntimeError("Device not in IQ streaming mode")

        bytes_needed = num_samples * self._bytes_per_sample
        collected = b''

        # Collect enough data from buffer
        timeout = time.time() + 1.0  # 1 second timeout
        while len(collected) < bytes_needed and time.time() < timeout:
            with self._iq_lock:
                while self._iq_buffer and len(collected) < bytes_needed:
                    chunk = self._iq_buffer.pop(0)
                    collected += chunk
            if len(collected) < bytes_needed:
                time.sleep(0.001)

        if len(collected) < bytes_needed:
            # Not enough data - return what we have padded with zeros
            self.recent_sample_loss += 1
            self.total_sample_loss += 1

        # Parse I/Q data based on bit depth
        if self._bit_depth == 16:
            # 16-bit: I16 Q16 (4 bytes per sample)
            samples_available = len(collected) // 4
            samples_to_use = min(samples_available, num_samples)

            if samples_to_use > 0:
                raw = np.frombuffer(collected[:samples_to_use * 4], dtype=np.int16)
                # Reshape to (N, 2) then view as interleaved I, Q
                iq_int = raw.reshape(-1, 2)

                # Filter out sync patterns and invalid samples
                # Per Icom I/Q Reference: valid range is -32767 to +32767
                # Value -32768 (0x8000) only appears in sync patterns
                # Sync pattern for 16-bit: 0x8000, 0x8000 every 1024 samples
                # Filter any sample where I OR Q is -32768 (invalid/sync)
                invalid_mask = (iq_int[:, 0] == -32768) | (iq_int[:, 1] == -32768)
                if np.any(invalid_mask):
                    iq_int = iq_int[~invalid_mask]

                # Convert to complex64, normalizing to [-1, 1]
                iq = (iq_int[:, 0].astype(np.float32) + 1j * iq_int[:, 1].astype(np.float32)) / 32768.0

                # Remove DC offset (per Icom I/Q Reference Guide, DC component exists)
                # Update DC estimate with exponential moving average
                block_dc = np.mean(iq)
                self._dc_offset = self._dc_alpha * block_dc + (1 - self._dc_alpha) * self._dc_offset
                iq = iq - self._dc_offset

                # Apply gain (should be 1.0 for FM since demod is phase-based)
                if self._iq_gain != 1.0:
                    iq = iq * self._iq_gain
            else:
                iq = np.zeros(0, dtype=np.complex64)

            # Pad if needed
            if len(iq) < num_samples:
                iq = np.concatenate([iq, np.zeros(num_samples - len(iq), dtype=np.complex64)])

        else:
            # 24-bit: I24 Q24 (6 bytes per sample)
            samples_available = len(collected) // 6
            samples_to_use = min(samples_available, num_samples)

            if samples_to_use > 0:
                # Parse 24-bit samples using numpy for efficiency
                # Each sample is 6 bytes: I0 I1 I2 Q0 Q1 Q2 (little-endian)
                raw = np.frombuffer(collected[:samples_to_use * 6], dtype=np.uint8)
                raw = raw.reshape(-1, 6)

                # Extract I and Q as 24-bit values (little-endian)
                # Combine bytes: val = b0 | (b1 << 8) | (b2 << 16)
                i_vals = (raw[:, 0].astype(np.int32) |
                          (raw[:, 1].astype(np.int32) << 8) |
                          (raw[:, 2].astype(np.int32) << 16))
                q_vals = (raw[:, 3].astype(np.int32) |
                          (raw[:, 4].astype(np.int32) << 8) |
                          (raw[:, 5].astype(np.int32) << 16))

                # Sign extend from 24-bit to 32-bit
                i_vals = np.where(i_vals & 0x800000, i_vals - 0x1000000, i_vals)
                q_vals = np.where(q_vals & 0x800000, q_vals - 0x1000000, q_vals)

                # Filter out sync patterns per Icom I/Q Reference Guide:
                # 24-bit sync is 6 bytes: 0x8000, 0x8001, 0x8002 (as 16-bit words)
                # When parsed as 24-bit I/Q: I=0x018000=98304, Q=0x800280→-8387968
                # Valid I/Q range is -8387967 to +8387966, so Q < -8387967 indicates sync
                # Also check for I matching sync pattern (I == 98304 AND Q == -8387968)
                sync_mask = (i_vals == 98304) & (q_vals == -8387968)
                if np.any(sync_mask):
                    i_vals = i_vals[~sync_mask]
                    q_vals = q_vals[~sync_mask]

                # Convert to complex64, normalizing to [-1, 1]
                # Normalize by 8388608 (2^23) for full 24-bit range
                iq = (i_vals.astype(np.float32) + 1j * q_vals.astype(np.float32)) / 8388608.0

                # Remove DC offset (per Icom I/Q Reference Guide, DC component exists)
                # Update DC estimate with exponential moving average
                block_dc = np.mean(iq)
                self._dc_offset = self._dc_alpha * block_dc + (1 - self._dc_alpha) * self._dc_offset
                iq = iq - self._dc_offset

                # Apply gain (should be 1.0 for FM since demod is phase-based)
                if self._iq_gain != 1.0:
                    iq = iq * self._iq_gain
            else:
                iq = np.zeros(0, dtype=np.complex64)

            # Pad if needed
            if len(iq) < num_samples:
                iq = np.concatenate([iq, np.zeros(num_samples - len(iq), dtype=np.complex64)])

        # Put excess data back in buffer
        excess_bytes = len(collected) - (num_samples * self._bytes_per_sample)
        if excess_bytes > 0:
            with self._iq_lock:
                self._iq_buffer.insert(0, collected[-(excess_bytes):])

        self.recent_sample_loss = 0
        return iq[:num_samples]

    def flush_iq(self):
        """Flush stale IQ data from the buffer."""
        with self._iq_lock:
            self._iq_buffer.clear()
        # Reset DC offset estimate (DC varies with frequency per Icom docs)
        self._dc_offset = 0.0 + 0.0j
        self.total_sample_loss = 0
        self.recent_sample_loss = 0

    def set_frequency(self, freq):
        """Change the tuned frequency."""
        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, freq))

        if self.streaming_mode == "iq" and self.device:
            bcd_freq = _freq_to_bcd(self.frequency)
            resp = self._send_command(_build_civ_command([0x05] + list(bcd_freq)))
            if resp and CIV_NG in resp:
                raise RuntimeError("Failed to set frequency")

    def tune_up(self):
        """Tune up by FM step (100 kHz)."""
        new_freq = self.frequency + self.FM_STEP
        if new_freq > self.FM_MAX_FREQ:
            new_freq = self.FM_MIN_FREQ
        self.set_frequency(new_freq)
        # Reconfigure to apply new frequency
        if self.streaming_mode == "iq":
            self.flush_iq()

    def tune_down(self):
        """Tune down by FM step (100 kHz)."""
        new_freq = self.frequency - self.FM_STEP
        if new_freq < self.FM_MIN_FREQ:
            new_freq = self.FM_MAX_FREQ
        self.set_frequency(new_freq)
        if self.streaming_mode == "iq":
            self.flush_iq()

    @property
    def frequency_mhz(self):
        """Current frequency in MHz."""
        return self.frequency / 1e6

    @property
    def iq_gain(self):
        """Current I/Q software gain multiplier."""
        return self._iq_gain

    @iq_gain.setter
    def iq_gain(self, value):
        """Set I/Q software gain multiplier."""
        self._iq_gain = max(1.0, float(value))

    # ========== RF Gain Management ==========

    def query_rf_settings(self):
        """
        Query current RF settings from the radio.

        Returns:
            dict with rf_gain, attenuator, preamp, ip_plus settings
        """
        settings = {}

        # Query RF gain (14 02) - response includes 2-byte BCD value
        resp = self._send_command(_build_civ_command([0x14, 0x02]))
        if resp and len(resp) >= 8 and CIV_OK not in resp:
            # Response format: FE FE E0 96 14 02 XX XX FD
            # XX XX is BCD value 0000-0255
            try:
                idx = resp.index(0x14)
                if idx + 3 < len(resp):
                    high = resp[idx + 2]
                    low = resp[idx + 3]
                    # BCD decode: 0x02 0x55 = 255
                    settings['rf_gain'] = (high >> 4) * 1000 + (high & 0x0F) * 100 + (low >> 4) * 10 + (low & 0x0F)
            except (ValueError, IndexError):
                pass

        # Query attenuator (11) - response includes setting byte
        resp = self._send_command(_build_civ_command([0x11]))
        if resp and len(resp) >= 6:
            try:
                idx = resp.index(0x11)
                if idx + 1 < len(resp):
                    att_val = resp[idx + 1]
                    # 0x00=0dB, 0x10=10dB, 0x20=20dB, 0x30=30dB
                    settings['attenuator'] = (att_val >> 4) * 10
            except (ValueError, IndexError):
                pass

        # Query preamp (16 02)
        resp = self._send_command(_build_civ_command([0x16, 0x02]))
        if resp and len(resp) >= 7:
            try:
                idx = resp.index(0x16)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x02:
                    settings['preamp'] = resp[idx + 2] == 0x01
            except (ValueError, IndexError):
                pass

        # Query IP+ (16 65)
        resp = self._send_command(_build_civ_command([0x16, 0x65]))
        if resp and len(resp) >= 7:
            try:
                idx = resp.index(0x16)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x65:
                    settings['ip_plus'] = resp[idx + 2] == 0x01
            except (ValueError, IndexError):
                pass

        return settings

    def set_rf_gain(self, level):
        """
        Set RF gain level.

        Args:
            level: 0-255 (0=min, 255=max)

        Note: RF gain commands may be rejected if I/Q output is actively streaming.
        Set RF gain before enabling I/Q output, or temporarily disable I/Q output.
        """
        level = max(0, min(255, int(level)))
        # Convert to 4-digit BCD: 255 -> 0x02 0x55 (digits: 0,2,5,5)
        d0 = level // 1000         # thousands
        d1 = (level // 100) % 10   # hundreds
        d2 = (level // 10) % 10    # tens
        d3 = level % 10            # ones
        high = (d0 << 4) | d1      # 0x02 for 255
        low = (d2 << 4) | d3       # 0x55 for 255
        resp = self._send_command(_build_civ_command([0x14, 0x02, high, low]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set RF gain to {level}")

    def set_attenuator(self, db):
        """
        Set attenuator level.

        Args:
            db: 0, 10, 20, or 30 dB
        """
        if db not in (0, 10, 20, 30):
            raise ValueError("Attenuator must be 0, 10, 20, or 30 dB")
        att_val = (db // 10) << 4  # 0->0x00, 10->0x10, 20->0x20, 30->0x30
        resp = self._send_command(_build_civ_command([0x11, att_val]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set attenuator to {db} dB")

    def set_preamp(self, enabled):
        """
        Enable or disable preamplifier.

        Args:
            enabled: True to enable, False to disable
        """
        resp = self._send_command(_build_civ_command([0x16, 0x02, 0x01 if enabled else 0x00]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set preamp to {enabled}")

    def set_ip_plus(self, enabled):
        """
        Enable or disable IP+ (Intercept Point Plus).

        Args:
            enabled: True to enable, False to disable
        """
        resp = self._send_command(_build_civ_command([0x16, 0x65, 0x01 if enabled else 0x00]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set IP+ to {enabled}")

    def auto_gain(self):
        """
        Automatically adjust RF gain based on signal strength.
        For FM broadcast, aims for moderate gain to avoid clipping.
        """
        # Query S-meter (15 02) to check signal level
        resp = self._send_command(_build_civ_command([0x15, 0x02]))
        if resp:
            try:
                idx = resp.index(0x15)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x02:
                    # S-meter value is 2-byte BCD, 0000-0255
                    high = resp[idx + 2]
                    low = resp[idx + 3] if idx + 3 < len(resp) else 0
                    s_meter = (high >> 4) * 1000 + (high & 0x0F) * 100 + (low >> 4) * 10 + (low & 0x0F)
                    print(f"S-Meter: {s_meter}")

                    # Adjust RF gain based on signal strength
                    # Strong signal (>150) -> reduce gain
                    # Weak signal (<50) -> increase gain
                    settings = self.query_rf_settings()
                    current_gain = settings.get('rf_gain', 128)

                    if s_meter > 200:
                        new_gain = max(0, current_gain - 50)
                    elif s_meter > 150:
                        new_gain = max(0, current_gain - 20)
                    elif s_meter < 30:
                        new_gain = min(255, current_gain + 20)
                    elif s_meter < 50:
                        new_gain = min(255, current_gain + 10)
                    else:
                        new_gain = current_gain

                    if new_gain != current_gain:
                        print(f"Adjusting RF gain: {current_gain} -> {new_gain}")
                        self.set_rf_gain(new_gain)
            except (ValueError, IndexError):
                pass


def get_api_version():
    """Return version string for compatibility with BB60D interface."""
    return "IC-R8600 USB I/Q 1.0"

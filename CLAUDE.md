pyfm is a software-defined FM radio receiver supporting the SignalHound BB60D and Icom IC-R8600.

It receives broadcast FM (88-108 MHz) and NOAA Weather Radio (162 MHz), performs all demodulation
in software, and plays audio through the default Linux audio device.

## Supported Hardware

- **SignalHound BB60D** - Spectrum analyzer with I/Q streaming at 312.5 kHz
- **Icom IC-R8600** - Communications receiver with USB I/Q output at 240-5120 kHz

## Running

```bash
# With BB60D (default)
./pyfm.py [frequency_mhz]

# With IC-R8600
./pyfm.py --icom [frequency_mhz]
```

## Controls

| Key | Function |
|-----|----------|
| ←/→ | Tune down/up (100 kHz FM, 25 kHz Weather) |
| ↑/↓ | Volume up/down |
| 1-5 | Recall preset (FM) or WX1-WX7 (Weather) |
| !@#$% | Set preset to current frequency (FM mode) |
| w | Toggle Weather radio mode (NBFM) |
| r | Toggle RDS decoding (FM mode) |
| b | Toggle bass boost (+3 dB) |
| t | Toggle treble boost (+3 dB) |
| a | Toggle spectrum analyzer |
| G | Toggle GPU acceleration (FM mode) |
| Q | Toggle squelch |
| q | Quit |

## Dependencies

```bash
pip install numpy scipy sounddevice rich pyusb  # pyusb for IC-R8600
pip install torch  # Optional: GPU acceleration (ROCm or CUDA)
```

## Architecture

### Core Components

- `pyfm.py` - Main application, terminal UI, audio playback
- `demodulator.py` - FM stereo decoder with PLL-based pilot tracking
- `rds_decoder.py` - RDS/RBDS decoding (station ID, program type, radio text)
- `bb60d.py` - SignalHound BB60D I/Q streaming interface
- `icom_r8600.py` - IC-R8600 USB I/Q interface with CI-V control
- `gpu.py` - GPU-accelerated FM demod, FIR filters, polyphase resampler

### IC-R8600 I/Q Interface

The IC-R8600 requires firmware upload on USB connection (automatic):
- Firmware file: `IC-R8600_usb_iq.spt` (from Icom USB I/Q Package for HDSDR)
- Place in `~/dev/`, `~/.local/share/pyfm/`, or same directory as script

Sample rates: 240 kHz, 480 kHz, 960 kHz, 1.92 MHz, 3.84 MHz, 5.12 MHz
Bit depth: 16-bit (default), 24-bit available (needs testing)

Per Icom I/Q Reference Guide:
- Sync patterns inserted periodically (filtered out automatically)
- DC offset present in I/Q data (removed via EMA tracking)
- Valid sample range: -32767 to +32767 (16-bit)

### Signal Processing

**FM Demodulation**: Quadrature discriminator extracts instantaneous frequency:
```python
baseband = angle(s[n] * conj(s[n-1])) * (sample_rate / (2π * deviation))
```

**Stereo Decoding**: PLL tracks 19 kHz pilot, regenerates 38 kHz carrier for L-R demod

**Weather Radio**: NBFM with 5 kHz deviation for NOAA NWS (162.400-162.550 MHz)

**GPU Acceleration**: PyTorch-based (ROCm/CUDA) for:
- FM demodulation (arctangent + differentiation)
- FIR filter bank (pilot BPF, L+R LPF, L-R BPF)
- Polyphase resampler (IQ rate → 48 kHz audio)

## Panadapter GUI

`panadapter.py` is a PyQt5-based spectrum analyzer with waterfall display.

### Running

```bash
python panadapter.py [--freq FREQ_MHZ]
```

Default center frequency: 162.500 MHz (NOAA weather radio band)

### Features

- Real-time spectrum display with peak hold (press 'P' to toggle)
- Scrolling waterfall display with viridis colormap
- NOAA weather radio preset buttons (WX1-WX7)
- Keyboard controls: Left/Right arrows (±25 kHz), Up/Down (±100 kHz)
- Direct frequency entry
- FM demodulation with audio output

### Dependencies

```bash
pip install PyQt5 pyqtgraph
```

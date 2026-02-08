/**
 * pjfm Web Interface - Client-side JavaScript
 *
 * Handles WebSocket communication for state updates and audio streaming,
 * keyboard controls, and UI updates.
 */

// Audio context and playback state
let audioContext = null;
let audioPlaying = false;
let audioQueue = [];
let nextPlayTime = 0;
const BUFFER_SIZE = 4096;
const SAMPLE_RATE = 48000;

// Opus decoder
let opusDecoder = null;
let opusDecoderReady = false;

// WebSocket connection
let socket = null;
let statePollingInterval = null;

// Connect to the server
function connect() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
        document.getElementById('audio-indicator').textContent = 'Connected';
        // Start polling for state updates via REST API (more reliable than WebSocket push)
        startStatePolling();
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        document.getElementById('audio-indicator').textContent = 'Disconnected';
        document.getElementById('audio-indicator').classList.remove('active');
        stopStatePolling();
    });

    socket.on('audio', (data) => {
        if (audioPlaying && audioContext) {
            playAudioChunk(data);
        }
    });
}

// Poll for state updates via REST API
async function pollState() {
    try {
        const response = await fetch('/api/state');
        if (response.ok) {
            const state = await response.json();
            updateUI(state);
        }
    } catch (e) {
        console.error('Error polling state:', e);
    }
}

function startStatePolling() {
    if (statePollingInterval) return;
    // Poll at 10 Hz
    statePollingInterval = setInterval(pollState, 100);
    // Also poll immediately
    pollState();
}

function stopStatePolling() {
    if (statePollingInterval) {
        clearInterval(statePollingInterval);
        statePollingInterval = null;
    }
}

// Update UI with state data
function updateUI(state) {
    // Frequency (only update if not focused, to avoid interrupting user input)
    const freqInput = document.getElementById('frequency');
    if (document.activeElement !== freqInput) {
        freqInput.value = state.frequency_mhz.toFixed(1);
    }

    // Band marker position (88-108 MHz)
    const bandPos = ((state.frequency_mhz - 88) / 20) * 100;
    document.getElementById('band-marker').style.left = bandPos + '%';

    // Signal strength
    document.getElementById('s-meter').textContent = state.s_meter;
    document.getElementById('signal-dbm').textContent = `(${state.signal_dbm.toFixed(1)} dBm)`;

    // S-meter bar - aligned with visual scale (S1=0%, S9=80%, S9+20=100%)
    const S1_DBM = -141;
    const S9_DBM = -93;
    const S9_PLUS_20 = -73;
    let barPercent;
    if (state.signal_dbm <= S9_DBM) {
        // S1 to S9: maps to 0-80% of bar
        barPercent = ((state.signal_dbm - S1_DBM) / (S9_DBM - S1_DBM)) * 80;
    } else {
        // S9 to S9+20: maps to 80-100% of bar
        barPercent = 80 + ((state.signal_dbm - S9_DBM) / (S9_PLUS_20 - S9_DBM)) * 20;
    }
    barPercent = Math.max(0, Math.min(100, barPercent));
    document.getElementById('s-meter-fill').style.width = barPercent + '%';

    // SNR
    const snr = state.snr_db;
    document.getElementById('snr').textContent = snr.toFixed(1) + ' dB';
    let quality = 'Very Poor';
    let qualityColor = '#ff4444';
    if (snr > 40) { quality = 'Excellent'; qualityColor = '#00ff88'; }
    else if (snr > 30) { quality = 'Good'; qualityColor = '#00ff88'; }
    else if (snr > 20) { quality = 'Fair'; qualityColor = '#ffdd00'; }
    else if (snr > 10) { quality = 'Poor'; qualityColor = '#ffdd00'; }
    document.getElementById('snr-quality').textContent = `(${quality})`;
    document.getElementById('snr-quality').style.color = qualityColor;

    // Audio mode (stereo/mono)
    if (state.pilot_detected) {
        const blend = state.stereo_blend;
        if (blend >= 0.99) {
            document.getElementById('audio-mode').textContent = 'Stereo';
            document.getElementById('audio-detail').textContent = '(19 kHz pilot detected)';
        } else if (blend <= 0.01) {
            document.getElementById('audio-mode').textContent = 'Mono';
            document.getElementById('audio-detail').textContent = '(blended - low SNR)';
        } else {
            document.getElementById('audio-mode').textContent = `Blend ${Math.round(blend * 100)}%`;
            document.getElementById('audio-detail').textContent = '(reduced stereo for noise)';
        }
    } else {
        document.getElementById('audio-mode').textContent = 'Mono';
        document.getElementById('audio-detail').textContent = '(no pilot)';
    }

    // Squelch
    const squelchBtn = document.getElementById('btn-squelch');
    if (state.squelch_enabled) {
        document.getElementById('squelch-status').textContent = `ON @ ${state.squelch_threshold} dBm`;
        squelchBtn.classList.add('active');
        document.getElementById('muted-indicator').style.display = state.is_squelched ? 'inline' : 'none';
    } else {
        document.getElementById('squelch-status').textContent = 'OFF';
        squelchBtn.classList.remove('active');
        document.getElementById('muted-indicator').style.display = 'none';
    }

    // RDS
    const rdsBtn = document.getElementById('btn-rds');
    if (state.rds_enabled) {
        rdsBtn.classList.add('active');
        const rds = state.rds;
        if (rds.synced) {
            document.getElementById('rds-status').textContent = 'ON [SYNC]';
        } else {
            document.getElementById('rds-status').textContent = 'ON [SRCH]';
        }

        // Station info
        if (rds.callsign) {
            document.getElementById('station-callsign').textContent = rds.callsign;
        } else if (rds.pi_hex) {
            document.getElementById('station-callsign').textContent = 'PI:' + rds.pi_hex;
        } else {
            document.getElementById('station-callsign').textContent = '';
        }

        if (rds.program_type && rds.program_type !== 'None') {
            document.getElementById('station-pty').textContent = `(${rds.program_type})`;
        } else {
            document.getElementById('station-pty').textContent = '';
        }

        document.getElementById('station-name').textContent = rds.station_name || '';
        document.getElementById('radio-text').textContent = rds.radio_text || '';
    } else {
        rdsBtn.classList.remove('active');
        document.getElementById('rds-status').textContent = 'OFF';
        document.getElementById('station-callsign').textContent = '';
        document.getElementById('station-pty').textContent = '';
        document.getElementById('station-name').textContent = '';
        document.getElementById('radio-text').textContent = '';
    }

    // Bass/Treble
    const bassBtn = document.getElementById('btn-bass');
    const trebleBtn = document.getElementById('btn-treble');
    if (state.bass_boost) {
        bassBtn.classList.add('active');
    } else {
        bassBtn.classList.remove('active');
    }
    if (state.treble_boost) {
        trebleBtn.classList.add('active');
    } else {
        trebleBtn.classList.remove('active');
    }

    // Error
    const errorDiv = document.getElementById('error-display');
    if (state.error) {
        errorDiv.textContent = state.error;
        errorDiv.style.display = 'block';
    } else {
        errorDiv.style.display = 'none';
    }

    // Spectrum analyzer
    if (state.spectrum) {
        drawSpectrum(state.spectrum.levels, state.spectrum.peaks);
    }
}

// Draw spectrum analyzer on canvas
function drawSpectrum(levels, peaks) {
    const canvas = document.getElementById('spectrum-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const numBands = levels.length;
    const barWidth = Math.floor((width - (numBands - 1) * 2) / numBands);
    const gap = 2;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw bars
    for (let i = 0; i < numBands; i++) {
        const x = i * (barWidth + gap);
        const level = levels[i];
        const peak = peaks[i];
        const barHeight = Math.floor(level * height);
        const peakY = Math.floor((1 - peak) * height);

        // Color gradient based on height (dark blue -> blue -> cyan)
        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#0044aa');
        gradient.addColorStop(0.5, '#0088ff');
        gradient.addColorStop(1, '#00ccff');

        // Draw bar
        ctx.fillStyle = gradient;
        ctx.fillRect(x, height - barHeight, barWidth, barHeight);

        // Draw peak indicator
        if (peak > 0.01) {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(x, peakY, barWidth, 2);
        }
    }
}

// Play audio chunk received from WebSocket
function playAudioChunk(data) {
    // Decode base64 audio data
    const binaryString = atob(data.data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    if (data.codec === 'opus') {
        // Decode Opus frame
        if (opusDecoder && opusDecoderReady) {
            try {
                const decoded = opusDecoder.decodeFrame(bytes);
                if (decoded && decoded.channelData && decoded.channelData.length > 0) {
                    playOpusAudio(decoded);
                }
            } catch (e) {
                console.error('Opus decode error:', e);
            }
        } else {
            console.warn('Opus decoder not ready, dropping frame');
        }
    } else {
        // PCM fallback
        playPcmAudio(bytes, data.channels, data.sample_rate);
    }
}

// Play decoded PCM audio
function playPcmAudio(bytes, channels, sampleRate) {
    const int16View = new Int16Array(bytes.buffer);
    const numSamples = int16View.length / channels;

    // Create audio buffer
    const audioBuffer = audioContext.createBuffer(channels, numSamples, sampleRate);

    for (let channel = 0; channel < channels; channel++) {
        const channelData = audioBuffer.getChannelData(channel);
        for (let i = 0; i < numSamples; i++) {
            channelData[i] = int16View[i * channels + channel] / 32768;
        }
    }

    scheduleAudioBuffer(audioBuffer);
}

// Play decoded Opus audio (called by decoder callback)
function playOpusAudio(decodedData) {
    // Check what format the decoded data is in
    if (!decodedData) {
        console.error('playOpusAudio: no decoded data');
        return;
    }

    let audioBuffer;

    // Handle different possible decoded data formats
    if (decodedData.channelData) {
        // Format: { channelData: [Float32Array, Float32Array], samplesDecoded: N, sampleRate: N }
        const numSamples = decodedData.samplesDecoded || decodedData.channelData[0].length;
        const sampleRate = decodedData.sampleRate || SAMPLE_RATE;
        audioBuffer = audioContext.createBuffer(decodedData.channelData.length, numSamples, sampleRate);
        for (let ch = 0; ch < decodedData.channelData.length; ch++) {
            audioBuffer.getChannelData(ch).set(decodedData.channelData[ch]);
        }
    } else if (decodedData.left && decodedData.right) {
        // Format: { left: Float32Array, right: Float32Array }
        const numSamples = decodedData.left.length;
        audioBuffer = audioContext.createBuffer(2, numSamples, SAMPLE_RATE);
        audioBuffer.getChannelData(0).set(decodedData.left);
        audioBuffer.getChannelData(1).set(decodedData.right);
    } else if (decodedData instanceof Float32Array) {
        // Mono or interleaved
        const numSamples = decodedData.length / 2;
        audioBuffer = audioContext.createBuffer(2, numSamples, SAMPLE_RATE);
        for (let i = 0; i < numSamples; i++) {
            audioBuffer.getChannelData(0)[i] = decodedData[i * 2];
            audioBuffer.getChannelData(1)[i] = decodedData[i * 2 + 1];
        }
    } else {
        console.error('playOpusAudio: unknown data format', decodedData);
        return;
    }

    scheduleAudioBuffer(audioBuffer);
}

// Schedule audio buffer for playback
function scheduleAudioBuffer(audioBuffer) {
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    // Calculate start time
    const currentTime = audioContext.currentTime;
    if (nextPlayTime < currentTime) {
        // Buffer underrun - restart with small delay
        nextPlayTime = currentTime + 0.05;
    }

    source.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;
}

// Start audio playback
function startAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE
        });
    }

    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    // Initialize Opus decoder if available
    // The library exports to window["opus-decoder"].OpusDecoder
    const OpusDecoderClass = window["opus-decoder"]?.OpusDecoder || window.OpusDecoder;
    if (OpusDecoderClass && !opusDecoder) {
        console.log('Creating Opus decoder...');
        opusDecoder = new OpusDecoderClass({
            channels: 2,
            sampleRate: SAMPLE_RATE
        });
        opusDecoder.ready.then(() => {
            opusDecoderReady = true;
            console.log('Opus decoder ready');
        }).catch((err) => {
            console.error('Opus decoder failed to initialize:', err);
        });
    } else if (!OpusDecoderClass) {
        console.warn('OpusDecoder not available, will use PCM fallback');
    }

    audioPlaying = true;
    nextPlayTime = audioContext.currentTime + 0.1;

    document.getElementById('btn-audio').textContent = 'Stop Audio';
    document.getElementById('btn-audio').classList.add('playing');
    document.getElementById('audio-indicator').textContent = 'Audio playing';
    document.getElementById('audio-indicator').classList.add('active');
}

// Stop audio playback
function stopAudio() {
    audioPlaying = false;

    // Reset Opus decoder for clean state on next start
    if (opusDecoder) {
        try {
            opusDecoder.free();
        } catch (e) {
            // Ignore cleanup errors
        }
        opusDecoder = null;
        opusDecoderReady = false;
    }

    document.getElementById('btn-audio').textContent = 'Start Audio';
    document.getElementById('btn-audio').classList.remove('playing');
    document.getElementById('audio-indicator').textContent = 'Audio stopped';
    document.getElementById('audio-indicator').classList.remove('active');
}

// Toggle audio
function toggleAudio() {
    if (audioPlaying) {
        stopAudio();
    } else {
        startAudio();
    }
}

// API calls
async function tuneUp() {
    await fetch('/api/tune/up', { method: 'POST' });
}

async function tuneDown() {
    await fetch('/api/tune/down', { method: 'POST' });
}

async function setVolume(vol) {
    await fetch('/api/volume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ volume: vol / 100 })
    });
}

async function toggleSquelch() {
    await fetch('/api/squelch/toggle', { method: 'POST' });
}

async function toggleRDS() {
    await fetch('/api/rds/toggle', { method: 'POST' });
}

async function toggleBass() {
    await fetch('/api/bass/toggle', { method: 'POST' });
}

async function toggleTreble() {
    await fetch('/api/treble/toggle', { method: 'POST' });
}

async function tuneToFrequency(freqMhz) {
    const freq = parseFloat(freqMhz);
    if (isNaN(freq) || freq < 88.0 || freq > 108.0) {
        // Invalid frequency - reset to current
        pollState();
        return;
    }
    await fetch('/api/tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frequency: freq })
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    connect();

    // Tune buttons
    document.getElementById('tune-up').addEventListener('click', tuneUp);
    document.getElementById('tune-down').addEventListener('click', tuneDown);

    // Frequency input - direct entry
    const freqInput = document.getElementById('frequency');
    freqInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            tuneToFrequency(freqInput.value);
            freqInput.blur();
        } else if (e.key === 'Escape') {
            freqInput.blur();
            pollState(); // Reset to current frequency
        }
    });
    freqInput.addEventListener('blur', () => {
        tuneToFrequency(freqInput.value);
    });
    // Select all text on focus for easy replacement
    freqInput.addEventListener('focus', () => {
        freqInput.select();
    });

    // Mouse wheel tuning over frequency display
    freqInput.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.deltaY < 0) {
            tuneUp();
        } else if (e.deltaY > 0) {
            tuneDown();
        }
    }, { passive: false });

    // Control buttons
    document.getElementById('btn-squelch').addEventListener('click', toggleSquelch);
    document.getElementById('btn-rds').addEventListener('click', toggleRDS);
    document.getElementById('btn-bass').addEventListener('click', toggleBass);
    document.getElementById('btn-treble').addEventListener('click', toggleTreble);

    // Audio button
    document.getElementById('btn-audio').addEventListener('click', toggleAudio);

    // Volume slider
    const volumeSlider = document.getElementById('volume-slider');
    volumeSlider.addEventListener('input', (e) => {
        document.getElementById('volume-value').textContent = e.target.value + '%';
        setVolume(parseInt(e.target.value));
    });

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        // Don't handle if focus is on an input
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case 'ArrowLeft':
                tuneDown();
                e.preventDefault();
                break;
            case 'ArrowRight':
                tuneUp();
                e.preventDefault();
                break;
            case 'ArrowUp':
                volumeSlider.value = Math.min(100, parseInt(volumeSlider.value) + 5);
                volumeSlider.dispatchEvent(new Event('input'));
                e.preventDefault();
                break;
            case 'ArrowDown':
                volumeSlider.value = Math.max(0, parseInt(volumeSlider.value) - 5);
                volumeSlider.dispatchEvent(new Event('input'));
                e.preventDefault();
                break;
            case 'q':
            case 'Q':
                toggleSquelch();
                break;
            case 'r':
            case 'R':
                toggleRDS();
                break;
            case 'b':
            case 'B':
                toggleBass();
                break;
            case 't':
            case 'T':
                toggleTreble();
                break;
            case ' ':
                toggleAudio();
                e.preventDefault();
                break;
        }
    });
});

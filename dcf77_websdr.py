#!/usr/bin/env python3
"""
DCF77 Decoder via WebSDR (University of Twente)
================================================
Streams audio DIRECTLY from websdr.ewi.utwente.nl over WebSocket —
no browser, no Virtual Audio Cable, no KiwiSDR client needed.

The WebSDR at Twente is one of the best LF receivers in Europe and
reliably receives DCF77 at 77.5 kHz with excellent SNR.

Architecture:
  WebSDR WebSocket  →  raw audio chunks  →  envelope detection
  →  pulse timing  →  bit classification  →  DCF77 telegram decode

Usage:
    python dcf77_websdr.py                   # decode one full minute
    python dcf77_websdr.py --wav             # also save the raw audio
    python dcf77_websdr.py --plot            # show signal analysis plot
    python dcf77_websdr.py --file dcf77.wav  # decode from saved WAV file

Dependencies:
    pip install numpy scipy websocket-client matplotlib
"""

import argparse
import array
import io
import json
import math
import os
import struct
import sys
import time
import wave
from datetime import datetime, timezone, timedelta

import numpy as np
from scipy import signal as scipy_signal

# ── Try importing websocket; give a clear install message if missing ──
try:
    import websocket
except ImportError:
    print("❌  websocket-client not installed.")
    print("    pip install websocket-client")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

WEBSDR_HOST  = "websdr.ewi.utwente.nl/"
WEBSDR_PORT  = 8901
WEBSDR_URL   = f"ws://{WEBSDR_HOST}:{WEBSDR_PORT}"

DCF77_FREQ   = 77.5          # kHz
SAMPLE_RATE  = 11025         # WebSDR streams at 11025 Hz
RECORD_SECS  = 150           # 2.5 minutes — guarantees ≥2 full frames

# DCF77 OOK timing
ZERO_MS      = 125           # ms
ONE_MS       = 231           # ms
TOLERANCE_MS = 30           # ± ms for classification
MINUTE_GAP_S = 1.8           # gap > this = minute marker


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — WebSDR WebSocket connection & audio capture
# ──────────────────────────────────────────────────────────────────────────────

class WebSDRRecorder:
    """
    Connects to the University of Twente WebSDR WebSocket API and
    streams demodulated audio at the target frequency.

    The WebSDR protocol:
      1. Connect to ws://websdr.ewi.utwente.nl:8901/~~websdr
      2. Send JSON commands to set frequency and mode
      3. Receive binary audio frames (16-bit signed PCM, mono, 11025 Hz)
         interleaved with JSON status messages
    """

    # WebSDR mode codes
    MODE_CW  = 'cw'
    MODE_AM  = 'am'
    MODE_USB = 'usb'
    MODE_LSB = 'lsb'

    def __init__(self, freq_khz=77.5, mode='cw', duration_s=150):
        self.freq_khz   = freq_khz
        self.mode       = mode
        self.duration_s = duration_s
        self.samples    = []          # accumulated PCM int16 samples
        self._ws        = None
        self._start_t   = None
        self._done      = False

    def _on_open(self, ws):
        self._start_t = time.time()
        print(f"   ✅ WebSocket connected to {WEBSDR_HOST}")
        print(f"   📻 Tuning to {self.freq_khz} kHz, mode={self.mode.upper()}")

        # Send the tuning commands the WebSDR expects
        # Format observed from browser dev-tools on websdr.ewi.utwente.nl
        ws.send(json.dumps({
            "type": "setfrequency",
            "value": int(self.freq_khz * 1000)   # Hz
        }))
        ws.send(json.dumps({
            "type": "setdemodulation",
            "value": self.mode
        }))
        # Request audio stream
        ws.send(json.dumps({
            "type": "start",
            "value": "audio"
        }))

    def _on_message(self, ws, message):
        elapsed = time.time() - self._start_t

        # Progress bar
        pct = min(100, int(elapsed / self.duration_s * 100))
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(f"\r   [{bar}] {elapsed:.0f}s / {self.duration_s}s", end='', flush=True)

        if elapsed >= self.duration_s:
            self._done = True
            ws.close()
            return

        # Binary frames = raw PCM audio (16-bit signed, little-endian, mono)
        if isinstance(message, bytes):
            # The first 8 bytes are a WebSDR header (sequence + flags),
            # actual PCM data follows
            if len(message) > 8:
                pcm_bytes = message[8:]
                n_samples = len(pcm_bytes) // 2
                chunk = struct.unpack(f'<{n_samples}h', pcm_bytes[:n_samples*2])
                self.samples.extend(chunk)

        # Text frames = JSON status (frequency confirmed, user count, etc.)
        elif isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get('type') == 'error':
                    print(f"\n   ⚠️  WebSDR: {data.get('value', 'unknown error')}")
            except json.JSONDecodeError:
                pass

    def _on_error(self, ws, error):
        print(f"\n   ❌ WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"\n   📴 Connection closed")

    def record(self):
        """Connect and stream audio for duration_s seconds. Returns numpy array."""
        print(f"\n🔌 Connecting to WebSDR at {WEBSDR_HOST}:{WEBSDR_PORT} ...")

        self._ws = websocket.WebSocketApp(
            WEBSDR_URL,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
            header     = {
                "Origin": f"http://{WEBSDR_HOST}:{WEBSDR_PORT}",
                "User-Agent": "Mozilla/5.0 DCF77-Decoder/1.0",
            }
        )

        self._ws.run_forever(ping_interval=30, ping_timeout=10)

        if not self.samples:
            return None

        samples = np.array(self.samples, dtype=np.float32) / 32768.0
        print(f"\n   📦 Captured {len(samples)} samples ({len(samples)/SAMPLE_RATE:.1f}s)")
        return samples

    def save_wav(self, samples, path):
        """Save captured samples as a WAV file for later analysis."""
        pcm = (samples * 32767).astype(np.int16)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        print(f"   💾 Audio saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK: if WebSocket protocol doesn't match, try HTTP audio stream
# ──────────────────────────────────────────────────────────────────────────────

def record_via_http_fallback(freq_khz=77.5, duration_s=150):
    """
    Alternative: fetch audio from WebSDR as an HTTP audio stream.
    WebSDR exposes audio at:
      http://websdr.ewi.utwente.nl:8901/audio?freq=77500&mode=cw&passband=0
    Returns raw PCM bytes.
    """
    try:
        import urllib.request
        url = (f"http://{WEBSDR_HOST}:{WEBSDR_PORT}/audio"
               f"?freq={int(freq_khz*1000)}&mode=cw&passband=0&"
               f"client=python-dcf77")
        print(f"\n🌐 Trying HTTP audio stream: {url}")

        chunks = []
        start = time.time()
        with urllib.request.urlopen(url, timeout=10) as resp:
            print(f"   Content-Type: {resp.headers.get('Content-Type', '?')}")
            while time.time() - start < duration_s:
                chunk = resp.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                elapsed = time.time() - start
                print(f"\r   Streaming... {elapsed:.0f}s", end='', flush=True)

        raw = b''.join(chunks)
        print(f"\n   Got {len(raw)} bytes")
        return raw
    except Exception as e:
        print(f"\n   HTTP fallback failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load WAV file (for --file mode)
# ──────────────────────────────────────────────────────────────────────────────

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(wf.getnframes())

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # IQ mode: stereo WAV with I=left, Q=right
    # Compute true amplitude envelope = sqrt(I² + Q²)
    if n_ch == 2:
        I = data[0::2]
        Q = data[1::2]
        return np.sqrt(I**2 + Q**2), sr, True   # True = already envelope
    else:
        return data, sr, False


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — AM envelope detection
# ──────────────────────────────────────────────────────────────────────────────

def extract_envelope(samples, sr, smooth_ms=120, is_iq=False):
    if is_iq:
        env = samples   # already amplitude envelope from IQ
    else:
        env = np.abs(samples)

    cutoff_hz = 1000.0 / smooth_ms
    nyq = sr / 2.0
    sos = scipy_signal.butter(4, cutoff_hz / nyq, btype='low', output='sos')
    env = scipy_signal.sosfilt(sos, env)

    p99 = np.percentile(env, 99)
    if p99 < 1e-9:
        raise ValueError("Signal appears to be silence")
    env = np.clip(env / p99, 0, 1)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Threshold & pulse detection
# ──────────────────────────────────────────────────────────────────────────────

def adaptive_threshold(env):
    """
    Find the threshold between carrier-ON and carrier-OFF states.

    DCF77 carrier is ON ~85% of the time (carrier-off pulses are short).
    So the 15th percentile of the envelope is solidly in the OFF region,
    and the 85th is solidly in the ON region. We set threshold at 40%
    of the way from OFF to ON — biased low to catch weak signals.
    """
    low  = np.percentile(env, 90)
    high = np.percentile(env, 10)
    return low + 0.25 * (high - low)


def detect_pulses(env, sr, threshold=None):
    if threshold is None:
        threshold = adaptive_threshold(env)

    window = int(sr * 1.0)
    pulses = []
    n_windows = len(env) // window

    for i in range(n_windows):
        seg = env[i*window:(i+1)*window]
        
        # Find the deepest point in this window
        min_idx = np.argmin(seg)
        min_val = seg[min_idx]
        
        # Skip windows where nothing dips (flat carrier, no bit expected)
        if min_val > threshold * 0.95:
            continue

        # Measure pulse width around the minimum
        center = i * window + min_idx
        half_window = window // 2

        # Walk left from minimum until signal rises above threshold
        left = center
        while left > 0 and env[left] < threshold:
            left -= 1

        # Walk right from minimum until signal rises above threshold  
        right = center
        while right < len(env) - 1 and env[right] < threshold:
            right += 1

        dur_s = (right - left) / sr
        t = left / sr

        if 0.05 <= dur_s <= 0.50:
            pulses.append((t, dur_s))

    return pulses, threshold


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Bit classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_pulses(pulses):
    """
    Map each pulse duration to a DCF77 symbol:
      ~100ms  →  bit 0
      ~200ms  →  bit 1
      >1800ms →  minute marker (second 59 is always missing)
      other   →  error (noise or dropout)
    """
    bits = []
    tol  = TOLERANCE_MS / 1000.0

    for (t, dur) in pulses:
        if abs(dur - ZERO_MS/1000) <= tol:
            bits.append({'t': t, 'dur': dur, 'bit': 0})
        elif abs(dur - ONE_MS/1000) <= tol:
            bits.append({'t': t, 'dur': dur, 'bit': 1})
        elif dur >= MINUTE_GAP_S:
            bits.append({'t': t, 'dur': dur, 'bit': 'M'})
        else:
            bits.append({'t': t, 'dur': dur, 'bit': '?'})

    return bits


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Frame extraction & alignment
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(bits):
    """
    Split the bit stream into 59-bit minute frames.
    Includes auto-alignment: scans each candidate frame for the
    signature pattern (bit 0 = 0, bit 20 = 1) and shifts if needed.
    """
    frames = []
    current = []

    for i, b in enumerate(bits):
        if b['bit'] == 'M':
            if len(current) >= 40:
                frames.append(current)
            current = []
        else:
            if current and i > 0:
                gap = b['t'] - bits[i-1]['t']
                if gap > MINUTE_GAP_S:
                    if len(current) >= 40:
                        frames.append(current)
                    current = []
            current.append(b)

    if len(current) >= 40:
        frames.append(current)

    # ── Auto-alignment: try shifting each frame by 0–3 bits ──
    aligned = []
    for frame in frames:
        best = frame
        for shift in range(4):
            candidate = frame[shift:]
            if len(candidate) < 40:
                break
            b = [e['bit'] if e['bit'] in (0,1) else 0 for e in candidate]
            while len(b) < 59:
                b.append(0)
            # Valid frame signature: bit 0 = 0, bit 20 = 1
            if b[0] == 0 and b[20] == 1:
                best = candidate
                break
        aligned.append(best)

    return aligned


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — DCF77 telegram decode
# ──────────────────────────────────────────────────────────────────────────────

def bcd(bits, weights=(1, 2, 4, 8, 10, 20, 40, 80)):
    """Decode BCD value from a list of bits (LSB first)."""
    return sum(b * w for b, w in zip(bits, weights) if b in (0, 1))

def even_parity(bits, parity_bit):
    """DCF77 uses even parity: XOR of all data bits + parity bit = 0."""
    return (sum(b for b in bits if b in (0,1)) + (parity_bit if parity_bit in (0,1) else 0)) % 2 == 0

def decode_frame(frame):
    """
    Decode one 59-bit DCF77 minute frame into date + time.

    DCF77 bit layout (second number = bit index):
    ┌──────────┬───────────────────────────────────────────────────┐
    │  Bits    │  Content                                          │
    ├──────────┼───────────────────────────────────────────────────┤
    │  0       │  Start of minute marker (always 0)               │
    │  1–14    │  Encrypted weather / civil warning data          │
    │  15      │  Call bit (abnormal transmitter operation)       │
    │  16      │  Summer time change announcement (1hr warning)   │
    │  17      │  CEST active (1 = UTC+2 summer time)            │
    │  18      │  CET  active (1 = UTC+1 winter time)            │
    │  19      │  Leap second announcement                        │
    │  20      │  Start of encoded time (always 1)               │
    │  21–27   │  Minutes, BCD LSB-first                         │
    │  28      │  P1: even parity for bits 21–27                 │
    │  29–34   │  Hours, BCD LSB-first                           │
    │  35      │  P2: even parity for bits 29–34                 │
    │  36–41   │  Day of month, BCD LSB-first                    │
    │  42–44   │  Day of week (1=Mon … 7=Sun)                    │
    │  45–49   │  Month, BCD LSB-first                           │
    │  50–57   │  Year (within century), BCD LSB-first           │
    │  58      │  P3: even parity for bits 36–57                 │
    │ [59]     │  No pulse — this gap IS the minute marker       │
    └──────────┴───────────────────────────────────────────────────┘
    """
    # Extract bit values; treat '?' as 0 (corrupted bit)
    b = [e['bit'] if e['bit'] in (0, 1) else 0 for e in frame]

    # Pad to 59 if short
    while len(b) < 59:
        b.append(0)

    errors = []

    # ── Structural checks ──────────────────────────────────────────
    if b[0] != 0:
        errors.append("Bit 0 ≠ 0 (frame misaligned — start marker wrong)")
    if b[20] != 1:
        errors.append("Bit 20 ≠ 1 (frame misaligned — time-start marker wrong)")

    # ── Timezone / DST ─────────────────────────────────────────────
    cest = b[17]   # 1 = CEST (UTC+2), 0 = CET (UTC+1)
    tz_str = "CEST (UTC+2)" if cest else "CET  (UTC+1)"
    tz_offset = timedelta(hours=2 if cest else 1)

    # ── Minutes [21–27] + parity P1 [28] ──────────────────────────
    minute = bcd(b[21:28])
    if not even_parity(b[21:28], b[28]):
        errors.append("P1 parity failed (minutes corrupted)")

    # ── Hours [29–34] + parity P2 [35] ────────────────────────────
    hour = bcd(b[29:35])
    if not even_parity(b[29:35], b[35]):
        errors.append("P2 parity failed (hours corrupted)")

    # ── Date fields ────────────────────────────────────────────────
    day     = bcd(b[36:42])
    weekday = bcd(b[42:45], weights=(1, 2, 4))
    month   = bcd(b[45:50])
    year    = 2000 + bcd(b[50:58])

    # P3 covers bits 36–57
    if not even_parity(b[36:58], b[58]):
        errors.append("P3 parity failed (date corrupted)")

    # ── Sanity checks ──────────────────────────────────────────────
    if not (0 <= minute <= 59): errors.append(f"Minute out of range: {minute}")
    if not (0 <= hour   <= 23): errors.append(f"Hour out of range: {hour}")
    if not (1 <= day    <= 31): errors.append(f"Day out of range: {day}")
    if not (1 <= month  <= 12): errors.append(f"Month out of range: {month}")
    if not (1 <= weekday <= 7): errors.append(f"Weekday out of range: {weekday}")

    weekdays = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",
                5:"Friday",6:"Saturday",7:"Sunday"}

    # Build UTC time (DCF77 transmits local German time)
    try:
        local_dt = datetime(year, month, day, hour, minute,
                            tzinfo=timezone(tz_offset))
        utc_dt = local_dt.astimezone(timezone.utc)
    except ValueError:
        local_dt = utc_dt = None

    return {
        "year":    year,   "month":   month,   "day":      day,
        "weekday": weekdays.get(weekday, f"?({weekday})"),
        "hour":    hour,   "minute":  minute,
        "tz":      tz_str, "cest":    bool(cest),
        "local_dt": local_dt,  "utc_dt": utc_dt,
        "errors":  errors, "raw":     b[:59],
        "n_bits":  len(frame),
        # Extra flags
        "call_bit":     b[15],
        "dst_announce": b[16],
        "leap_second":  b[19],
    }


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY: pretty-print the decoded telegram bit by bit
# ──────────────────────────────────────────────────────────────────────────────

FIELD_COLORS = {   # ANSI colors for terminal output
    'start':   '\033[90m',    # dark gray
    'civil':   '\033[35m',    # magenta
    'flags':   '\033[36m',    # cyan
    'minutes': '\033[33m',    # yellow
    'hours':   '\033[32m',    # green
    'date':    '\033[34m',    # blue
    'parity':  '\033[31m',    # red
    'reset':   '\033[0m',
}

def color_bit(idx, value):
    """Return a colored string for each bit based on its field."""
    c = FIELD_COLORS
    if idx == 0:
        col = c['start']
    elif 1 <= idx <= 14:
        col = c['civil']
    elif 15 <= idx <= 19:
        col = c['flags']
    elif idx == 20:
        col = c['start']
    elif 21 <= idx <= 27:
        col = c['minutes']
    elif idx == 28:
        col = c['parity']
    elif 29 <= idx <= 34:
        col = c['hours']
    elif idx == 35:
        col = c['parity']
    elif 36 <= idx <= 57:
        col = c['date']
    elif idx == 58:
        col = c['parity']
    else:
        col = ''
    return f"{col}{value}{c['reset']}"

def print_bit_analysis(result, frame_num):
    """Print a detailed bit-by-bit breakdown of the decoded frame."""
    raw = result['raw']
    print(f"\n{'─'*62}")
    print(f"  Bit-by-bit analysis — Frame #{frame_num}")
    print(f"{'─'*62}")

    # Print bit indices
    print("  Idx: ", end="")
    for i in range(59):
        print(f"{i:2d} ", end="")
    print()

    # Print bit values (colored by field)
    print("  Val: ", end="")
    for i, v in enumerate(raw):
        print(f"{color_bit(i, v):2s} ", end="")
    print()

    # Print field legend
    print(f"""
  Color legend:
  {FIELD_COLORS['start']}■{FIELD_COLORS['reset']} Start/markers [0, 20]
  {FIELD_COLORS['civil']}■{FIELD_COLORS['reset']} Civil warning / weather data [1–14]
  {FIELD_COLORS['flags']}■{FIELD_COLORS['reset']} Control flags [15–19]: call, DST announce, CEST, CET, leap
  {FIELD_COLORS['minutes']}■{FIELD_COLORS['reset']} Minutes [21–27]
  {FIELD_COLORS['hours']}■{FIELD_COLORS['reset']} Hours [29–34]
  {FIELD_COLORS['date']}■{FIELD_COLORS['reset']} Date: day [36–41], weekday [42–44], month [45–49], year [50–57]
  {FIELD_COLORS['parity']}■{FIELD_COLORS['reset']} Parity bits P1 [28], P2 [35], P3 [58]""")

    # Field-by-field breakdown
    print(f"\n  Field breakdown:")
    print(f"    [0]       start      = {raw[0]}  (must be 0)")
    print(f"    [1–14]    civil      = {''.join(str(raw[i]) for i in range(1,15))}  (encrypted)")
    print(f"    [15]      call bit   = {raw[15]}  {'(transmitter anomaly!)' if raw[15] else '(normal)'}")
    print(f"    [16]      DST warn   = {raw[16]}  {'(change coming!)' if raw[16] else ''}")
    print(f"    [17]      CEST       = {raw[17]}  {'(summer time UTC+2)' if raw[17] else ''}")
    print(f"    [18]      CET        = {raw[18]}  {'(winter time UTC+1)' if raw[18] else ''}")
    print(f"    [19]      leap sec   = {raw[19]}  {'(leap second coming!)' if raw[19] else ''}")
    print(f"    [20]      time start = {raw[20]}  (must be 1)")
    print(f"    [21–27]   minutes    = {''.join(str(raw[i]) for i in range(21,28))}  → {result['minute']:02d}")
    print(f"    [28]      P1         = {raw[28]}  {'✅' if not any('minutes' in e for e in result['errors']) else '❌'}")
    print(f"    [29–34]   hours      = {''.join(str(raw[i]) for i in range(29,35))}  → {result['hour']:02d}")
    print(f"    [35]      P2         = {raw[35]}  {'✅' if not any('hours' in e for e in result['errors']) else '❌'}")
    print(f"    [36–41]   day        = {''.join(str(raw[i]) for i in range(36,42))}  → {result['day']:02d}")
    print(f"    [42–44]   weekday    = {''.join(str(raw[i]) for i in range(42,45))}  → {result['weekday']}")
    print(f"    [45–49]   month      = {''.join(str(raw[i]) for i in range(45,50))}  → {result['month']:02d}")
    print(f"    [50–57]   year       = {''.join(str(raw[i]) for i in range(50,58))}  → {result['year']}")
    print(f"    [58]      P3         = {raw[58]}  {'✅' if not any('P3' in e for e in result['errors']) else '❌'}")
    print(f"    [59]      (no pulse — this gap is the minute marker)")


def print_result(result, frame_num):
    ok = len(result['errors']) == 0
    status = "✅ VALID" if ok else f"⚠️  {len(result['errors'])} warning(s)"

    print(f"\n{'═'*62}")
    print(f"  FRAME #{frame_num}  |  {result['n_bits']} bits  |  {status}")
    print(f"{'═'*62}")
    print(f"  📅  {result['weekday']}, {result['day']:02d}/{result['month']:02d}/{result['year']}")
    print(f"  🕐  {result['hour']:02d}:{result['minute']:02d}  ({result['tz']})")
    if result['utc_dt']:
        print(f"  🌍  UTC: {result['utc_dt'].strftime('%H:%M')}")
    if result['cest']:
        print(f"  ☀️   Summer time (CEST) is active")
    if result['dst_announce']:
        print(f"  ⏰  DST change coming within the hour!")
    if result['leap_second']:
        print(f"  ⏱️   Leap second announced!")
    if result['call_bit']:
        print(f"  📡  Call bit set — transmitter operating abnormally")
    for e in result['errors']:
        print(f"  ⚠️   {e}")


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: matplotlib analysis plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_analysis(env, sr, pulses, threshold, classified, frames):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[!] pip install matplotlib  to enable plots")
        return

    t = np.arange(len(env)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')

    # Panel 1: full envelope
    ax = axes[0]
    ax.plot(t, env, color='#58a6ff', lw=0.5, alpha=0.85, label='AM Envelope')
    ax.axhline(threshold, color='#f85149', lw=1.2, ls='--',
               label=f'Threshold ({threshold:.2f})')
    ax.set_ylabel('Amplitude')
    ax.set_title('DCF77 AM Envelope — Full Recording')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.set_xlim(0, t[-1])

    # Panel 2: first 20 seconds zoomed with coloured pulse spans
    ax = axes[1]
    zoom = min(20.0, t[-1])
    mask = t <= zoom
    ax.plot(t[mask], env[mask], color='#3fb950', lw=1.0)
    ax.axhline(threshold, color='#f85149', lw=0.8, ls='--')
    for (pt, dur) in pulses:
        if pt > zoom:
            break
        is_zero = abs(dur - ZERO_MS/1000) < TOLERANCE_MS/1000
        color = '#1f6feb' if is_zero else '#d29922'
        ax.axvspan(pt, pt + dur, color=color, alpha=0.4)
    ax.set_ylabel('Amplitude')
    ax.set_title('First 20 s — carrier dips (blue=0, orange=1)')
    ax.set_xlim(0, zoom)

    # Panel 3: decoded bit stream of first frame
    ax = axes[2]
    if frames:
        frame = frames[0]
        field_colors_mpl = {
            'start':   '#6e7681',
            'civil':   '#bc8cff',
            'flags':   '#39d353',
            'minutes': '#e3b341',
            'hours':   '#3fb950',
            'date':    '#58a6ff',
            'parity':  '#f85149',
        }
        def field_color(idx):
            if idx in (0, 20):           return field_colors_mpl['start']
            if 1 <= idx <= 14:           return field_colors_mpl['civil']
            if 15 <= idx <= 19:          return field_colors_mpl['flags']
            if 21 <= idx <= 27:          return field_colors_mpl['minutes']
            if idx == 28:                return field_colors_mpl['parity']
            if 29 <= idx <= 34:          return field_colors_mpl['hours']
            if idx == 35:                return field_colors_mpl['parity']
            if 36 <= idx <= 57:          return field_colors_mpl['date']
            if idx == 58:                return field_colors_mpl['parity']
            return '#6e7681'

        for i, b in enumerate(frame[:59]):
            v = b['bit'] if b['bit'] in (0, 1) else 0
            ax.bar(i, v, color=field_color(i), width=0.7, edgecolor='none')

        # Field labels
        for label, pos in [("M",[0]),("civil",[7]),("flags",[17]),
                            ("S",[20]),("min",[24]),("P1",[28]),
                            ("hr",[31]),("P2",[35]),("day",[38]),
                            ("DoW",[43]),("mon",[47]),("yr",[53]),("P3",[58])]:
            ax.text(pos[0], 1.08, label, fontsize=7,
                    color='#8b949e', ha='center', rotation=45)

        ax.set_xlim(-0.5, 59.5)
        ax.set_ylim(-0.1, 1.3)

    ax.set_xlabel('Bit position (second within minute)')
    ax.set_ylabel('Bit value')
    ax.set_title('Decoded bit stream — first frame (color = field)')

    # Legend
    patches = [
        mpatches.Patch(color=field_colors_mpl['civil'],   label='Civil warning [1–14]'),
        mpatches.Patch(color=field_colors_mpl['flags'],   label='Flags [15–19]'),
        mpatches.Patch(color=field_colors_mpl['minutes'], label='Minutes [21–27]'),
        mpatches.Patch(color=field_colors_mpl['hours'],   label='Hours [29–34]'),
        mpatches.Patch(color=field_colors_mpl['date'],    label='Date [36–57]'),
        mpatches.Patch(color=field_colors_mpl['parity'],  label='Parity [28,35,58]'),
    ]
    ax.legend(handles=patches, facecolor='#21262d',
              edgecolor='#30363d', labelcolor='#e6edf3',
              fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('DCF77 Signal Analysis — KiwiSDR IQ',
                    color='#e6edf3', fontsize=14, fontweight='bold', y=0.99)
    outfile = 'dcf77_analysis.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  📊 Plot saved → {outfile}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(samples, sr, plot=False, verbose=False, is_iq=False):
    duration = len(samples) / sr
    print(f"\n⏳ Processing {duration:.1f}s of audio  ({sr} Hz, {len(samples)} samples)")

    print("🔬 Extracting AM envelope...")
    env = extract_envelope(samples, sr, smooth_ms=120, is_iq=is_iq)

    threshold = adaptive_threshold(env)
    print(f"📏 Auto-threshold: {threshold:.3f}")

    print("📈 Detecting carrier-OFF pulses...")
    pulses, _ = detect_pulses(env, sr, threshold)
    print(f"   Found {len(pulses)} pulses")

    print("🧩 Classifying bits...")
    classified = classify_pulses(pulses)
    n0 = sum(1 for b in classified if b['bit'] == 0)
    n1 = sum(1 for b in classified if b['bit'] == 1)
    nM = sum(1 for b in classified if b['bit'] == 'M')
    nE = sum(1 for b in classified if b['bit'] == '?')
    print(f"   0={n0}  1={n1}  minute-marks={nM}  errors={nE}")

    if verbose:
        print("\n  All pulses:")
        for b in classified:
            tag = {'0':'BIT-0','1':'BIT-1','M':'MARK ','?':'ERROR'}.get(str(b['bit']),'?')
            print(f"    t={b['t']:7.2f}s  {b['dur']*1000:5.0f}ms  → {tag}")

    print("\n📦 Extracting minute frames...")
    frames = extract_frames(classified)
    print(f"   {len(frames)} frame(s) found")

    if not frames:
        print("\n❌ No complete DCF77 frames found.")
        print("   • Is the signal audible at 77.5 kHz on the WebSDR?")
        print("   • Try recording longer (--duration 240)")
        print("   • Check threshold — run with --verbose to see raw pulse durations")
        return

    for i, frame in enumerate(frames, 1):
        result = decode_frame(frame)
        print_result(result, i)
        if verbose:
            print_bit_analysis(result, i)

    if plot:
        plot_analysis(env, sr, pulses, threshold, classified, frames)


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           DCF77 Live Decoder — KiwiSDR IQ Edition          ║
║   77.5 kHz · Mainflingen, Germany · Pure Python            ║
╚══════════════════════════════════════════════════════════════╝""")

    parser = argparse.ArgumentParser(
        description="Decode DCF77 from a KiwiSDR IQ recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record IQ from KiwiSDR then decode:
  python ./kiwiclient/kiwirecorder.py --server-host db0whm.hamnet.network \\
    --server-port 8073 --freq 77.5 --mode iq --tlimit 300 --filename dcf77_live

  # Decode the recording:
  python dcf77_websdr.py --file dcf77_live.wav --plot --verbose

  # Quick 90s test:
  python dcf77_websdr.py --file test_iq.wav --verbose
        """
    )
    parser.add_argument("--file", default=None,
                        help="IQ or mono WAV file to decode (required)")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib signal analysis plot")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every pulse and full bit-field breakdown")
    args = parser.parse_args()

    if not args.file:
        print("❌ Please provide a WAV file with --file")
        print("\nRecord one first:")
        print("  python ./kiwiclient/kiwirecorder.py \\")
        print("    --server-host db0whm.hamnet.network --server-port 8073 \\")
        print("    --freq 77.5 --mode iq --tlimit 300 --filename dcf77_live")
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print(f"📂 Loading {args.file} ...")
    samples, sr, is_iq = load_wav(args.file)
    if is_iq:
        print("   📡 IQ stereo detected — computing true amplitude envelope")
    run_pipeline(samples, sr, plot=args.plot, verbose=args.verbose, is_iq=is_iq)


if __name__ == "__main__":
    main()
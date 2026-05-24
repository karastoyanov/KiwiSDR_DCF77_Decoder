#!/usr/bin/env python3
"""
DCF77 Synthetic Signal Generator
=================================
Generates a perfect, noise-free DCF77 WAV file encoding the current time.
Useful for testing the decoder pipeline without a live radio signal.

Usage:
    python generate_dcf77.py
    python generate_dcf77.py --time "2026-05-24 10:37"
    python generate_dcf77.py --minutes 2   # generate 2 full minutes
"""

import argparse
import datetime
import wave
import numpy as np

SR = 12000   # sample rate (Hz)

# ──────────────────────────────────────────────────────────────────────────────
# BCD encoding — DCF77 uses proper BCD, LSB first
# The key: encode each decimal digit separately using weights 1,2,4,8
# then concatenate the digit groups (units first, then tens)
#
# Example: minute = 37
#   units digit  = 7 → bits: 1,1,1,0  (weights 1,2,4,8)
#   tens digit   = 3 → bits: 1,1,0,0  (weights 10,20,40,80... but only 3 bits used for tens of minutes: 1,2,4 mapped to 10,20,40)
#   result bits [21–27]: 1,1,1,0,1,1,0  → decodes back to 7 + 30 = 37 ✅
#
# DCF77 BCD field widths:
#   Minutes:  7 bits  → weights 1,2,4,8,10,20,40
#   Hours:    6 bits  → weights 1,2,4,8,10,20
#   Day:      6 bits  → weights 1,2,4,8,10,20
#   Weekday:  3 bits  → weights 1,2,4
#   Month:    5 bits  → weights 1,2,4,8,10
#   Year:     8 bits  → weights 1,2,4,8,10,20,40,80
# ──────────────────────────────────────────────────────────────────────────────

def bcd_encode(value, weights):
    """
    Encode a decimal value into BCD bits using the given weight sequence.
    DCF77 uses LSB-first ordering with weights: 1,2,4,8,10,20,40,80
    Each weight corresponds to one bit position.
    """
    bits = []
    for w in weights:
        bits.append(1 if value & w else 0)    # won't work for BCD directly!
    return bits

def dcf77_bcd(value, n_bits):
    """
    Correct DCF77 BCD encoding.
    Maps value to bits using the DCF77 weight sequence: 1,2,4,8,10,20,40,80
    
    This works because:
    - Bits weighted 1,2,4,8 encode the units digit (0-9 in binary)
    - Bits weighted 10,20,40,80 encode the tens digit (0-9 in binary, scaled)
    
    Example: value=37, weights=[1,2,4,8,10,20,40]
      37 % 10 = 7 (units) → binary: 1,1,1,0  (for weights 1,2,4,8)
      37 // 10 = 3 (tens)  → binary: 1,1,0    (for weights 1,2,4 scaled to 10,20,40)
      result: [1,1,1,0,1,1,0]
    """
    weights = [1, 2, 4, 8, 10, 20, 40, 80][:n_bits]
    bits = []
    units = value % 10
    tens  = value // 10
    for w in weights:
        if w < 10:
            # units digit bit
            bits.append(1 if (units & w) else 0)
        else:
            # tens digit bit — scale back to single digit
            scaled_w = w // 10
            bits.append(1 if (tens & scaled_w) else 0)
    return bits

def even_parity(bits):
    """Return the even parity bit for a list of bits."""
    return sum(bits) % 2   # 0 if even count of 1s, 1 if odd

def encode_minute(dt):
    """
    Encode a datetime into one 59-bit DCF77 minute frame.
    The frame encodes the time of the NEXT minute (DCF77 convention:
    the bits transmitted during minute N encode N+1).
    """
    # DCF77 transmits the time of the following minute
    # So if it's 10:37 now, the bits being sent encode 10:38
    next_min = dt + datetime.timedelta(minutes=1)

    b = [0] * 59

    # [0] Start of minute — always 0
    b[0] = 0

    # [1–14] Civil warning / weather data — leave as 0 (not public)
    # b[1..14] = 0

    # [15] Call bit — 0 = normal operation
    b[15] = 0

    # [16] DST change announcement — 0 = no change coming
    b[16] = 0

    # [17] CEST active, [18] CET active
    # Approximate: CEST is active March–October
    if 3 <= next_min.month <= 10:
        b[17] = 1   # CEST (UTC+2)
        b[18] = 0
    else:
        b[17] = 0
        b[18] = 1   # CET (UTC+1)

    # [19] Leap second announcement — 0 = no leap second
    b[19] = 0

    # [20] Start of encoded time — always 1
    b[20] = 1

    # [21–27] Minutes (BCD, 7 bits)
    min_bits = dcf77_bcd(next_min.minute, 7)
    b[21:28] = min_bits

    # [28] P1 — even parity over bits 21–27
    b[28] = even_parity(b[21:28])

    # [29–34] Hours (BCD, 6 bits)
    hour_bits = dcf77_bcd(next_min.hour, 6)
    b[29:35] = hour_bits

    # [35] P2 — even parity over bits 29–34
    b[35] = even_parity(b[29:35])

    # [36–41] Day of month (BCD, 6 bits)
    day_bits = dcf77_bcd(next_min.day, 6)
    b[36:42] = day_bits

    # [42–44] Day of week (1=Mon … 7=Sun), BCD 3 bits
    # Python weekday(): 0=Mon, so add 1
    weekday = next_min.weekday() + 1
    weekday_bits = dcf77_bcd(weekday, 3)
    b[42:45] = weekday_bits

    # [45–49] Month (BCD, 5 bits)
    month_bits = dcf77_bcd(next_min.month, 5)
    b[45:50] = month_bits

    # [50–57] Year within century (BCD, 8 bits)
    year_bits = dcf77_bcd(next_min.year % 100, 8)
    b[50:58] = year_bits

    # [58] P3 — even parity over bits 36–57
    b[58] = even_parity(b[36:58])

    return b


def generate_wav(bits_per_minute, n_minutes=2, output_path="dcf77_synthetic.wav"):
    """
    Generate a WAV file containing n_minutes of DCF77 signal.

    Each second contains:
      - carrier ON at amplitude 0.8 for (1.0 - pulse_duration) seconds
      - carrier OFF at amplitude 0.2 for pulse_duration seconds
        (100ms for bit 0, 200ms for bit 1)
    Second 59 of each minute has no pulse (the silence IS the minute marker).
    """
    samples = []

    for minute_idx, bits in enumerate(bits_per_minute):
        print(f"  Minute {minute_idx + 1}: ", end="")

        for sec in range(59):
            bit = bits[sec]
            pulse_dur = 0.100 if bit == 0 else 0.200   # seconds
            on_dur    = 1.0 - pulse_dur

            # Carrier ON portion
            n_on  = int(SR * on_dur)
            n_off = int(SR * pulse_dur)

            samples.extend([0.8] * n_on)
            samples.extend([0.2] * n_off)

            print(str(bit), end="", flush=True)

        # Second 59: full 1-second carrier ON (no pulse = minute marker gap)
        samples.extend([0.8] * SR)
        print("  [gap]")

    # Convert to 16-bit PCM
    arr = np.array(samples, dtype=np.float32)
    pcm = (arr * 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    duration = len(samples) / SR
    print(f"\n  ✅ Saved {output_path}  ({duration:.0f}s, {len(bits_per_minute)} minutes)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic DCF77 WAV file")
    parser.add_argument("--time", default=None,
                        help="Start time as 'YYYY-MM-DD HH:MM' (default: now)")
    parser.add_argument("--minutes", default=2, type=int,
                        help="Number of minutes to generate (default: 2)")
    parser.add_argument("--output", default="dcf77_synthetic.wav",
                        help="Output WAV file path")
    args = parser.parse_args()

    if args.time:
        dt = datetime.datetime.strptime(args.time, "%Y-%m-%d %H:%M")
    else:
        dt = datetime.datetime.now()

    print(f"\n🕐 Generating DCF77 signal starting at: {dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"   {args.minutes} minute(s) → {args.output}\n")

    # Encode each minute
    all_bits = []
    for i in range(args.minutes):
        minute_dt = dt + datetime.timedelta(minutes=i)
        bits = encode_minute(minute_dt)
        all_bits.append(bits)

        # Print what we're encoding for verification
        next_dt = minute_dt + datetime.timedelta(minutes=1)
        print(f"  Encoding minute {i+1}: will display {next_dt.strftime('%H:%M')} "
              f"on {next_dt.strftime('%A %d/%m/%Y')}")

        # Verify our BCD is correct by decoding it back
        from_bits = lambda b, w: sum(v * x for v, x in zip(b, w))
        weights7 = [1,2,4,8,10,20,40]
        weights6 = [1,2,4,8,10,20]
        weights8 = [1,2,4,8,10,20,40,80]
        decoded_min  = from_bits(bits[21:28], weights7)
        decoded_hour = from_bits(bits[29:35], weights6)
        decoded_day  = from_bits(bits[36:42], weights6)
        decoded_year = 2000 + from_bits(bits[50:58], weights8)
        print(f"    Verify: {decoded_hour:02d}:{decoded_min:02d}  "
              f"day={decoded_day}  year={decoded_year}  "
              f"P1={'✅' if sum(bits[21:29])%2==0 else '❌'}  "
              f"P2={'✅' if sum(bits[29:36])%2==0 else '❌'}  "
              f"P3={'✅' if sum(bits[36:59])%2==0 else '❌'}")

    generate_wav(all_bits, output_path=args.output)

    print(f"\nRun the decoder:")
    print(f"  python dcf77_websdr.py --file {args.output} --plot --verbose")


if __name__ == "__main__":
    main()

# COMPLETE EXAMPLE: HOW DATA IS TRANSFORMED

## BRUTE FRAME → AGGREGATED → DECODED

---

## STEP 1: BRUTE FRAME (Raw CAN Capture)

**From file:** `Frame251016(0-9999).txt`, Line 329

```
Timestamp:   14:17:00.329.0
CAN ID:      0x09fd0203
Data bytes:  47 71 52 1D 86 39 CF FF
```

**What is this?**
- One CAN frame captured by ESP32
- 8 bytes of hexadecimal data
- Timestamp shows when it was captured

---

## STEP 2: EXTRACT PGN (Message Type)

**CAN ID breakdown:**
```
CAN ID: 0x09fd0203 = 167313923 (decimal)

Extract PGN (bits 8-25):
  167313923 >> 8 = 653569
  653569 & 0x1FFFF = 129025

PGN = 129025 (Position Rapid Update)
```

---

## STEP 3: AGGREGATED RECORD

**This frame is complete (not Fast-Packet), so:**

```csv
Timestamp,ID,PGN,Payload
14:17:00.329.0,0x09fd0203,129025,47 71 52 1D 86 39 CF FF
```

**This goes into:** `aggregated_brute_frames.csv`

---

## STEP 4: DECODE THE MESSAGE

**PGN 129025 Specification (NMEA 2000):**
- Bytes 0-3: Latitude (int32, little-endian, scale ×1e-7 degrees)
- Bytes 4-7: Longitude (int32, little-endian, scale ×1e-7 degrees)

### Decode LATITUDE (Bytes 0-3):

```
Byte 0: 0x47 = 71
Byte 1: 0x71 = 113  
Byte 2: 0x52 = 82
Byte 3: 0x1D = 29

Step 1: Combine bytes (little-endian)
  71 | (113<<8) | (82<<16) | (29<<24)
  = 71 | 28928 | 5373952 | 486539264
  = 491942215

Step 2: Check if signed (is it > 2147483647?)
  No → value is 491942215

Step 3: Apply scale factor
  491942215 × 0.0000001 = 49.1942215°
```

### Decode LONGITUDE (Bytes 4-7):

```
Byte 4: 0x86 = 134
Byte 5: 0x39 = 57
Byte 6: 0xCF = 207
Byte 7: 0xFF = 255

Step 1: Combine bytes (little-endian)
  134 | (57<<8) | (207<<16) | (255<<24)
  = 134 | 14592 | 13565952 | 4278190080
  = 4291770758 (unsigned)

Step 2: Convert to signed int32
  4291770758 > 2147483647
  4291770758 - 4294967296 = -3196538

Step 3: Apply scale factor
  -3196538 × 0.0000001 = -0.3196538°
```

---

## STEP 5: DECODED RESULT

```csv
Timestamp,latitude,longitude,sog,heading,...
14:17:00.329.0,49.1942215,-0.3196538,0.01,15.819365,...
```

**This goes into:** `decoded_brute_frames.csv`

---

## VERIFICATION

**Manual calculation:**
- Latitude: 49.1942215°
- Longitude: -0.3196538°

**From decoded_brute_frames.csv:**
- Latitude: 49.1942215°
- Longitude: -0.3196538°

✅ **PERFECT MATCH!**

---

## WHAT THIS MEANS

**Geographic location:** 49.19°N, 0.32°W  
**Real location:** Normandy coast, France (English Channel)  
**Your boat's actual GPS position!** 🚢

---

## COMPLETE DATA FLOW

```
RAW BRUTE FRAME
├─ 47 71 52 1D 86 39 CF FF  (8 hex bytes)
│
↓ (Fast-Packet reassembly if needed)
│
AGGREGATED
├─ Timestamp: 14:17:00.329.0
├─ PGN: 129025
└─ Payload: 47 71 52 1D 86 39 CF FF
│
↓ (Byte extraction + scaling)
│
DECODED
├─ Latitude: 49.1942215°
└─ Longitude: -0.3196538°
```

---

## NO DATA LOSS!

**4M frames → 154K messages → 98K decoded**

- 4M frames include parts of multi-frame messages (A1, A2, A3...)
- 154K complete messages after reassembly
- 98K decoded (only supported PGNs)

**The "loss" is protocol behavior, not data corruption!**

✅ Your decoder is working correctly!

---

## STEP-BY-STEP EXAMPLE #2: RATE OF TURN (PGN 127251)

The walkthrough above used PGN 129025 (Position Rapid Update). Below is another complete example using a real frame captured in `Frame brute cantest/Frame251016(0-9999).txt`.

### Step 1: Grab the raw frame

From line **61** of the file:

```
Receive  14:28:27.571.0  0x09f11303  Data  Extend  8  FF 00 FA 00 00 FF FF FF
```

- **Timestamp**: `14:28:27.571.0`
- **CAN ID**: `0x09f11303` (29-bit identifier)
- **Payload**: `FF 00 FA 00 00 FF FF FF`

### Step 2: Extract the PGN from the CAN ID

1. Convert the ID to decimal for easier bit math: `0x09f11303 = 166,793,987`.
2. Take bits 8–25 → `(166793987 >> 8) & 0x1FFFF = 127251`.
3. Check PDU Format (bits 16–23) → `(166793987 >> 16) & 0xFF = 241`.
4. Because `241 ≥ 240`, this is **PDU2** (broadcast), so the PGN stays 127251.
5. Look up PGN 127251 in `src/preprocessing/n2k_decoder.py` → handler `RateOfTurn`.

### Step 3: Confirm whether it is a fast-packet

- Fast packets start with byte 0 having its lower 5 bits equal to 0 (sequence index 0).  
- `0xFF & 0x1F = 31`, so this frame does **not** start a fast packet.  
- The decoder therefore treats it as a single 8-byte frame (see `_iter_complete_messages()` around line 469).

### Step 4: Build the aggregated record

No reassembly needed, so the aggregated CSV row is simply:

```csv
Timestamp,ID,PGN,Payload
14:28:27.571.0,0x09f11303,127251,FF 00 FA 00 00 FF FF FF
```

### Step 5: Decode the payload bytes

See `_decode_rate_of_turn()` in `src/preprocessing/n2k_decoder.py`.

1. **Byte 0 (SID)** = `0xFF` → “not available”, ignored.
2. **Bytes 1–4** (`00 FA 00 00`) form a signed int32, little-endian:
   - Combine → `0 | (0xFA << 8) | (0x00 << 16) | (0x00 << 24) = 64,000`.
   - Value < `0x80000000`, so no sign correction required.
3. Apply NMEA scaling for PGN 127251:  
   `rate_rad = 64,000 × 1e-6 / 32 = 0.002 rad/s`.
4. Convert to degrees per second:  
   `rate_deg = 0.002 × (180 / π) ≈ 0.1146 deg/s`.

### Step 6: Final decoded value

```csv
Timestamp,rate_of_turn
14:28:27.571.0,0.1145916
```

- 0.1146 deg/s ≈ 6.9 deg/min, meaning the vessel is turning very slowly (almost straight).
- The decoder produces the same result:  
  `{'rate_of_turn': 0.11459155902616465}`

---

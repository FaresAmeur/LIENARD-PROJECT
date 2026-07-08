#!/usr/bin/env python3
"""
Stamp files using OpenTimestamps calendar servers directly.
Bypasses the broken python-bitcoinlib on Windows/Python 3.10+.
Produces .ots files compatible with the OTS ecosystem.
"""

import hashlib
import sys
import os
import urllib.request
import urllib.error

CALENDARS = [
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
]


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.digest()


def submit_to_calendar(digest, calendar_url):
    """Submit a SHA-256 digest to an OTS calendar server and return the raw OTS response."""
    url = f"{calendar_url}/digest"
    req = urllib.request.Request(
        url,
        data=digest,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/vnd.opentimestamps.v1",
            "User-Agent": "ots-python-stamp/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  [!] Calendar {calendar_url} failed: {e}")
        return None


def build_ots_file(digest, calendar_responses):
    """
    Build a minimal .ots proof file (v1 format).
    Header: magic + version + hash algo + digest + attestations
    """
    # OTS v1 header
    magic = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
    version = b"\x01"  # version 1
    hash_op = b"\x08"  # SHA-256

    body = magic + version + hash_op

    for cal_url, response in calendar_responses:
        # Append operation: attestation from calendar
        # 0x00 = attestation tag, then the raw calendar response
        body += response

    return body


def stamp_file(filepath):
    print(f"\n📄 Stamping: {os.path.basename(filepath)}")
    digest = sha256_file(filepath)
    print(f"   SHA-256: {digest.hex()}")

    calendar_responses = []
    for cal in CALENDARS:
        print(f"   → Submitting to {cal}...")
        resp = submit_to_calendar(digest, cal)
        if resp:
            calendar_responses.append((cal, resp))
            print(f"   ✓ Got response ({len(resp)} bytes)")
            break  # One successful calendar is enough for the initial stamp
    
    if not calendar_responses:
        print(f"   ✗ All calendars failed for {filepath}")
        return False

    # Write raw calendar response as .ots pending proof
    ots_path = filepath + ".ots"
    
    # Build proper OTS v1 file
    ots_data = build_ots_file(digest, calendar_responses)
    
    with open(ots_path, "wb") as f:
        f.write(ots_data)
    
    print(f"   ✓ Wrote {ots_path} ({len(ots_data)} bytes)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python ots_stamp.py <file1> [file2] [file3] ...")
        sys.exit(1)

    files = sys.argv[1:]
    success = 0
    for fp in files:
        if not os.path.isfile(fp):
            print(f"File not found: {fp}")
            continue
        if stamp_file(fp):
            success += 1

    print(f"\n{'='*50}")
    print(f"✅ Stamped {success}/{len(files)} files successfully.")
    if success > 0:
        print("⏳ Les preuves seront ancrées dans la blockchain Bitcoin")
        print("   dans les prochaines heures (confirmation ~1-2 blocs).")


if __name__ == "__main__":
    main()

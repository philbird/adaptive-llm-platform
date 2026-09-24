"""Provision/rotate offline signing keys; output contains no key material."""

import argparse
from pathlib import Path

from adaptive_llm.signing import rotate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-keys", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    args = parser.parse_args()
    try:
        rotate(args.private_key, args.public_keys, args.key_id)
    except Exception:
        parser.exit(1, "signing_rotation_failed\n")
    print("signing_key_rotated")


if __name__ == "__main__":
    main()

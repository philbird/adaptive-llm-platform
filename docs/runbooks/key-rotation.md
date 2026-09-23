# Payload key rotation

`PayloadCipher` accepts a mapping of version names to 32-byte keys and one current version.
Writes use the current version; reads select the recorded version. Missing keys fail with
`unknown_payload_key_version`; authentication failures use `payload_authentication_failed`.
Ciphertext remains AES-256-GCM with a fresh random nonce and tenant/interaction/field AAD.

Provision an operator-managed JSON keyring outside the repository. Its shape is
`{"current_version":"local-2","keys":{"local-1":"<64 hex characters>","local-2":"<64 hex characters>"}}`.
Preserve the actual existing key under its original version; generate an independent new key.
Do not replace the old key's bytes or log this file. Local slice-1b data uses `Settings().payload_key`
as its original synthetic `local-1` key; production obtains keys from its managed key system.

Restart the gateway with both keys loaded and the new current version, then migrate old blobs:

```sh
PAYLOAD_KEYRING=/secure/payload-keys.json make dev
make rotate-key DATA_DIR=.local TENANT=synthetic-a NEW_KEY_VERSION=local-2 KEYRING=/secure/payload-keys.json
```

`Settings.payload_keyring_path` or the `PAYLOAD_KEYRING` environment variable loads this file;
embedded apps may inject `payload_keys` and `payload_key_version` directly. The CLI refuses
rotation if `NEW_KEY_VERSION` differs from the file's current version. Expected output is
`rotated_payloads=<count>`; repeating a completed rotation prints `rotated_payloads=0`.

Rotation uses transactions of at most 100 blobs by default, preserves references and expiry,
and scopes every read/write to the selected tenant. It processes live blobs present at the
start of the command, skipping expired blobs and blobs already using the current key. New
traffic must already use the new version. A failed batch rolls back; earlier batches remain
rotated and rerunning is safe. The Python CLI also accepts `--batch-size` for drill sizing.
Keep old keys until every tenant and retained backup that needs them has been handled. Backups
are not rewritten by rotation. No key bytes are generated or retired by this command.

```sh
uv run --locked pytest tests/drills/test_resilience.py -k key_rotation_with_live_traffic -s
```

Measured locally on 2026-09-23: **100 old blobs rotated in batches of 7 during 100 new
requests; all 200 blobs used the current key; old request replay matched before and after
rotation, elapsed 0.453 s**. Unit tests also prove tenant isolation, failed-batch rollback,
missing-key failure and replay after restart with only the new key.

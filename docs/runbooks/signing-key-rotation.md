# Signing key provisioning and rotation

Use a directory outside source control with operator-only permissions. Initial provisioning
and rotation both use the same command; a key id and private-key destination must be new.

```sh
make sign-rotate SIGNING_KEY_PATH=/secure/signing/a.pem SIGNING_PUBLIC_KEYS=/secure/signing/public.json SIGNING_KEY_ID=a
make sign-rotate SIGNING_KEY_PATH=/secure/signing/b.pem SIGNING_PUBLIC_KEYS=/secure/signing/public.json SIGNING_KEY_ID=b
```

Success prints `signing_key_rotated`. Private PEM files are mode 0600. The ring contains only
public keys, one `active_key_id`, and statuses `active`/`verify-only`. Rotation publishes the
ring last via atomic rename and never overwrites existing key material. Serialize operator
rotation commands. Preserve old private keys offline only if archival policy requires them;
old public keys are necessary to verify existing artifacts, reports, approvals and checkpoints.

Distribute the updated public ring and restart verification/deployment processes first,
with only `SIGNING_PUBLIC_KEYS` configured and `SIGNING_KEY_PATH` unset. Then restart writers
with `SIGNING_KEY_PATH=/secure/signing/b.pem`, `SIGNING_KEY_ID=b` and
`SIGNING_PUBLIC_KEYS=/secure/signing/public.json`. A verifier-only process
refuses artifact creation with `signing_key_required`; inference and verified promotion do
not require the signing key. Loading Settings with a retired or mismatched signing key fails.

Default local development provisions a persistent key in `<data_dir>/signing`; this is solely
for synthetic work. Explicit production/staging configuration requires verification keys.
The payload-encryption keyring and subject/content HMAC secret are independent and unchanged
by signing rotation. MAC-only artifacts, dataset approvals, reports, checkpoints and research
study envelopes require `Settings.legacy_mac_records=True` or `LEGACY_MAC_RECORDS=true`
when verification keys are configured. This defaults to true only in `local`, and false in
all other environments. Legacy opt-in also needs the historical HMAC secret. Plan to re-sign
these records with an authorized writer or retire them, then disable the flag. Partial or
invalid signatures always fail closed, even with legacy opt-in. Do not rewrite historical
records or remove old public keys during ordinary rotation.

```sh
uv run --locked pytest tests/unit/test_signing.py tests/integration/test_signatures.py
```

Tests generate fresh PEM keys, sign with A, rotate to B, delete private keys before verification,
and verify old records, new records, tamper rejection and promotion without a signer.

# ADR 0005: Ed25519 artifact signatures and public verification

Status: accepted for local implementation, 2026-09-24.

Dataset manifests and approvals, model manifests, evaluation/benchmark reports, training
checkpoints and research study envelopes now use purpose-bound Ed25519 signatures. Each
signature records its key id and `ed25519-v1` version. The canonical payload includes immutable
metadata; mutable registry state and evaluation associations remain protected by the
transactional promotion controls.
An approval binds the dataset manifest and the authenticated approver/reason/time.

`Signer` and `Verifier` are independent protocols. PEM private keys are loaded from
`Settings.signing_key_path`. Deployment processes configure only `signing_public_keys_path`;
they can verify, load and promote signed artifacts without a private key. They cannot publish
new signed records. Local development generates a private key in the ignored data directory
on first startup; tests generate new keys under their temporary directories. No private keys
are checked into source control. Production provisioning and custody remain operator duties.

Legacy records retain their exact recorded v1/v2 MAC encoding only when
`Settings.legacy_mac_records` / `LEGACY_MAC_RECORDS=true` permits them. The default is true
only in `local`; development, staging and production default to false. With a configured
verifier, a MAC-only record fails closed unless this flag is explicitly enabled. Knowing the
shared HMAC secret cannot bypass signatures by stripping signature fields. Signature-bearing
records, including partial signatures, never fall back to a MAC after a signature error.
New signed artifacts leave old MAC fields/sidecars empty (research envelopes omit the MAC).
Legacy opt-in also requires the historical HMAC secret and is a migration allowance: legacy
records are expected to be re-signed by an authorized writer or retired. Re-signing is not
automatic, and ordinary key rotation does not rewrite historical records.

Rotation adds an independent key and retains prior public keys as verify-only. Restart writers
with the new private key/id and updated public ring; deployment processes receive only the ring.
Offline revocation is removal from that ring, after assessing retained artifacts and backups.
See [the rotation runbook](../runbooks/signing-key-rotation.md).

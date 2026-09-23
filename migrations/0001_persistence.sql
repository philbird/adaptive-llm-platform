CREATE TABLE interactions (
    tenant_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    interaction_id TEXT NOT NULL,
    subject_id TEXT,
    data TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'expired', 'deleted')),
    PRIMARY KEY (tenant_id, record_id)
);
CREATE INDEX interactions_subject ON interactions (tenant_id, subject_id);
CREATE INDEX interactions_expiry ON interactions (tenant_id, state, expires_at);
CREATE TABLE started (
    tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, interaction_id TEXT NOT NULL,
    data TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tenant_id, record_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE TABLE retrieval_runs (
    tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, interaction_id TEXT NOT NULL,
    data TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tenant_id, record_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE TABLE route_decisions (
    tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, interaction_id TEXT NOT NULL,
    data TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tenant_id, record_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE TABLE attempts (
    tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, interaction_id TEXT NOT NULL,
    data TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tenant_id, record_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE TABLE feedback (
    tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, interaction_id TEXT NOT NULL,
    data TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tenant_id, record_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE TABLE deletion_tombstones (
    tenant_id TEXT NOT NULL, interaction_id TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (tenant_id, interaction_id)
);
CREATE TABLE payloads (
    tenant_id TEXT NOT NULL, reference TEXT NOT NULL, interaction_id TEXT NOT NULL,
    field TEXT NOT NULL, nonce BLOB NOT NULL, ciphertext BLOB NOT NULL,
    key_version TEXT NOT NULL, expires_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, reference),
    UNIQUE (key_version, nonce),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id)
);
CREATE INDEX payloads_interaction ON payloads (tenant_id, interaction_id);
CREATE TABLE replay_entries (
    tenant_id TEXT NOT NULL, application_id TEXT NOT NULL, request_id TEXT NOT NULL,
    interaction_id TEXT NOT NULL, fingerprint TEXT NOT NULL, response_ref TEXT NOT NULL,
    reserved_at TEXT NOT NULL, expires_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, application_id, request_id),
    FOREIGN KEY (tenant_id, interaction_id) REFERENCES interactions (tenant_id, record_id),
    FOREIGN KEY (tenant_id, response_ref) REFERENCES payloads (tenant_id, reference)
);

CREATE TABLE subject_tombstones (
    tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (tenant_id, subject_id)
);
CREATE INDEX replay_entries_oldest ON replay_entries
    (tenant_id, reserved_at, application_id, request_id);

CREATE TABLE evaluation_reports (
    evaluation_id TEXT PRIMARY KEY,
    tenant_ids TEXT NOT NULL,
    report TEXT NOT NULL,
    report_mac TEXT NOT NULL,
    operator_note_hash TEXT,
    actor_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE baselines (
    deployment_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    evaluation_id TEXT NOT NULL REFERENCES evaluation_reports(evaluation_id),
    manifest_version TEXT NOT NULL,
    locked_at TEXT NOT NULL,
    PRIMARY KEY (deployment_id, dataset_version)
);

-- Remove empty tenant tables inherited by the slice-2b control database.
CREATE TEMP TABLE control_upgrade_guard (count INTEGER CHECK (count = 0));
CREATE TABLE IF NOT EXISTS replay_entries (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM replay_entries;
DROP TABLE replay_entries;
CREATE TABLE IF NOT EXISTS payloads (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM payloads;
DROP TABLE payloads;
CREATE TABLE IF NOT EXISTS feedback (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM feedback;
DROP TABLE feedback;
CREATE TABLE IF NOT EXISTS attempts (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM attempts;
DROP TABLE attempts;
CREATE TABLE IF NOT EXISTS route_decisions (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM route_decisions;
DROP TABLE route_decisions;
CREATE TABLE IF NOT EXISTS retrieval_runs (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM retrieval_runs;
DROP TABLE retrieval_runs;
CREATE TABLE IF NOT EXISTS started (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM started;
DROP TABLE started;
CREATE TABLE IF NOT EXISTS interactions (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM interactions;
DROP TABLE interactions;
CREATE TABLE IF NOT EXISTS deletion_tombstones (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM deletion_tombstones;
DROP TABLE deletion_tombstones;
CREATE TABLE IF NOT EXISTS subject_tombstones (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM subject_tombstones;
DROP TABLE subject_tombstones;
CREATE TABLE IF NOT EXISTS dataset_manifests (legacy_placeholder TEXT);
INSERT INTO control_upgrade_guard SELECT count(*) FROM dataset_manifests;
DROP TABLE dataset_manifests;
DROP TABLE control_upgrade_guard;
DELETE FROM schema_migrations WHERE version IN (1, 2, 4, 5);
CREATE TABLE training_jobs (
    job_id TEXT PRIMARY KEY,
    tenant_ids TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE model_versions (
    version TEXT PRIMARY KEY,
    registry_id TEXT NOT NULL,
    tenant_ids TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE lifecycle_transitions (
    transition_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL REFERENCES model_versions(version),
    data TEXT NOT NULL
);
CREATE TABLE deployments (
    deployment_id TEXT PRIMARY KEY,
    current_version TEXT NOT NULL REFERENCES model_versions(version),
    previous_version TEXT REFERENCES model_versions(version)
);

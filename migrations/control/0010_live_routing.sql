ALTER TABLE route_policy_active ADD COLUMN disabled_specialists TEXT NOT NULL DEFAULT '[]';
CREATE TABLE live_observations (
    interaction_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES route_policies(policy_id),
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX live_observations_policy_time ON live_observations(policy_id, created_at);
CREATE TABLE rollback_measurements (
    measurement_id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    specialist_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);

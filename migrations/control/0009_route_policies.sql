CREATE TABLE route_policies (
    policy_id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    tenant_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE route_policy_active (
    environment TEXT PRIMARY KEY,
    policy_id TEXT REFERENCES route_policies(policy_id),
    kill_switch INTEGER NOT NULL DEFAULT 0,
    disabled_tenants TEXT NOT NULL DEFAULT '[]',
    disabled_tasks TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE route_policy_history (
    transition_id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    policy_id TEXT,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE shadow_observations (
    interaction_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES route_policies(policy_id),
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    segments TEXT NOT NULL
);
CREATE INDEX shadow_observations_policy_time ON shadow_observations(policy_id, created_at);
CREATE TABLE shadow_comparisons (
    interaction_id TEXT PRIMARY KEY REFERENCES shadow_observations(interaction_id),
    data TEXT NOT NULL
);

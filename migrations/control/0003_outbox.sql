CREATE TABLE outbox (
    sequence INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    interaction_id TEXT,
    event_type TEXT NOT NULL,
    envelope TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'delivered', 'dead')),
    last_error_code TEXT CHECK (last_error_code IN ('sink_unavailable', 'invalid_event'))
);
CREATE INDEX outbox_due ON outbox (state, next_attempt_at, sequence);
CREATE INDEX outbox_interaction ON outbox (tenant_id, interaction_id, state, sequence);
CREATE INDEX outbox_dead ON outbox (tenant_id, state);

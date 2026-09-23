ALTER TABLE interactions ADD COLUMN started_at TEXT NOT NULL DEFAULT '';
-- Canonical UTC spelling matches timestamp() writes and SQL window bounds.
UPDATE interactions SET started_at = replace(json_extract(data, '$.started_at'), 'Z', '+00:00');
CREATE INDEX interactions_tenant_started_at ON interactions (tenant_id, started_at);

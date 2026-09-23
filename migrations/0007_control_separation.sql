-- Slice 2b created unused control tables in tenant databases. Reject populated legacy
-- tables rather than silently discard records; those require an operator export first.
CREATE TABLE IF NOT EXISTS evaluation_reports (evaluation_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS baselines (evaluation_id TEXT);
CREATE TEMP TABLE control_separation_guard (count INTEGER CHECK (count = 0));
INSERT INTO control_separation_guard SELECT count(*) FROM evaluation_reports;
INSERT INTO control_separation_guard SELECT count(*) FROM baselines;
DROP TABLE baselines;
DROP TABLE evaluation_reports;
DROP TABLE control_separation_guard;

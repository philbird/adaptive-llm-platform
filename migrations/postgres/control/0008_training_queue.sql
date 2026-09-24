-- Identity comes exclusively from authenticated submission, never the specification.
CREATE TABLE training_submitters (
    job_id TEXT PRIMARY KEY REFERENCES training_jobs(job_id),
    identity TEXT NOT NULL
);
CREATE INDEX training_queue_state_created ON training_jobs (
    (data::jsonb ->> 'state'), (data::jsonb ->> 'created_at')
);

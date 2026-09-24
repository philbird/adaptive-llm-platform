CREATE TABLE benchmark_reports (
    benchmark_id TEXT PRIMARY KEY,
    candidate_version TEXT NOT NULL,
    evaluation_id TEXT NOT NULL,
    report TEXT NOT NULL
);
CREATE INDEX benchmark_evaluation ON benchmark_reports(candidate_version, evaluation_id);

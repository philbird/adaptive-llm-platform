CREATE TABLE dataset_manifests (
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (dataset_id, version)
);

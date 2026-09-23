# Migrations

No database schema exists yet. Introduce numbered, transactional migrations with the milestone 1
operational/event stores. Record applied versions; test upgrade/rollback, outbox atomicity,
tenant filtering and deletion tombstones. Do not create or mutate production tables manually.


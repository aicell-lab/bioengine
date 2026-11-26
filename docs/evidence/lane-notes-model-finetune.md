# Lane notes — findings NOT filed as rows

Per the standing direction: no new rows from this lane unless the finding is a
security hole, a data-loss path, or production broken right now. Everything else
is one line here and the lead triages in batches.

One line each. Measured facts only; a guess says so.

## Open

- 2026-11-26 · model-finetune · `0o222` umask makes `write_status` fail at its own
  `.mutex-*` file, not at the status file — so that umask is unreachable in tests
  and the mutex is the narrower constraint. Measured, not filed.
- 2026-11-26 · model-finetune · `caller_may_resume_run` admits an administrator to
  a run it cannot LIST (pre-existing, preserved deliberately by #0916's design
  note). Reads oddly; nobody has asked for it to change.
- 2026-11-26 · annotation-broker · a failure during a crowd of gallery visitors is
  not single-flighted: with the artifact manager down, 5 visitors at N=10 made 50
  reads. Direct consequence of not caching failures, accepted by the lead on #0963.
- 2026-11-26 · annotation-broker · `_public_card_locks` can have an entry evicted
  while an in-flight fill holds it, if the dataset is unpublished in between.
  Worst case one extra read; it cannot be served, because the gallery filters on
  `iter_public_metadata`. Raised by #0963's reviewer.
- 2026-11-26 · model-finetune · the resume rule's documented statements are pinned
  by phrase presence, so a narrowing BY NEGATION ("…and to an administrator, not
  to anyone who owns or manages the dataset") keeps every phrase and inverts the
  meaning. Inherent to phrase pinning; closing it needs the test to parse English.

# Fix Plans

Architect-generated issue index. Each row is a distinct anomaly type — not a per-run finding.
Full fix plan detail (root cause, acceptance criteria, source runs) lives in the linked plan file.

| Plan | Title | Type | Severity | Status |
|------|-------|------|----------|--------|
| [P010](plans/P010.json) | Stable exits never listed → agent re-tries only known edges (missing-exit extraction) | exit_not_tried | medium | fixed |
| [P001](plans/P001.json) | Redundant post-take inspection verbs | redundant_object_inspection | low | deferred |
| [P009](plans/P009.json) | Verb sequence runs redundant inspection verbs after first non-informative response | redundant_object_interaction | low | deferred |
| [P004](plans/P004.json) | Two-room oscillation loop | repetitive_oscillation | high | fixed |
| [P005](plans/P005.json) | "out" oscillation loop — reverse-exit not wired for in/out | loop_detected | high | fixed |
| [P006](plans/P006.json) | Non-productive "You can't go that way" streak from unmarked futile exits | futile_streak | high | fixed |
| [P007](plans/P007.json) | position_lost look never fires on repeated no-room extractions | lost_position | high | fixed |
| [P002](plans/P002.json) | Zigzag navigation between two rooms | repetitive_zigzag_navigation | medium | fixed |
| [P008](plans/P008.json) | Re-examine exits after combat-forced relocation | nav_contradiction | medium | fixed |
| [P003](plans/P003.json) | Futile up/down direction probing | futile_direction_probing | low | fixed |

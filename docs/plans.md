# Fix Plans

Architect-generated issue registry. Each entry represents a distinct anomaly type — not a per-run finding.
When a new run surfaces an existing type, the run is added under **Also seen** and priority is reviewed.
Full fix plan JSON lives in `../plans/P<N>.json`.

---

## Open

*(none)*
---

## Deferred

### P001 — redundant_object_inspection [low]
Post-take inspection sequence runs all verbs without checking whether a prior verb already returned identical output, wasting steps per object.
- **First seen:** watch_20260702_152816 (a1, a2)
- **Also seen:** *(none yet)*
- **Note:** Deferred: identical response doesn’t mean all verbs are useless for the object (e.g. drop/throw may differ). Revisit when a run shows this causing meaningful progress loss.
- **Plan:** [plans/P001.json](../plans/P001.json)

---

## Fixed

### P002 — repetitive_zigzag_navigation [medium]
Navigation always picks the nearest room with Unknown exits, creating diagonal sw/east chains across grid-like forest areas without making directional progress.
- **First seen:** watch_20260702_152816 (a3)
- **Also seen:** *(none yet)*
- **Fixed in:** PR #45 — direction-diversity scoring added to navigation target selection (path_len + min_recent_dir_frequency)
- **Plan:** [plans/P002.json](../plans/P002.json)

### P003 — futile_direction_probing [low]
LLM infers up/down from ‘Exits lead in all directions’; Unknown vertical edges are created and always fail, wasting two steps per room visit.
- **First seen:** watch_20260702_152816 (a4)
- **Also seen:** *(none yet)*
- **Fixed in:** PR #45 — up/down filtered from extracted exits unless those words appear literally in the response text
- **Plan:** [plans/P003.json](../plans/P003.json)

---

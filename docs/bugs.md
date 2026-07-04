# Bugs

Incorrect or broken behaviour observed during runs. Original issue numbers preserved for git/PR reference.

Architect-generated anomaly plans are tracked separately in [plans.md](plans.md) with canonical plan files in `plans/`.

| ID | File | Title | Status |
|----|------|-------|--------|
| 1 | [1.md](bugs/1.md) | Engine reads and inspects horse as object | ✅ Fixed |
| 4 | [4.md](bugs/4.md) | "You can see" not parsed as room contents | ✅ Fixed |
| 5 | [5.md](bugs/5.md) | Agent assumes seen items are in inventory | ✅ Fixed |
| 7 | [7.md](bugs/7.md) | Agent doesn't understand failure phrases | ✅ Fixed |
| 8 | [8.md](bugs/8.md) | NoneType error on null potential_solution | ✅ Fixed |
| 11 | [11.md](bugs/11.md) | Timeout message lacks raw received text | ✅ Fixed |
| 12 | [12.md](bugs/12.md) | Engine still tries to read the horse | ✅ Fixed |
| 13 | [13.md](bugs/13.md) | Model non-deterministically returns empty extraction | ✅ Fixed |
| 14 | [14.md](bugs/14.md) | Loop detection false-positive on object inspection | ✅ Fixed |
| 17 | [17.md](bugs/17.md) | Prompt change to ">" breaks pexpect match | ✅ Fixed | 
| 19 | [19.md](bugs/19.md) | Crash on use command with None node | ✅ Fixed |
| 29 | [29.md](bugs/29.md) | Command echo appearing in game responses | ✅ Fixed |
| 31 | [31.md](bugs/31.md) | Sub-objects revealed by examine not extracted | ✅ Fixed |
| 32 | [32.md](bugs/32.md) | Crash details not visible in log file | ✅ Fixed |
| 33 | [33.md](bugs/33.md) | "Exits lead in all directions" not parsed | ✅ Fixed |
| 35 | [35.md](bugs/35.md) | LLM hallucinates inventory on hard-failure responses | ✅ Fixed |
| 36 | [36.md](bugs/36.md) | Wear command produces bad location string | ✅ Fixed |
| 37 | [37.md](bugs/37.md) | Examine command sets room to object name | ✅ Fixed |
| 38 | [38.md](bugs/38.md) | LLM hallucinates direction string as room name | ✅ Fixed |
| 39 | [39.md](bugs/39.md) | LLM places NPC name in room field | ✅ Fixed |
| 41 | [41.md](bugs/41.md) | Missing "Don't be silly" failure pattern | ✅ Fixed |
| 42 | [42.md](bugs/42.md) | Crash on null list fields from LLM | ✅ Fixed |
| 45 | [45.md](bugs/45.md) | Maze rooms with non-unique names loop agent | 🔲 Open |
| 46 | [46.md](bugs/46.md) | Article variants create duplicate room nodes | ✅ Fixed |
| 47 | [47.md](bugs/47.md) | Unknown placeholder re-added on same update call | ✅ Fixed |
| 48 | [48.md](bugs/48.md) | Direction aliases overwrite edge label | ✅ Fixed |
| 49 | [49.md](bugs/49.md) | Non-cardinal directions missing from reverse map | ✅ Fixed |
| 50 | [50.md](bugs/50.md) | Article variations create duplicate graph nodes | ✅ Fixed |
| 51 | [51.md](bugs/51.md) | Diagonal rooms placed at wrong map positions | ✅ Fixed |
| 52 | [52.md](bugs/52.md) | Map PNG uses dark theme, hard to read | ✅ Fixed |
| 53 | [53.md](bugs/53.md) | Unknown nodes clutter map at full size | ✅ Fixed |
| 54 | [54.md](bugs/54.md) | Agent loops on look with no local unknown exits | ✅ Fixed |
| 55 | [55.md](bugs/55.md) | Agent falls back to look on pre-seeded graph | ✅ Fixed |
| 57 | [57.md](bugs/57.md) | LLM returns full description as room name | ✅ Fixed |
| 58 | [58.md](bugs/58.md) | Scenery items not in hard failure patterns | ✅ Fixed |
| 59 | [59.md](bugs/59.md) | Compound edge labels grow infinitely | ✅ Fixed |
| 60 | [60.md](bugs/60.md) | Lost room after successful movement | ✅ Fixed |
| 61 | [61.md](bugs/61.md) | Streamlit UI broken after cloud LLM support added | ✅ Fixed |
| 62 | [62.md](bugs/62.md) | Boot/Reset buttons have no guard against misuse | ✅ Fixed |
| 63 | [63.md](bugs/63.md) | TypeError crash on first step when boot LLM returns null room | ✅ Fixed |
| 64 | [64.md](bugs/64.md) | Known-from-prior-run objects never queued for take | ✅ Fixed |
| 65 | [65.md](bugs/65.md) | agent_dev_loop.py subprocess output invisible until the step finishes | ✅ Fixed |
| 66 | [66.md](bugs/66.md) | run_and_analyze.py silent for the whole run (no per-step output, no warning suppression) | ✅ Fixed |
| 67 | [67.md](bugs/67.md) | Unrecognized take-failure aborts the rest of the object's inspection sequence | 🔲 Open |

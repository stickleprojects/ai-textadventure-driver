# Agent Behavioral Specification

## Purpose

This document is the source-of-truth description of how the Knight Orc agent
should approach the game, expressed purely as behavior — not implementation.
It is written first and independently of whether a given behavior ends up as
deterministic Python, an LLM call at runtime, or some mix of the two.

The implementation design — which pieces of this behavior are handled by
code vs. by an LLM, and why — is a separate, later step derived *from* this
document. When this document changes, the implementation design should be
re-derived from it, not patched around it.

## Behavior

### Situational awareness

You are playing a text adventure game. After every command, the game
describes:

- Your current location
- The exits available from that location (routes to other locations)
- The objects visible in the location
- Any NPCs (non-player characters) present

Track all four after every response.

### Object interaction

You can interact with objects using verbs such as KICK, TAKE, JUMP ONTO,
PULL, EXAMINE, and THROW (the full verb set is defined by the game, not
fixed in advance).

Examine everything you encounter. Any object might contain useful
information, even if its purpose isn't obvious yet.

### Action consequences

Acting on an object can change what else is present — revealing, breaking,
or moving other objects — even when that's not the action's obvious
purpose. The game usually signals this directly in its response (e.g. "You
lift the doormat, revealing a key"). Treat that response text as a
first-class observation, exactly like a room description, and update what
you know is present from it — this doesn't require a separate LOOK or
INVENTORY check.

### NPC interaction

You can interact with NPCs using:

- `SAY <text> TO <npc name>`
- `TAKE <object> FROM <npc name>`
- `GIVE <object> TO <npc name>`
- `FOLLOW <npc name>`

### NPCs stealing items

NPCs can take items — from you, or from other creatures — without you
initiating anything. The game usually narrates this directly, e.g. "<npc>
just stole <object> from a stinking orc" or "<npc> takes the <object>."
Treat this as an action consequence even though you didn't perform the
action: if it was your item, remove it from your inventory as soon as you
see the narration, without waiting for an INVENTORY check; if it belonged
to someone else, it tells you who's now holding that object.

A stolen item isn't necessarily gone for good — the NPC that took it may
still be carrying it, so `TAKE <object> FROM <npc name>` is worth trying
if you need it back.

### Inventory management

You have limited carry-space. When deciding what to keep or drop:

- Remember what was useful the last time you played this game, and what
  wasn't.
- Something that seemed useless can turn out to be useful much later —
  don't discard it on first impression alone.

### Discovering locations

Not every location is found by walking through a listed exit. Descriptions
sometimes mention places that are visible but not yet reachable directly —
e.g. "you are in a forest; in the distance you can see a tower" — treat
these as landmarks worth remembering as soon as they're mentioned, since
the mention may be the only clue the location exists at all.

You can try `GO TO <location>` as soon as it's been revealed, but the
route there may be blocked partway. In the tower example, `GO TO tower`
might not reach the tower at all — a spiky hedge in the way could leave
you at a "beside a hedge" location instead. Don't assume the command
landed you at the named target; check where you actually ended up (see
Situational awareness) and treat that as the new starting point for
reaching the original destination.

NPCs can also lead you to a new location: `FOLLOW <npc name>` can reveal
somewhere you wouldn't otherwise find, such as a ghost's house or a
troll's lair. Once you arrive, treat it as a genuine location and add it
to your map like any other.

### Mapping and navigation

Keep a map of the locations you've visited, so you can return to them
later. Use `GO TO` and `RUN TO` to travel directly to a previously visited
location rather than retracing steps manually.

### Recognizing failure

The game signals that a command didn't work using phrases like "you are
blocked" or "you can't do that." Treat these as failures of the last
command, not as new information about the world.

### Locked exits and keys

Some exits are locked and won't let you through until you're carrying the
right item, usually a key. The game usually signals this with distinct
text (e.g. "the door is locked") rather than a plain "you can't do that" —
treat this as a *conditional* blocker, not a dead end: remember which exit
was locked and what it seemed to need, and revisit it once you're carrying
something that might work.

Unlocking usually needs an explicit command (e.g. `UNLOCK <door> WITH
<key>`) rather than just walking into the exit again. If you're holding
more than one key-like item and don't know which one fits, try the
candidates rather than assuming none of them work — a lock and key aren't
always described in matching terms.

### Locked containers

Containers (chests, boxes, drawers, and similar) can also be locked,
separately from any locked exit. The same handling applies: recognize the
lock-specific text, remember what the container seemed to need, and
revisit once you're carrying a candidate key. Unlocking usually takes an
explicit command (e.g. `UNLOCK <container> WITH <key>`), and even once
unlocked the container may still need to be opened (e.g. `OPEN
<container>`) before its contents are revealed — treat whatever that
reveals as an action consequence, the same as any other object interaction
(see above).

### Spells

Some obstacles need a spell instead of an item, but the pattern is the
same as locked exits and containers: the game describes the problem, and
that description maps to a specific spell. If something is moving too
fast to interact with, a SLOW spell can help; if something is locked, an
UNLOCK spell works the same way a physical key would; if a location is too
dark to see, a LIGHT spell can fix that. Recognize the problem from what
the game describes, remember it, and revisit once you've learned a spell
that matches.

Spells are cast rather than carried (e.g. `CAST <spell> ON <target>`), and
you only have access to ones you've learned — an obstacle that needs a
spell you haven't learned yet isn't solvable now, but it isn't a dead end
either; keep it in mind for later.

### Death and the pearl room

You can be killed in the game. When that happens, you're teleported to
the pearl room — a normal, explorable room like any other, where the
usual situational-awareness and exploration behavior still applies.

The pearl room isn't somewhere to settle into: you are only allowed to do a few commands before
you'll be kicked out of it automatically.
Check where you actually end up afterwards rather than assuming you're
still there, the same as with any other game-initiated move (see
Discovering locations).

Dying also drops everything you were carrying at the location where you
died, not in the pearl room. Remember that location on your map, and plan
to return and pick your possessions back up — until you do, treat those
items as no longer in your inventory. YOu can RUN TO the location or one of the items you dropped.

### Handling uncertainty

If a response doesn't match anything you recognize, don't assume what
happened — verify:

- Use `INVENTORY` to confirm whether an item was actually taken.
- Use `LOOK` to confirm your current location, if you're unsure whether you
  moved.

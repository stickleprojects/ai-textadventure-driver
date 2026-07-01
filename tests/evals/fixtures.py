EVAL_CASES = [
    # room hallucination: terse take confirmation should NOT produce a room name
    {
        "id": "taken_no_room_hallucination",
        "action": "take putty knife",
        "game_output": "Taken.",
        "expected": {"added_to_inventory": ["putty knife"]},
        "absent": ["room"],
    },
    # terse failure response should produce no room
    {
        "id": "failure_no_room_hallucination",
        "action": "examine putty knife",
        "game_output": "You can't do that.",
        "expected": {},
        "absent": ["room"],
    },
    {
        "id": "horse_is_npc",
        "action": "look",
        "game_output": "You are in the Courtyard. A horse stands nearby. You can see a stone and a sword.",
        "expected": {"objects": ["stone", "sword"], "npcs": ["horse"]},
        "must_not": {"objects": ["horse"]},
    },
    {
        "id": "taken_confirmation",
        "action": "take sword",
        "game_output": "Taken.",
        "expected": {"added_to_inventory": ["sword"]},
    },
    {
        "id": "no_phantom_inventory",
        "action": "look",
        "game_output": "You can see a key on the table.",
        "expected": {"added_to_inventory": []},
    },
    {
        "id": "exits_extracted",
        "action": "look",
        "game_output": "You are in the Forest Path. Exits: north, east.",
        "expected": {"exits": ["north", "east"]},
    },
    {
        "id": "knight_is_npc",
        "action": "look",
        "game_output": "A huge knight blocks the doorway. There is a key on the floor.",
        "expected": {"objects": ["key"], "npcs": ["knight"]},
        "must_not": {"objects": ["knight"]},
    },
    {
        "id": "exits_all_directions",
        "action": "look",
        "game_output": "You are in a dismal fairground. Exits lead in all directions.",
        "expected": {"exits": ["north", "south", "east", "west"]},
    },
    {
        "id": "sub_object_from_examine",
        "action": "examine flagpole",
        "game_output": "It is an ugly white post, a hundred feet tall. Fastened to it is a halyard.",
        "expected": {"objects": ["halyard"]},
    },
    {
        "id": "handles_you_dont_need_to_use_the_word",
        "action": "take green beef",
        "game_output": "You don't need to use the word \"beef\" to finish this part of the game.",
        "expected": {"added_to_inventory": []},
    },
    {
        "id": "direction_string_not_used_as_room",
        "action": "east",
        "game_output": "You go east.",
        "absent": ["room"],
    },
    {
        "id": "npc_name_not_used_as_room",
        "action": "look",
        "game_output": "Denzyl is here.",
        "expected": {"npcs": ["Denzyl"]},
        "absent": ["room"],
    },
    {
        "id": "outside_compound_description_is_valid_room",
        "action": "north",
        "game_output": "You go north and are outside a cave in a juniper scrubland. In the distance is a castle. Exits lead in all directions and inside.",
        "expected": {"room": "outside a cave in a juniper scrubland"},
    },
]

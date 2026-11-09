You are writing a GAME CARD. Another agent (the player) is playing a grid puzzle
game ({game}). It has cleared {levels_completed} of {win_levels} levels and
written a document on each cleared level; it is now on level {level}, after
{step} actions. Its documents are below. Write this game's card from the levels
it has cleared. The level it is playing now is not finished: leave it out.

A card is read by a reviewer that watches a player in a DIFFERENT game, one with
other rules. The reviewer skims the cards for analogies -- "this looks like the
trap in game X; the key there was Y; test it like Z" -- and passes them on. So a
card is worth something only if it helps someone stuck in another game: what
this game looked like, how it really worked, where its player lost time, and
what cracked its levels.

## The documents

- `level_stats` -- per cleared level: minutes, actions, RESETs and GAME_OVERs,
  from the game record. A level with many minutes, RESETs or GAME_OVERs had
  something hard to see; that is where the blind spots are.
- `level_document` -- one per cleared level, written by the player right after
  the clear: the level's goal, the rules it relied on, the actions that cleared
  it, what did not work, and earlier beliefs that turned out wrong. These are
  your main source.
- `notes` -- what the player believes now. Take from it only what the cleared
  levels confirmed.
- `previous_card` (only when there is one) -- the card written for this game
  earlier, before its latest clears. Update it: keep what still holds, correct
  what the newer level documents show was wrong, add the new levels.
- `example_card` -- a finished card of another game, to show the shape and the
  amount of detail. Do not copy anything from it into this card.

{documents}

## The card

Exactly this shape, in English, at most about 900 words:

```
# {game} -- <a 3-6 word title naming the core mechanic>
Keywords: <8-15 comma-separated mechanic words a reviewer would search for, not this game's private names>
Looks like: <2-3 sentences: what the first board shows and which actions exist, so a reviewer can tell when
another game looks similar>

## Rules
<4-8 bullets: how the game works, as the cleared levels showed it: goal, what each action and object does, what
ends or costs a life, counters and budgets>

## Levels
<one line per cleared level: what is new in it and its key idea, with the minutes from level_stats>

## Blind spots -- where the player lost time
<2-6 items, the costliest first, each 2-4 lines:
 **Believed:** what the player believed or kept doing; **Actually:** what the clear showed;
 **Cue:** what on screen should have tipped it off; **Test:** the cheapest few-action probe that settles it.>

## Breakthroughs
<1-5 items: what the player realised or did differently that cracked the hard levels>

## Lessons that carry to other games
<3-6 lines with NONE of this game's nouns, each grounded in a blind spot or breakthrough above, for example "an
action can follow different rules for the object you control and for the objects it hits -- check both".>
```

Rules for what the card may say:

- Everything on the card comes from the cleared levels: what a level document
  says the clear relied on. Leave out the level the player is on now and its
  unconfirmed guesses.
- Blind spots come from what the level documents say did not work, was
  confounded, or was wrong earlier, and from the levels with the most minutes,
  RESETs or GAME_OVERs. Say what each one cost when level_stats shows it.
- No action sequences and no coordinates. Numbers only when they define a rule.
  The player's routine, correct play is not interesting; the card is about what
  was hard to see.
- A lesson is a thing to check, not a law: "check whether", not "always".

Reply with the card only: start with the `#` title line, write nothing before
or after the card, and do not put it in a code block.

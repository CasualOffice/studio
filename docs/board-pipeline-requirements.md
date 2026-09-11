# What the picture-board pipeline has to do

Written from defects found by using it, not from a plan. Each line below was
confirmed in the code before it was written down.

## Confirmed defects in the current build

**Length is guessed, not derived.** The panel count is a slider, defaulting to
six, capped at twelve. It is chosen before the story has been read. A long
story is therefore compressed into six panels and most of it is lost, silently.
`Storyboard.tsx:49`, engine cap at `worker.py` `op_shotlist`.

**Characters are not extracted.** There is no character-extraction code at all.
The user types a description by hand, and the tool refuses to start without
one: a hard return at `Storyboard.tsx:122`. A story that names its people is
read as though it named nobody.

**Only one character exists.** The board holds a single character sheet. A
story with two people in a room cannot be drawn correctly, because the second
person is invented fresh in every panel.

**Coverage is not verifiable.** The division either covers the story or does
not, and nothing tells the user which. There is no report of what was kept and
what was compressed.

**Correction is re-rolling, not fixing.** Redraw rebuilds the identical prompt
and bumps the seed, so a wrong frame can only be re-rolled, not corrected. The
prompt that produced a frame is composed at draw time and discarded: it is
stored nowhere and shown nowhere. When a frame comes out wrong there is no way
to see what asked for it.

## What it has to do instead

1. **Derive length from the story.** The story decides how many panels it
   needs. The user adjusts that number, rather than authoring it.

2. **Extract the cast.** Read the characters out of the prose, including the
   same person under different names. Where the prose does not describe
   someone, say so and offer a description to accept or replace, rather than
   refusing to start.

3. **Hold more than one character.** A panel names who is in it; each of them
   is drawn against their own sheet.

4. **Report coverage.** Show what was kept and what was compressed, so a story
   losing its middle is visible before a minute of drawing is spent on it.

5. **Correct a frame, not re-roll it.** Every frame keeps the exact prompt that
   produced it, visible and editable. A correction changes that prompt and
   redraws that frame alone. Re-rolling the seed stays available, but it is the
   lesser tool and should not be the only one.

## Constraints any design has to live inside

- A 4B text model running locally. Stages needing frontier reasoning are out.
- Roughly 45 seconds per panel. A heuristic that yields sixty panels for a
  short story is a forty-five minute run, and the tool has to say so up front
  rather than discovering it partway.
- Every stage reviewable and correctable before image generation starts, since
  that is where the time goes.
- Arithmetic belongs in code, where it can be tested, rather than being asked
  of a model.

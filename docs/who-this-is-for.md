# Who pastes a story into this, and what they actually want

Written before any architecture, because the last attempt at requirements was
a list of broken things, which is a developer's view of a product and not a
user's.

## Start with why they are here at all

Someone with a budget and no secrets uses a hosted tool. It is faster, it is
someone else's problem, and it has a nicer website. So a person who chooses a
local one has a specific reason, and there are only three:

**Their material cannot be uploaded.** Unpublished work. A client's property.
A manuscript that is not theirs to share. For these people the choice is not
local-versus-hosted, it is local-or-nothing.

**The volume makes metered pricing absurd.** A comic chapter is not one image,
it is sixty. At hosted prices, illustrating a novel is a budget line. Someone
adapting their own long work needs the marginal image to cost nothing.

**They want to keep the tool.** Models change under you. Terms change under
you. A person mid-way through a two-year project cannot have the art change
because a vendor shipped an update.

All three are people with a **long** piece of work and a **continuing**
relationship to it. That matters more than any feature: this is not a tool for
one picture, it is a tool for chapter eleven of something.

## What they are actually trying to do

They have a story. It already exists, in a document or in their head. They can
see it. What they cannot do is draw it, or cannot draw it sixty times.

So the job is not "generate images from text". The job is:

> **Make the thing I can already see.**

That single sentence rules on nearly every design question, because it means
the tool is not being creative on the user's behalf. It is *reading*. Every
place the tool asks the user to invent something the story already contains is
the tool failing at its job and passing the work back.

Success is one sentence from the user, and it is not "that looks good". It is:

> **"That's my story."**

## What that makes them expect

People do not arrive with a mental model of a pipeline. They arrive with a
mental model of **handing a story to an artist**, because that is the only
version of this transaction that has ever existed.

What an artist does with a manuscript:

- **Reads it.** They do not ask you to describe your protagonist. It is in
  chapter one. Asking is a sign they have not read it.
- **Comes back with their read.** "I make this about twenty pages. There are
  three people in it. I picture her like this." That read is a proposal, and
  you correct it.
- **Draws once you agree.** Not before.
- **Takes notes.** "Her coat is green, not blue." "She should look frightened
  here." You say it in words, about one panel, and that panel changes. You do
  not say "try again and hope".

Every one of those is missing or inverted in what exists now. The tool asks
the user to describe a character the story already describes. It asks for a
page count before it has read a word. And when a frame is wrong it offers to
roll the dice again rather than take a note.

## What they do not care about

Worth stating, because it is where effort gets wasted:

- **Panels, tiers, gutters.** These are the artist's vocabulary, not the
  author's. A writer thinks in scenes and moments. Asking for a panel count is
  asking them to do the adaptation themselves.
- **Which model.** They care that it looks right and stays looking right.
- **Seeds.** A seed is not a creative control, it is an implementation detail
  leaking into the interface.
- **Steps, guidance, quantisation.** Not one of these belongs in front of
  someone adapting a novel.

## Where they give up

The failure modes that lose this person, in order of how quickly they lose
them:

1. **It does not look like the person in their head.** This is fatal and it is
   fatal immediately. Everything else is recoverable.
2. **The characters change between panels.** The single thing that makes the
   output unusable as a comic rather than as a set of pictures.
3. **Setup before seeing anything.** If the first twenty minutes are forms and
   downloads with nothing on screen, they leave.
4. **Blind iteration.** Twenty minutes of drawing, five frames wrong, and each
   fix is another roll of the dice. This is where a working tool still gets
   abandoned.
5. **The middle of the story missing.** They notice, and it destroys trust in
   everything the tool did not show them.

## What this implies, before any architecture

- The tool reads first and asks second. Anything the story states, it must
  extract. Anything the story leaves open, it proposes and the user corrects.
- Its first output is a **read**, not a picture: who is in this, how long it
  is, what happens. Cheap, fast, wrong in ways that are obvious and fixable.
- The expensive work happens last, after the user has agreed with the read.
- Notes are given in words, about one frame, and change that frame.
- Time has to be stated before it is spent, in minutes, honestly.
- Nothing about the machinery appears in the interface unless the user asks.

## The question worth settling

Which of the three reasons above is the primary one determines a lot. A person
adapting their own unpublishable manuscript needs completeness and fidelity. A
person publishing webtoon chapters weekly needs speed, consistency across
episodes, and export. A person illustrating a child's story needs neither and
wants warmth and control.

They are the same pipeline with different defaults, but they are not the same
product, and the difference decides what is built first.

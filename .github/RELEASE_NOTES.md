## What changed since 0.1.0

**The board is a workspace, not a form.** Your prose on the left, the panels in
the middle, and on the right the exact words that produced the selected panel.
Clicking a panel lights the prose it came from and clicking the prose selects
its panel, so the comparison you actually make costs nothing.

**It reads the story.** Paste prose and the cast comes out of it — people and
places, ranked by how often each appears. Every name is checked against your
text, so a character the model invented is dropped rather than drawn into a
story it was never in.

**The panel count is derived, not chosen.** The slider is gone. The story is
divided by beat and the number is reported, along with the range that much
prose usually needs, and it says so when the division falls outside it.

**Panels are drawn a few at a time.** The button prices itself — "Draw 4 of 14",
then "Draw next 4" — and skips panels that already have a picture. A style you
do not like costs three minutes to discover instead of half an hour.

**A wrong frame is fixed by saying what is wrong with it.** Write "older, grey
at the temples" on a panel and redraw only that one. The note stays with the
frame through every later redraw.

**A board is one thing in the library.** One entry, with a count, that opens as
the comic in reading order with its reference sheets. Export or delete the whole
of it in one action. Panels, sheets and rooms are no longer seventeen loose
pictures sharing a timestamp.

**Upscaling finishes instead of failing.** A 1024 px output needs 16.57 GiB,
which a 16 GB machine does not have, and going over used to abort the engine
outright rather than raise an error. Anything too large for one pass is now done
in overlapping pieces — 1024 px at 10.11 GiB — with the picture filling in as it
goes.

**Live previews** while upscaling and generating video, multi-select in the
library with export and delete across a whole selection, and errors that stay on
screen with a Copy button, because an error you cannot copy is an error you
cannot report.

**More than one person in a scene.** The shot list names who is visible in each
panel, those names are checked against the story, and up to four share one
labelled lineup reference — so a second character is conditioned on the same
identity rather than invented fresh every frame.

**The board is a workflow with gates** — read, plan, prepare, draw, compose —
rather than buttons that can be pressed in any order. Batches resume, coverage
is reported against the source, and state that no longer matches the story is
invalidated instead of quietly reused.

**Drafts are encrypted**, and the sensitive data that was sitting in browser
storage is migrated out of it. Vault ids, exports, masks, temporary plaintext
and token storage are hardened, and the vault locks while a job is running.

**The Python runtime is pinned by checksum** and its dependency set locked
outright, along with CI's own dependencies and the actions it calls.

## What does not work

**Animating a still.** Wan 2.2 TI2V-5B fits — 9.72 GiB measured for a complete
run — but every clip so far has come back as coloured noise. It was run far
below the resolution and step count the model asks for, and whether any setting
on a 16 GB machine is both watchable and worth the wait is an open question.
Text to video works. The catalog says which is which.

**Signing and notarisation.** The build is ad-hoc signed, so macOS quarantines
it and the command above is required. Proper notarisation is untested.

**A generation smoke test in CI.** Nothing in the pipeline actually runs a model
— that needs multi-gigabyte weights and real GPU time — so every check here is
of the code around the models, not of a picture.

**Six unmaintained transitive crates** arrive through Tauri. `cargo deny`
offers no safe upgrade, so they are watched rather than fixed.

## Installing

Download the `.dmg`, open it, and drag **Model Studio** to Applications.

**It will refuse to open the first time.** macOS says the app "is damaged and
can't be opened" for anything not signed by a paid Apple Developer account.
It is not damaged; it is unsigned. Clear the quarantine flag once:

```sh
xattr -dr com.apple.quarantine "/Applications/Model Studio.app"
```

Then open it normally. You only need to do this once per install.

## What happens on first run

The app has no models and no Python inside it. On first launch it sets up a
private runtime under `~/Library/Application Support/com.melp.modelstudio`,
which is about **3 GB** and takes a few minutes. Nothing is installed
system-wide and nothing else on your Mac is touched.

It then offers to install the models that make it work — around **9 GB** for
the starter set. You can pick your own instead.

Budget roughly **12 GB of disk and twenty minutes** before you can make a
picture, most of it downloading.

## Requirements

- **Apple Silicon.** M1 or later. There is no Intel build and there cannot be
  one: MLX has no other backend.
- **macOS 13** or later.
- **16 GB of memory** to run the models in the catalog comfortably. 8 GB will
  run the smallest ones only, and the app tells you which before you download
  anything.
- **Disk.** 12 GB to get started; more for each additional model. Sizes are
  listed before you commit to a download.

## Nothing leaves the machine

No account, no API key, no telemetry, no cloud inference. The only network
traffic is downloading models from Hugging Face, which you initiate.
Everything the app produces is encrypted at rest before it touches disk.

## Verifying the download

`SHA256SUMS.txt` is attached. Check it with:

```sh
shasum -a 256 -c SHA256SUMS.txt
```

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

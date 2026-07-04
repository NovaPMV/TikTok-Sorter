# TikTok Sorter — by NovaPMV

A local app that lets you **search a folder of videos by typing what you want**
("girl in a red dress", "dancing outside", "close up") and instantly get a grid
of matching clips. Select the ones you like and copy the originals into a folder,
ready to drop into your editor. You can also just browse everything and copy
without searching.

It runs **entirely on your own computer**. Once you've used it once and the model
is downloaded, it can run offline. If you have an NVIDIA graphics card it uses it 
automatically for speed; otherwise it runs on your CPU (slower, but works fine).

**This app works best for tiktoks only**. Since it generates frames for each video, 
indexing will take up more space for longer videos. If you do however want to use 
it for longer videos, then I recommend looking at the section below explaining how 
to change some settings to optimize disk space. 

---

## What you need (one-time setup)

You'll install three things once: **Python**, **ffmpeg**, and some **Python
packages**. Follow every step in order. It looks long because it's spelled out
for total beginners — the actual work is maybe 10 minutes plus download time.

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/ and download Python (3.10 or newer).
2. Run the installer.
3. **VERY IMPORTANT:** on the first screen, check the box that says
   **"Add Python to PATH"** before clicking Install. If you miss this, nothing
   else will work. If you're not sure whether you did, just reinstall and tick it.

To confirm it worked: open **Command Prompt** (press the Windows key, type `cmd`,
hit Enter) and type:

    python --version

If you see a version number, you're good. If it says "not recognized," Python
isn't on your PATH — reinstall and make sure you tick that box.

### Step 2 — Install ffmpeg

ffmpeg is the tool that reads your videos. In the same Command Prompt window, run:

    winget install Gyan.FFmpeg

Let it finish, then **close Command Prompt and open a new one** (this matters —
the new window is what picks up ffmpeg). Confirm it worked:

    ffmpeg -version

You should see version text. If `winget` isn't recognized on your machine, get
ffmpeg from https://www.gyan.dev/ffmpeg/builds/ (the "release full" build),
unzip it, and add its `bin` folder to your PATH — or just ask in the Discord and
someone will walk you through it.

### Step 3 — Install the Python packages

In Command Prompt, run these **one line at a time**. Wait for each to finish
before running the next. Doing them one by one avoids errors.

    pip install numpy
    pip install pillow
    pip install fastapi
    pip install "uvicorn[standard]"
    pip install open_clip_torch

Now install **torch** (the AI engine). **Pick ONE of these:**

**Option A — you HAVE an NVIDIA graphics card (recommended, much faster):**

    pip install torch --index-url https://download.pytorch.org/whl/cu121

**Option B — you do NOT have an NVIDIA card (or Option A fails):**

    pip install torch

That's it for setup.

### Step 4 — Get the app files

Download this repository (green **Code** button → **Download ZIP**), and unzip it
somewhere easy like your Desktop. You should have `clipfinder.py`, the launcher
`.bat`, and this README together in one folder. **Keep them together.**

---

## Running it

Double-click **`Start TikTokSorter.bat`**.

- A black Command Prompt window opens. **Leave it open.** That window *is* the
  program. It's not an error — it's the engine running.
- Your browser opens to the app automatically. If it doesn't, open your browser
  and go to **http://localhost:8000** yourself.

**When you're finished, close that black window to shut the app down.**
Reopen the `.bat` next time. That's the whole on/off switch.

### First launch is slow — this is normal

The **very first time** you index a folder, the app downloads the AI model
(about **1.5 GB**, one time — it's cached forever after). Then it has to look at
every video, which takes a while. ***I recommend choosing a folder with under 100 tiktoks for your first index***. A folder of a few hundred clips can take
**many minutes to an hour+**, especially on CPU. **Let it run.** The progress bar
shows a live count (e.g. "37 / 200"). You can hit **Cancel** anytime — it keeps
everything already done, and next time it only processes what's left.

**Updates after that are fast** — it only processes new or changed videos when you 
manually update the index.

---

## How to use it

1. **Set your four folders.** Click **Browse** next to each (or paste the path):
   - **Source folder** — where your videos are.
   - **Frames folder** — an empty folder the app fills with tiny still images it
     uses to "see" your videos. You never open this; it's behind-the-scenes.
   - **Previews folder** — where the little preview videos are stored.
   - **Destination folder** — where copies of the clips you pick get sent.

   Then click **Save folders**. (The app remembers these next time — you only
   redo this when you want to change a folder.)

   > **Heads-up:** the Browse folder-picker sometimes opens *behind* your browser
   > window. If you click Browse and nothing appears, **Alt+Tab** to find it.

2. Click **Build / update index** and wait (see "first launch is slow" above).

3. **Search** by typing a phrase and pressing Enter — or just browse everything
   that's already indexed without searching.

4. **Click a preview to select it** (blue outline). Click again to deselect.
   Already-copied clips grey out so you don't copy duplicates.

5. Click **Copy selected to destination** to copy the originals into your
   destination folder. Or click **Open File** on any clip to play the original
   in your normal media player.

> **Note:** moving to another page clears your current selection — so copy the
> ones you want from a page *before* clicking Next.

---

## How much disk space does it use?

The app makes small **frames** and a **preview** for each video. These are tiny
compared to your source files. Rough estimates for a **10-second
vertical TikTok**:

| What | Size for one 10s clip |
|---|---|
| Frames (1 every two seconds, 5 total) | ~**50–150 KB** total (~10–30 KB each) |
| Preview (full length, 320px wide, muted) | ~**60–500 KB** |

So a 10-second clip adds very roughly **0.2–0.8 MB** of app data. **1,000 clips ≈
a few hundred MB to ~1 GB**, depending on how busy/detailed the footage is
(fast motion and lots of detail = bigger files). Longer clips scale up
proportionally. If space gets tight, see the tuning options below.

---

## Optional Tuning - Making it faster or more accurate

All of these are simple edits near the **top of `clipfinder.py`**. Open it in any
text editor (Notepad works). Change a value, save, and **rebuild the index** for
it to take effect.

> **Whenever you change the model or the frame rate, do a clean rebuild:** delete
> `clipfinder_index.json` and `clipfinder_vectors.npy` from your **Frames folder**
> (and you can clear the old frames/previews too), then click Build / update index.
> Otherwise old data mixes with new and your counts look wrong.

### Faster indexing — use a lighter model

Find these lines near the top:

    CLIP_MODEL_NAME = "ViT-L-14"
    CLIP_PRETRAINED = "laion2b_s32b_b82k"
    # CLIP_MODEL_NAME = "ViT-B-32"
    # CLIP_PRETRAINED = "laion2b_s34b_b79k"

The app ships with **ViT-L-14** (most accurate, but slower to index). For faster
indexing at slightly lower search precision, **comment out the two L-14 lines and
uncomment the two B-32 lines** (swap which pair has the `#` in front):

    # CLIP_MODEL_NAME = "ViT-L-14"
    # CLIP_PRETRAINED = "laion2b_s32b_b82k"
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

Rough guide to your options (all run locally, all free):
- **ViT-B-32** — fastest, smallest download (~600 MB), good for most searches.
- **ViT-L-14** — the default. Slower, ~1.5 GB, noticeably more precise.
- **ViT-H-14** (`laion2b_s32b_b79k`) — even more accurate, but big and slow;
  only worth it on a strong GPU with lots of VRAM.

If you switch to a model not listed here, you may also need to update the
`MODEL_VECTOR_DIM` line (768 for L-14/H-14, 512 for B-32) — the file has a comment
explaining it.

### More accurate indexing — sample more frames

Find:

    FRAME_INTERVAL_SECONDS = 2

This means "look at 1 frame every 2 seconds." Change it to `1` (one frame every 1
second) to roughly **double** indexing time and frame disk usage. This will give you more
accurate searches, with the tradeoff being more storage needed and a longer indexing time. 
`1` is a good default; `2` is fine for most footage; `3`+ starts to skip things.

### Smaller previews

Find:

    PREVIEW_WIDTH = 320

Lower it (e.g. `240` or `200`) for smaller preview files and less disk use. Higher
(e.g. `400`) for sharper previews that take more space. `320` is a good balance.

You can also change how long previews are. Find:

    PREVIEW_SECONDS = None

`None` means the preview is the **full clip**. Set it to a number (e.g. `4`) to
cap previews at that many seconds — smaller files, but you only see the start of
each clip.

---

## Switching between GPU and CPU

The app uses your NVIDIA GPU automatically **if** you installed the GPU version of
torch. **How to check which you have:** look at the black Command Prompt window
when the app starts. It prints one of:

    Running on GPU: NVIDIA GeForce GTX 1660 SUPER
    Running on CPU (no GPU detected)

**To switch from CPU to GPU** (you have an NVIDIA card and want the speed):

    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu121

**To switch from GPU back to CPU** (e.g. you moved to a machine with no NVIDIA
card):

    pip uninstall -y torch
    pip install torch

Restart the app after switching. No code changes needed — it detects the right
one on its own.

---

## Common problems (read this before asking)

- **"python is not recognized"** → Python isn't on your PATH. Reinstall Python
  and tick **"Add Python to PATH"** on the first screen.
- **"ffmpeg is not recognized"** → You didn't open a *new* Command Prompt after
  installing ffmpeg. Close it and open a fresh one.
- **A package failed to install** → Install them one line at a time (Step 3), not
  all in one line.
- **The app page won't load / "can't connect"** → Give it a few seconds after
  launch and refresh. The server takes a moment to start on the first run.
- **It says "Running on CPU" but I have an NVIDIA card** → You installed the CPU
  torch. Use the "switch from CPU to GPU" commands above.
- **The Browse button does nothing** → The folder picker opened behind your
  browser. **Alt+Tab** to find it. Or just paste the folder path into the box.
- **The indexed count looks wrong (too high)** → Leftover data from an earlier
  run with different settings. Do a clean rebuild (delete `clipfinder_index.json`
  and `clipfinder_vectors.npy` from your Frames folder, then reindex).
- **The black window closed and the app stopped** → That window is the program.
  Keep it open while using the app; reopen the `.bat` to start again.
- **Windows SmartScreen or antivirus warns about the `.bat`** → It's a plain text
  launcher you can open in Notepad to inspect. Allow it if you trust the source.

---

## A note from Nova

I probably spent less than 4 hours vibecoding this, so feel free to rip this apart
and add any new features (or fix any bugs) that you see fit. There are still some 
features that I'd like to add in the future (OCR for detecting captioned videos, 
whether or not a person is in-frame, color matching, etc.)... but for now this does 
the job for me. If you have any issues or suggestions, feel free to message me on 
Discord @novapmv. Hope you get some use out of this project!

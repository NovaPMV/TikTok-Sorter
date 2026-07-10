# TikTok Sorter — by NovaPMV

<img width="1280" height="600" alt="TikTokSorterGIF" src="https://github.com/user-attachments/assets/577ebc3b-7035-413b-b46f-f9b2a31617e9" />

A local app that lets you **search a folder of videos by typing what you want**
("girl in a red dress", "dancing outside", "close up") and instantly get a grid
of matching clips. Select the ones you like and copy the originals into a folder,
ready to drop into your editor. You can also just browse everything and copy
without searching.

Now supports **multiple source folders at once** — point it at several TikTok
accounts and search across all of them together, or filter down to just one.

It runs **entirely on your own computer**. Once you've used it once and the model
is downloaded, it can run offline. If you have an NVIDIA graphics card it uses it
automatically for speed; otherwise it runs on your CPU (slower, but works fine).

**This app works best for tiktoks only**. Since it generates frames for each video,
indexing will take up more space for longer videos. If you do however want to use
it for longer videos, then I recommend looking at the section below explaining how
to change some settings to optimize disk space.

---

## What's new

If you used an older version, here's what got added:

- **Multiple source folders.** Add as many account folders as you like. They all
  get indexed into one searchable library that shares a single Frames and
  Previews folder.
- **All folders / Single folder tabs.** Search across every account at once, or
  switch to a single folder from a dropdown — instantly, without re-indexing.
- **Open Source button.** Opens the clip's folder with the file already
  highlighted, so you can drag it straight into Premiere (or anywhere else).
- **Show Copied toggle.** Turn the grey "copied" overlay on or off. Off lets you
  copy the same clip more than once.
- **Sort dropdown.** Sort the grid by relevance, filename, or duration.
- **Total video counter** next to the page controls, so you always know how many
  clips are in view (handy when switching between all folders and one folder).
- **Top and bottom page navigation**, so you don't have to scroll back up to
  change pages.
- **Moving/renaming folder.** Now you can move or rename folders without needing
  to re-index. Just make sure you click "Save Folders" after setting new paths,
  otherwise you risk splitting your indexing across two separate Frame folders.
  
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
every video, which takes a while. ***I recommend choosing a folder with under 100
tiktoks for your first index***. A folder of a few hundred clips can take
**many minutes to an hour+**, especially on CPU. **Let it run.** The progress bar
shows a live count (e.g. "37 / 200"). You can hit **Cancel** anytime — it keeps
everything already done, and next time it only processes what's left.

**Updates after that are fast** — it only processes new or changed videos when you
manually update the index.

---

## How to use it

### 1. Set your folders

Each folder has a **Browse** button (or you can paste the path):

- **Source folders** — where your videos are. **You can add more than one.**
  Click **+ Add source folder** for each TikTok account (or any folder of clips)
  you want included. Each row has its own Browse and Remove button. They all get
  indexed together into one searchable library.
- **Frames folder** — an empty folder the app fills with tiny still images it uses
  to "see" your videos. You never open this; it's behind-the-scenes. **One shared
  Frames folder holds everything, even with multiple sources** — clips can't
  collide because each is stored under a unique ID.
- **Previews folder** — where the little preview videos are stored (also shared
  across all sources).
- **Destination folder** — where copies of the clips you pick get sent.

Then click **Save folders**. The app remembers all of these next time — including
your whole list of source folders — so you only redo this when you want to change
something.

> **Heads-up:** the Browse folder-picker sometimes opens *behind* your browser
> window. If you click Browse and nothing appears, **Alt+Tab** to find it. Or just
> paste the folder path into the box.

### 2. Build the index

Click **Build / update index** and wait (see "first launch is slow" above). It
walks every source folder you added, including subfolders. Adding a new account
later only indexes that account's new clips — it doesn't redo the old ones.

### 3. Search or browse

Type a phrase and press Enter to search — or just browse everything that's already
indexed without searching.

**All folders vs. Single folder:** use the tabs above the grid.

- **All folders** searches and browses across every account at once.
- **Single folder** shows a dropdown — pick one account and the grid narrows to
  just that folder. Switching is instant and never re-indexes. Great for when you
  only want to pull from one creator.

**Sort by** lets you order the grid by relevance (best matches first, for
searches), filename (A–Z or Z–A), or duration (longest or shortest first).

### 4. Select the clips you want

**Click a preview to select it** (it gets a bright blue glowing border). Click
again to deselect. Use **Select all on page** or **Clear selection** for bulk
picking. The running count shows how many you've selected.

By default, clips you've already copied this session grey out with a "copied" tag
so you don't copy duplicates. If you actually *want* to copy something again, click
**Disable Show Copied** — that turns off the grey overlay and lets you re-select
copied clips. Click it again ("Show Copied") to turn the greying back on.

### 5. Copy or open

- **Copy selected to destination** copies the original files into your destination
  folder, ready for editing.
- **Open File** on any clip plays the original in your normal media player.
- **Open Source** opens the clip's folder with the file already highlighted — the
  easiest way to drag it straight into your editor.

> **Note:** moving to another page clears your current selection — so copy the ones
> you want from a page *before* changing pages. Page navigation is available at both
> the top and bottom of the grid.

---

## How much disk space does it use?

The app makes small **frames** and a **preview** for each video. These are tiny
compared to your source files. Rough estimates for a **10-second vertical
TikTok**:

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
second) to roughly **double** indexing time and frame disk usage. This will give
you more accurate searches, with the tradeoff being more storage needed and a
longer indexing time. `1` is a good default; `2` is fine for most footage; `3`+
starts to skip things.

### Change how many clips show per page

The grid shows **24 clips per page** by default. You can change this live with the
**Per page** slider in the app (no code editing needed). If you want a different
starting default, find this line near the top of `clipfinder.py`:

    DEFAULT_PER_PAGE = 24

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

Restart the app after switching. No code changes needed — it detects the right one
on its own.

---

## Uninstalling / cleaning up

There's no installer, so there's nothing in "Add or Remove Programs" — everything
lives in plain folders you can delete by hand. Here's every place the app leaves
something, so you can remove exactly what you want.

### 1. The app files themselves

Wherever you unzipped the download (e.g. your Desktop). Delete the whole folder —
it contains `clipfinder.py`, `Start TikTokSorter.bat`, and this README. Deleting
these stops the app from running; it does **not** touch your videos or the AI
model cache.

### 2. The AI model cache (~1.5 GB — the big one)

The CLIP model downloads once and is cached here:

    C:\Users\<your-name>\.cache\huggingface\

Replace `<your-name>` with your Windows username. This is the largest thing the app
leaves behind. Deleting it frees up the most space; the only cost is that the model
will re-download (~1.5 GB) if you ever run the app again. To reach a hidden `.cache`
folder, paste the path straight into the File Explorer address bar.

### 3. Your settings file

Sits next to `clipfinder.py` in the app folder:

    clipfinder_settings.json

This just remembers your folder paths (your source folders, frames, previews,
destination). Deleting it is harmless — the app simply forgets those paths and you
re-pick them next time.

### 4. The index (the app's "memory" of your clips)

Stored inside whatever you chose as your **Frames folder**:

    clipfinder_index.json
    clipfinder_vectors.npy

These are what make search instant without re-scanning. Delete them if you want the
app to forget everything it indexed (you'd rebuild the index next time).

### 5. The frames and previews

Also in your **Frames folder** and **Previews folder** — the tiny still images and
little preview videos the app generated. Safe to delete; they only get regenerated
when you re-index.

### 6. The Python packages (optional)

If you don't use Python for anything else and want the space back, you can remove
the packages this app installed:

    pip uninstall -y torch open_clip_torch fastapi "uvicorn[standard]" pillow numpy

Only do this if you're sure nothing else on your machine needs them. **torch is the
big one here** (often several GB for the GPU build), so uninstalling it reclaims a
lot of space.

> **The short version:** to reclaim the most space fast, delete the app folder and
> the `huggingface` cache in step 2. Your original videos are never touched by any
> of this — the app only ever *reads* and *copies* them, never moves or deletes.

---

## Common problems (read this before asking)

- **"python is not recognized"** → Python isn't on your PATH. Reinstall Python and
  tick **"Add Python to PATH"** on the first screen.
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
- **The Single folder dropdown is empty** → A folder shows up there once it's been
  saved or indexed. Add your sources, click Save folders, and build the index.
- **The indexed count looks wrong (too high)** → Leftover data from an earlier run
  with different settings. Do a clean rebuild (delete `clipfinder_index.json` and
  `clipfinder_vectors.npy` from your Frames folder, then reindex).
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

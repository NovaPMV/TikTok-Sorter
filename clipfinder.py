#!/usr/bin/env python3
# =============================================================================
#  ClipFinder
#  -------------------------------------------------------------------------
#  A single-file local web app that lets you search a folder of videos by
#  typing a phrase ("girls wearing jeans") and get back a grid of matching
#  clips as looping muted MP4 previews. You select the ones you want and copy
#  the ORIGINAL source files into a destination folder.
#
#  HOW IT WORKS (the whole machine in 6 lines):
#    1. INDEX: for each video, ffmpeg samples still frames (e.g. 1 every 2s).
#    2. EMBED: CLIP turns each frame into a vector (a list of numbers that
#       captures "what's in the picture"). We store these vectors.
#    3. PREVIEW: ffmpeg also makes ONE small looping muted mp4 per video.
#    4. SEARCH: your phrase is turned into a vector by the same CLIP model,
#       then compared to every frame vector. Closest = best match.
#    5. COLLAPSE: many frames belong to one video, so we keep each video's
#       best score and show ONE preview tile per video (frames never shown).
#    6. COPY: you tick the previews you want, click copy, originals are
#       duplicated into your destination folder.
#
#  Frames are backend-only machinery. The UI only ever shows previews.
#
#  This file IS the web server (FastAPI) AND serves the web page (HTML/JS).
#  Run it, it opens Chrome, you do everything from the browser.
# =============================================================================

# ----- standard library imports (these ship with Python, nothing to install) -
import os                # file paths, listing folders
import sys               # to detect platform / exit
import json              # saving + loading the index
import time              # timestamps
import shutil            # copying files
import hashlib           # making a quick fingerprint of a file (to detect changes)
import threading         # to open the browser after the server starts
import webbrowser        # to pop open Chrome automatically
import subprocess        # to call ffmpeg
from pathlib import Path # nicer path handling than raw strings

# ----- third-party imports (installed via the pip line in the setup note) ----
# If any of these fail, the script prints a friendly message telling you what
# to install, instead of a confusing traceback.
try:
    import torch                          # the deep-learning engine (uses your GPU)
    import open_clip                      # the CLIP model (image+text understanding)
    from PIL import Image                 # loading frame images for CLIP
    import numpy as np                    # fast math on the vectors
    from fastapi import FastAPI, Request  # the web backend framework
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    import uvicorn                        # the actual web server that runs FastAPI
except ImportError as e:
    print("\n[ClipFinder] A required package is missing:", e)
    print("Run this once in your terminal, then start the app again:\n")
    print('  pip install "fastapi" "uvicorn[standard]" "open_clip_torch" '
          '"torch --index-url https://download.pytorch.org/whl/cu121" '
          '"pillow" "numpy"\n')
    print("(The torch line installs the CUDA/GPU build for your NVIDIA card.)")
    sys.exit(1)


# =============================================================================
#  CONFIGURATION / CONSTANTS
#  These are the knobs. Tweak here if you want different defaults.
# =============================================================================

APP_PORT = 8000                      # the app will live at http://localhost:8000
INDEX_FILENAME = "clipfinder_index.json"   # where we save what we've indexed
FRAME_INTERVAL_SECONDS = 2           # sample one frame every N seconds of video (1 fps)
PREVIEW_SECONDS = None               # None = preview the FULL clip; or set a number of seconds
PREVIEW_WIDTH = 320                  # preview width in pixels (smaller = less disk/faster)
DEFAULT_PER_PAGE = 24                # how many preview tiles show per page by default
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

# Where we remember your four folder paths between launches. Saved next to this
# script so the boxes auto-fill next time you open the app.
SETTINGS_FILENAME = Path(__file__).parent / "clipfinder_settings.json"

# CLIP model choice.
#   ViT-L-14  = higher accuracy, the sweet spot for a 6GB card like a 1660 Super.
#               Slower to INDEX (bigger model), but search itself stays fast and
#               results are noticeably more precise. This is the default.
#   ViT-B-32  = the lighter/faster fallback. If indexing ever feels too slow on
#               your hardware, comment out the two L-14 lines below and uncomment
#               the two B-32 lines — that's the whole switch.
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"   # trained weights for ViT-L-14
# CLIP_MODEL_NAME = "ViT-B-32"
# CLIP_PRETRAINED = "laion2b_s34b_b79k"   # trained weights for ViT-B-32


# The size of the vectors the chosen model produces. ViT-L-14 = 768, ViT-B-32 = 512.
# Used only to make correctly-shaped empty arrays when the index is empty.
MODEL_VECTOR_DIM = 768 if CLIP_MODEL_NAME == "ViT-L-14" else 512


# =============================================================================
#  GLOBAL STATE
#  Things the whole app shares: the loaded model, the in-memory index, and a
#  small dict tracking indexing progress so the UI can show a progress bar.
# =============================================================================

STATE = {
    "model": None,            # the CLIP model (loaded once, reused)
    "preprocess": None,       # the function that prepares an image for CLIP
    "tokenizer": None,        # turns your search text into tokens for CLIP
    "device": None,           # "cuda" (your GPU) or "cpu"

    # The index: a list of "records", one per FRAME. Each record knows which
    # video it came from and holds that frame's CLIP vector.
    # We also keep a separate matrix of all vectors for fast searching.
    "frame_records": [],      # [{video, frame_path, vector_idx}, ...]
    "frame_vectors": None,    # numpy array, shape (num_frames, vector_size)

    # Per-video info: maps a source video path -> its preview file + a fingerprint
    # so we can skip re-indexing unchanged videos.
    "videos": {},             # {video_path: {"preview": path, "fingerprint": str}}

    # Folders the user picked in the UI (filled in from the frontend).
    #  sources  - a LIST of source folders (each an account's clips). Indexing
    #             walks all of them into one shared library; the UI can then
    #             search/browse either ALL of them or just one at a time.
    #  frames/previews/destination - single shared folders (see handoff notes).
    "folders": {"sources": [], "frames": "", "previews": "", "destination": ""},

    # Videos copied to the destination THIS SESSION. Used to grey out already-
    # copied tiles so you don't duplicate. Deliberately NOT saved to disk: it
    # resets when the server restarts, so cross-session duplicates are allowed.
    "copied_this_session": set(),

    # Live progress for the "Build/update index" button.
    #  running   - is a job going right now
    #  done/total- videos processed / videos to process this run
    #  message   - human-readable status line
    #  cancel    - set True to ask the running job to stop (keeps finished work)
    #  finished  - True once a run ends (so the UI can stop polling)
    "progress": {"running": False, "done": 0, "total": 0,
                 "message": "Idle", "cancel": False, "finished": False},
}


# =============================================================================
#  MODEL LOADING
#  Load CLIP once, on the GPU if available. Called lazily the first time we
#  need it (so the app window opens instantly and only loads the model when
#  you actually index or search).
# =============================================================================

def load_model_if_needed():
    if STATE["model"] is not None:
        return  # already loaded, nothing to do

    # Pick GPU if your CUDA torch build sees a supported NVIDIA card, else CPU.
    # This is the automatic fallback: if anything about the GPU isn't available
    # (no CUDA torch build installed, driver issue, etc.) we quietly use CPU so
    # the app still works for everyone.
    if torch.cuda.is_available():
        STATE["device"] = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "NVIDIA GPU"
        print("=" * 70)
        print(f"[ClipFinder] Running on GPU: {gpu_name}")
        print("=" * 70)
    else:
        STATE["device"] = "cpu"
        print("=" * 70)
        print("[ClipFinder] Running on CPU (no GPU detected).")
        print("  The app works fine, but indexing will be slower.")
        print("  To use your NVIDIA GPU, install the CUDA build of torch -")
        print("  see the 'GPU acceleration' section in SETUP_README.txt.")
        print("=" * 70)

    print(f"[ClipFinder] Loading model '{CLIP_MODEL_NAME}' (first run downloads it)...")

    # open_clip gives us three things: the model, an image preprocessor, and a
    # tokenizer for text. We download weights once (cached afterwards).
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(STATE["device"]).eval()   # move to GPU, set to inference mode
    STATE["model"] = model
    STATE["preprocess"] = preprocess
    STATE["tokenizer"] = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    print("[ClipFinder] Model ready.")


# =============================================================================
#  HELPERS: fingerprints, frame extraction, preview generation
# =============================================================================

def file_fingerprint(path: Path) -> str:
    """A cheap, reliable 'has this file changed?' signature: size + mtime.
    We avoid hashing the whole (large) video for speed; size+mtime is enough
    to catch new, replaced, or re-encoded files for our purposes."""
    stat = path.stat()
    raw = f"{stat.st_size}-{int(stat.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_frames(video: Path, frames_dir: Path) -> list[Path]:
    """Use ffmpeg to sample still frames from a video, one every
    FRAME_INTERVAL_SECONDS. Returns the list of frame image paths.
    These frames are ONLY for CLIP to look at — never shown in the UI."""
    # Each video gets its own subfolder of frames, named after the video, so
    # nothing collides and we can map a frame back to its source easily.
    safe_name = hashlib.md5(str(video).encode()).hexdigest()[:16]
    out_dir = frames_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # fps=1/N means "one frame every N seconds". %04d is a zero-padded counter.
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps=1/{FRAME_INTERVAL_SECONDS},scale=224:-1",  # 224px is plenty for CLIP
        "-q:v", "3",                                            # decent jpg quality
        pattern,
    ]
    # We hide ffmpeg's chatty output unless something errors.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return sorted(out_dir.glob("frame_*.jpg"))


def get_duration_seconds(video: Path) -> float:
    """Ask ffprobe how long the video is, in seconds. Used for the little
    duration tag under each preview. Returns 0.0 if it can't be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def make_preview(video: Path, previews_dir: Path) -> Path:
    """Make ONE looping muted mp4 preview for a video. This is the only visual
    the UI shows. By default (PREVIEW_SECONDS = None) it previews the FULL clip;
    set PREVIEW_SECONDS to a number to cap the length instead."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.md5(str(video).encode()).hexdigest()[:16]
    out_path = previews_dir / f"{safe_name}.mp4"

    # Scale down, strip audio (-an), and encode a small h264 mp4 that loops
    # cleanly in the browser. If PREVIEW_SECONDS is set, add a "-t" limit;
    # if it's None we encode the whole clip.
    cmd = ["ffmpeg", "-y"]
    if PREVIEW_SECONDS is not None:
        cmd += ["-t", str(PREVIEW_SECONDS)]
    cmd += [
        "-i", str(video),
        "-an",                                          # drop audio (muted)
        "-vf", f"scale={PREVIEW_WIDTH}:-2",             # shrink, keep aspect ratio
        "-c:v", "libx264", "-pix_fmt", "yuv420p",       # broadly compatible mp4
        "-movflags", "+faststart",                      # starts playing immediately
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out_path


def embed_images(image_paths: list[Path]) -> np.ndarray:
    """Run a batch of frame images through CLIP to get their vectors.
    Returns a numpy array, one row per image. Done in batches so the GPU
    stays busy without running out of memory."""
    load_model_if_needed()
    vectors = []
    batch = []
    batch_size = 32  # small enough for a 6GB card, large enough to be efficient

    def flush(batch_imgs):
        if not batch_imgs:
            return
        # Stack the preprocessed images into one tensor and move to GPU.
        tensor = torch.stack(batch_imgs).to(STATE["device"])
        with torch.no_grad():  # we're not training, so skip gradient tracking
            feats = STATE["model"].encode_image(tensor)
            # Normalize so that comparing vectors = comparing directions (cosine).
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue  # skip unreadable frames
        batch.append(STATE["preprocess"](img))
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)  # leftover partial batch

    if not vectors:
        return np.zeros((0, MODEL_VECTOR_DIM), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """Turn the search phrase into a CLIP vector, the same 'shape' as the
    frame vectors, so we can compare them directly."""
    load_model_if_needed()
    tokens = STATE["tokenizer"]([query]).to(STATE["device"])
    with torch.no_grad():
        feats = STATE["model"].encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]  # single vector


# =============================================================================
#  INDEX SAVE / LOAD
#  We persist the index to a JSON file in the frames folder so you don't have
#  to re-index every time you launch the app. Vectors are saved alongside as a
#  .npy (numpy's fast binary format) because JSON is bad at big number arrays.
# =============================================================================

def index_paths():
    base = Path(STATE["folders"]["frames"] or ".")
    return base / INDEX_FILENAME, base / "clipfinder_vectors.npy"


# ---- Remembering the four folder paths between launches -----------------------
# These live in a small settings file next to the script (separate from the
# index, which lives in the frames folder). So even before you've indexed
# anything, the app can pre-fill the boxes with what you used last time.

def save_settings():
    """Write the current four folder paths to the settings file."""
    try:
        SETTINGS_FILENAME.write_text(json.dumps(STATE["folders"]))
    except Exception as e:
        print(f"[ClipFinder] Could not save settings: {e}")


def load_settings():
    """Read the saved folder paths from the settings file, if it exists, and
    pre-fill them. Called once at startup.

    Handles the new multi-source format ("sources": [..]) and also MIGRATES an
    older settings file that had a single "source": "path" — so upgrading the
    app doesn't lose your previously-saved folder."""
    try:
        if SETTINGS_FILENAME.exists():
            saved = json.loads(SETTINGS_FILENAME.read_text())
            # sources: prefer the new list; fall back to an old single "source".
            if isinstance(saved.get("sources"), list):
                STATE["folders"]["sources"] = [s for s in saved["sources"] if s]
            elif saved.get("source"):
                STATE["folders"]["sources"] = [saved["source"]]
            # the three shared single folders
            for k in ("frames", "previews", "destination"):
                if saved.get(k):
                    STATE["folders"][k] = saved[k]
            print("[ClipFinder] Loaded saved folder paths.")
    except Exception as e:
        print(f"[ClipFinder] Could not load settings: {e}")


def save_index():
    meta_path, vec_path = index_paths()
    meta = {
        "videos": STATE["videos"],
        "frame_records": STATE["frame_records"],
        "folders": STATE["folders"],
    }
    meta_path.write_text(json.dumps(meta))
    if STATE["frame_vectors"] is not None:
        np.save(vec_path, STATE["frame_vectors"])


def load_index():
    meta_path, vec_path = index_paths()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        STATE["videos"] = meta.get("videos", {})
        STATE["frame_records"] = meta.get("frame_records", [])
        saved_folders = meta.get("folders", {})
        # Migrate an old single "source" saved inside an index to the new list.
        if "sources" not in saved_folders and saved_folders.get("source"):
            saved_folders = dict(saved_folders)
            saved_folders["sources"] = [saved_folders["source"]]
        # keep any folders the user already set this session, fill in the rest
        for k, v in saved_folders.items():
            if k == "source":
                continue  # legacy key; superseded by "sources"
            # For the sources list, only fill it if we don't already have one.
            if k == "sources":
                if not STATE["folders"].get("sources"):
                    STATE["folders"]["sources"] = [s for s in (v or []) if s]
            elif not STATE["folders"].get(k):
                STATE["folders"][k] = v
    if vec_path.exists():
        loaded = np.load(vec_path)
        # SAFETY: if this index was built with a DIFFERENT model, its vectors are
        # the wrong size (e.g. an old ViT-B-32 index = 512 dims vs ViT-L-14 = 768).
        # Mixing sizes would crash searches, so we discard the stale index and
        # tell the user to rebuild rather than load incompatible data.
        if loaded.shape[1] != MODEL_VECTOR_DIM:
            print(f"[ClipFinder] Existing index was built with a different model "
                  f"(vector size {loaded.shape[1]} vs current {MODEL_VECTOR_DIM}). "
                  f"Ignoring it - click 'Build / update index' to rebuild.")
            STATE["frame_vectors"] = None
            STATE["frame_records"] = []
            STATE["videos"] = {}   # forces a full re-index with the new model
        else:
            STATE["frame_vectors"] = loaded


# =============================================================================
#  THE INDEXING JOB (incremental)
#  This is what the "Build/update index" button triggers. It runs in a
#  background thread so the web page stays responsive and can poll progress.
# =============================================================================

def build_index_job():
    p = STATE["progress"]
    # Mark running IMMEDIATELY so the very first progress poll shows activity
    # (fixes the "bar stays Idle until you click again" issue).
    p.update({"running": True, "finished": False, "cancel": False,
              "done": 0, "total": 0, "message": "Scanning folder\u2026"})
    try:
        frames_dir = Path(STATE["folders"]["frames"])
        previews_dir = Path(STATE["folders"]["previews"])

        # Gather every source folder the user added. We index them all into one
        # shared library. Each video remembers which source it came from (its
        # "source" tag) so the UI can later filter the view to a single folder
        # without re-indexing.
        #
        # We CANONICALIZE each source path (and drop duplicates that only differ
        # by slash direction / trailing slash / case) so a folder is stored one
        # consistent way. This is what keeps the single-folder dropdown from
        # listing the same folder twice.
        raw_sources = [s for s in STATE["folders"].get("sources", []) if s]
        source_folders = []
        seen_src = set()
        for s in raw_sources:
            c = canon_path(s)
            if c and c not in seen_src:
                seen_src.add(c)
                source_folders.append(c)
        if not source_folders:
            p.update({"message": "No source folders set. Add at least one, "
                                 "click Save folders, then index."})
            return

        # Find every video across ALL source folders (recursively, incl.
        # subfolders). We keep the owning (canonical) source folder next to each.
        all_videos = []            # list of (video_path, source_folder_str)
        for sf in source_folders:
            base = Path(sf)
            if not base.exists():
                continue           # skip a folder that isn't there anymore
            for f in base.rglob("*"):
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    all_videos.append((f, sf))

        # If the same video path shows up under two source folders (e.g. nested
        # or overlapping sources), keep the first source we saw it under.
        seen_paths = set()
        deduped = []
        for v, sf in all_videos:
            if str(v) in seen_paths:
                continue
            seen_paths.add(str(v))
            deduped.append((v, sf))
        all_videos = deduped

        # INCREMENTAL: figure out which videos are new or changed since last time.
        # We also refresh the "source" tag on unchanged videos in case a folder
        # was re-pointed, so filtering stays correct.
        to_process = []
        for v, sf in all_videos:
            key = str(v)
            fp = file_fingerprint(v)
            known = STATE["videos"].get(key)
            if (known is None) or (known.get("fingerprint") != fp):
                to_process.append((v, fp, sf))
            elif known.get("source") != sf:
                known["source"] = sf   # cheap correction, no reprocessing needed

        # PRUNE: drop videos that no longer exist on disk.
        existing_keys = {str(v) for v, _sf in all_videos}
        removed = [k for k in STATE["videos"] if k not in existing_keys]
        for k in removed:
            STATE["videos"].pop(k, None)
        STATE["frame_records"] = [r for r in STATE["frame_records"]
                                  if r["video"] in existing_keys]

        total = len(to_process)
        skipped = len(all_videos) - total
        p.update({"done": 0, "total": total,
                  "message": f"Starting: {total} new/changed to index "
                             f"({skipped} unchanged)"})

        # We process videos one at a time. If the user hits Cancel, we STOP but
        # keep everything already finished (indexing is incremental, so a later
        # run resumes from here). We track exactly which videos we completed so
        # the vector matrix is rebuilt only from real, finished work.
        new_vectors_all = []
        processed_pairs = []          # the (video, fp) we actually finished
        cancelled = False
        for i, (video, fp, sf) in enumerate(to_process):
            # Check the cancel flag BEFORE starting the next video.
            if p.get("cancel"):
                cancelled = True
                break

            p["message"] = f"Indexing {i+1}/{total}: {video.name}"
            # 1) frames (backend only)
            frames = extract_frames(video, frames_dir)
            # 2) embeddings for those frames
            vecs = embed_images(frames)
            # 3) one preview for the UI
            preview = make_preview(video, previews_dir)
            # 4) how long the clip is (for the duration tag under the tile)
            duration = get_duration_seconds(video)

            new_vectors_all.append(vecs)
            processed_pairs.append((video, fp))

            STATE["videos"][str(video)] = {
                "preview": str(preview),
                "fingerprint": fp,
                "duration": duration,
                "source": sf,          # which source folder this clip belongs to
            }
            p["done"] = i + 1

        # Rebuild the master vector matrix from unchanged + newly finished videos.
        # NOTE: we pass processed_pairs (what actually completed), so a cancelled
        # run still produces a correct, consistent index of the finished work.
        rebuild_vector_matrix(new_vectors_all, processed_pairs)
        save_index()

        done_count = len(processed_pairs)
        total_in_library = len(STATE["videos"])
        if cancelled:
            p["message"] = (f"Cancelled. Indexed {done_count} of {total} this run "
                            f"(kept). {total_in_library} total in library.")
        else:
            p["message"] = (f"Done. Indexed {done_count} this run. "
                            f"{total_in_library} total in library.")
    except Exception as e:
        p["message"] = f"Error: {e}"
    finally:
        p["running"] = False
        p["finished"] = True
        p["cancel"] = False


def rebuild_vector_matrix(new_vectors_all, to_process):
    """Combine previously-stored vectors (for unchanged videos) with the newly
    computed ones, and renumber every frame record so vector_idx points at the
    right row. Kept deliberately straightforward over clever."""
    # Map: which videos were (re)processed this run -> their fresh vectors.
    processed_videos = [str(v) for v, _ in to_process]
    fresh_by_video = {}
    cursor = 0
    for (video, _fp), vecs in zip(to_process, new_vectors_all):
        fresh_by_video[str(video)] = vecs

    # Start from any existing matrix (for unchanged videos we keep old rows).
    old_matrix = STATE["frame_vectors"]
    old_records = [r for r in STATE["frame_records"] if "vector_idx" in r and r.get("vector_idx") is not None]

    rows = []
    new_records = []

    # Keep old rows for videos that were NOT reprocessed and still exist.
    if old_matrix is not None:
        for r in old_records:
            if r["video"] in fresh_by_video:
                continue  # will be replaced by fresh vectors
            if r["video"] not in STATE["videos"]:
                continue  # video removed
            idx = len(rows)
            rows.append(old_matrix[r["vector_idx"]])
            new_records.append({"video": r["video"], "vector_idx": idx})

    # Add fresh rows for reprocessed videos.
    for video, vecs in fresh_by_video.items():
        for k in range(len(vecs)):
            idx = len(rows)
            rows.append(vecs[k])
            new_records.append({"video": video, "vector_idx": idx})

    STATE["frame_vectors"] = (np.vstack(rows).astype(np.float32)
                              if rows else np.zeros((0, MODEL_VECTOR_DIM), dtype=np.float32))
    STATE["frame_records"] = new_records


# =============================================================================
#  SEARCH
#  Compare the query vector to every frame vector, then COLLAPSE to one result
#  per video (frames never reach the UI).
# =============================================================================

def canon_path(p: str) -> str:
    """Return a canonical form of a folder path so the SAME folder always
    compares equal, no matter how it was typed. Windows is case-insensitive and
    accepts both slash styles, so 'C:/tiktoks/X', 'C:\\tiktoks\\X\\' and
    'C:\\TikToks\\X' must all collapse to one string. Without this, the same
    folder can show up twice in the single-folder dropdown (one entry works, the
    other matches nothing). Empty stays empty."""
    if not p:
        return ""
    try:
        return os.path.normcase(os.path.normpath(str(p)))
    except Exception:
        return str(p)


def video_in_scope(video: str, source: str = "") -> bool:
    """Decide whether a video belongs in the current view.
    source="" (or "all") means the whole pooled library. Otherwise we only keep
    videos whose stored 'source' tag matches the chosen folder. We fall back to
    a path-prefix check so clips indexed before the 'source' tag existed still
    filter correctly."""
    if not source or source == "all":
        return True
    source = canon_path(source)
    info = STATE["videos"].get(video, {})
    tagged = info.get("source")
    if tagged is not None:
        return canon_path(tagged) == source
    # Fallback for older indexes without a source tag: match by path prefix.
    try:
        return canon_path(source) in {canon_path(str(par)) for par in Path(video).parents}
    except Exception:
        return False


def search(query: str, max_videos: int = 0, source: str = ""):
    """Search indexed frames for the query and return one result per video,
    ranked best-first. max_videos=0 means NO cap (return every matching video).
    source="" searches the whole library; a folder path limits it to that one
    source folder (the single-folder tab)."""
    if STATE["frame_vectors"] is None or len(STATE["frame_vectors"]) == 0:
        return []

    qvec = embed_text(query)                       # (vector_size,)
    # Cosine similarity = dot product, because everything is normalized.
    scores = STATE["frame_vectors"] @ qvec         # (num_frames,)

    # For each video, keep its single best frame score.
    best_per_video = {}
    for rec, score in zip(STATE["frame_records"], scores):
        v = rec["video"]
        if not video_in_scope(v, source):
            continue                               # skip out-of-scope folders
        if v not in best_per_video or score > best_per_video[v]:
            best_per_video[v] = float(score)

    # Sort videos by their best score, highest first.
    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1], reverse=True)
    # Only apply a cap if one was explicitly requested (max_videos > 0).
    if max_videos and max_videos > 0:
        ranked = ranked[:max_videos]

    # Build the result list the frontend will render (one tile per video).
    results = []
    for video, score in ranked:
        item = build_result_item(video, score)
        if item:
            results.append(item)
    return results


def build_result_item(video: str, score=None):
    """Make one result dict for the grid from a video path. Shared by search
    (which passes a score) and browse-all (no score)."""
    info = STATE["videos"].get(video)
    if not info:
        return None
    return {
        "video": video,                          # original source path (for copying/opening)
        "preview": info["preview"],              # the mp4 the grid shows
        "name": Path(video).name,
        "score": round(score, 3) if score is not None else None,
        "duration": info.get("duration", 0),     # seconds, for the little tag
        # whether this clip has already been copied to the destination this
        # session (so the UI can grey it out and block re-selection)
        "copied": video in STATE["copied_this_session"],
    }


def browse_all(source: str = ""):
    """Return indexed videos (no search), sorted by filename. This is the
    default view so you can browse and copy clips without typing a query.
    source="" browses the whole library; a folder path limits it to one source
    folder (the single-folder tab)."""
    videos = sorted(STATE["videos"].keys(), key=lambda v: Path(v).name.lower())
    results = []
    for video in videos:
        if not video_in_scope(video, source):
            continue
        item = build_result_item(video, None)
        if item:
            results.append(item)
    return results


def indexed_sources():
    """The distinct source folders currently represented in the index, so the
    single-folder tab can offer them as a dropdown. Falls back to the saved
    sources list so folders show up even before anything's indexed.

    Dedupes by CANONICAL path so the same folder never appears twice just
    because the saved string and the stored tag were typed differently
    (e.g. trailing slash, slash direction, or letter case)."""
    found = []
    seen = set()   # canonical paths we've already added

    def add(s):
        if not s:
            return
        key = canon_path(s)
        if key in seen:
            return
        seen.add(key)
        found.append(s)   # keep the first real (nicely-cased) string we saw

    # Prefer the paths actually stored on indexed videos (these are what
    # filtering matches against), then fill in any saved-but-unindexed folders.
    for info in STATE["videos"].values():
        add(info.get("source"))
    for s in STATE["folders"].get("sources", []):
        add(s)
    return sorted(found, key=lambda p: Path(p).name.lower())


# =============================================================================
#  THE WEB APP (FastAPI)
#  Below we define the "endpoints" — the URLs the web page talks to:
#    GET  /                -> the HTML page itself
#    POST /set_folders     -> save the 4 folder paths the user picked
#    POST /index           -> start the background indexing job
#    GET  /progress        -> the UI polls this for the progress bar
#    GET  /search?q=...    -> run a search, return ranked video results
#    GET  /preview?path=.. -> stream a preview mp4 to the grid
#    POST /copy            -> copy selected source videos to destination
# =============================================================================

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE  # defined near the bottom of this file


@app.post("/set_folders")
async def set_folders(request: Request):
    data = await request.json()
    # sources is a list; the other three are single path strings.
    if "sources" in data and isinstance(data["sources"], list):
        STATE["folders"]["sources"] = [s.strip() for s in data["sources"] if s and s.strip()]
    for key in ("frames", "previews", "destination"):
        if key in data:
            STATE["folders"][key] = data[key].strip()
    save_settings()      # remember these paths for next launch
    # Try loading an existing index now that we know where frames live.
    load_index()
    return {"ok": True, "folders": STATE["folders"]}


@app.get("/get_folders")
def get_folders():
    """The page calls this on load to pre-fill the four boxes with whatever
    folder paths were remembered from last time."""
    return {"folders": STATE["folders"]}


@app.get("/browse_folder")
def browse_folder():
    """Open a native Windows folder-picker (via tkinter, which ships with
    Python) and return the chosen path. The web page can't open a real folder
    dialog itself due to browser security, so the backend does it.

    Note: the dialog can sometimes open BEHIND the browser window — alt-tab to
    it if you don't see it pop up."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()                    # hide the empty root window
        root.attributes("-topmost", True)  # try to bring the dialog to front
        folder = filedialog.askdirectory() # the actual picker
        root.destroy()
        return {"ok": True, "path": folder or ""}
    except Exception as e:
        # If a machine has no GUI/tkinter, fail gracefully; user can still type.
        return {"ok": False, "path": "", "error": str(e)}


@app.post("/index")
def start_index():
    if STATE["progress"]["running"]:
        return {"ok": False, "message": "Already running."}
    # Run the heavy job in a background thread so the page stays responsive.
    threading.Thread(target=build_index_job, daemon=True).start()
    return {"ok": True}


@app.post("/cancel_index")
def cancel_index():
    """Ask a running index job to stop. It finishes the current video, keeps all
    completed work, then stops. A later index run resumes the remainder."""
    if STATE["progress"]["running"]:
        STATE["progress"]["cancel"] = True
        STATE["progress"]["message"] = "Cancelling after current video\u2026"
        return {"ok": True}
    return {"ok": False, "message": "Nothing is running."}


@app.get("/progress")
def get_progress():
    return STATE["progress"]


@app.get("/search")
def do_search(q: str, source: str = ""):
    # source="" searches the whole library; a folder path limits to one source.
    return JSONResponse(search(q, source=source))


@app.get("/browse_all")
def do_browse_all(source: str = ""):
    """Indexed videos, no search - the default browse view. source="" browses
    the whole library; a folder path limits to a single source folder."""
    return JSONResponse(browse_all(source=source))


@app.get("/sources")
def do_sources():
    """The list of source folders available to filter by (for the single-folder
    tab's dropdown), each with a short display name."""
    return {"sources": [{"path": s, "name": Path(s).name or s}
                        for s in indexed_sources()]}


@app.post("/open_file")
async def open_file(request: Request):
    """Open the ORIGINAL source video in the computer's default media player.
    Works because the server runs locally on the user's own machine."""
    data = await request.json()
    path = data.get("path", "")
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": "File not found."}
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))                       # Windows default handler
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])         # macOS
        else:
            subprocess.Popen(["xdg-open", str(p)])     # Linux
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/reveal_file")
async def reveal_file(request: Request):
    """Open the folder CONTAINING the original video and select/highlight the
    file, so you can drag it straight into Premiere (or anywhere else). On
    Windows this is Explorer's '/select' which opens the folder with the file
    already highlighted; mac/Linux fall back to opening the folder."""
    data = await request.json()
    path = data.get("path", "")
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": "File not found."}
    try:
        if sys.platform.startswith("win"):
            # explorer /select,"C:\path\to\file.mp4" -> folder opens, file selected.
            # We pass args as a list; explorer is picky, so the path is one arg.
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])   # -R reveals in Finder
        else:
            # Most Linux file managers don't have a universal 'select' flag,
            # so we just open the containing folder.
            subprocess.Popen(["xdg-open", str(p.parent)])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/preview")
def get_preview(path: str):
    # Serve a preview mp4 by absolute path. We only serve files that we know
    # are previews we created (basic safety check).
    p = Path(path)
    if p.exists() and p.suffix.lower() == ".mp4":
        return FileResponse(str(p), media_type="video/mp4")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/copy")
async def copy_selected(request: Request):
    data = await request.json()
    videos = data.get("videos", [])           # list of ORIGINAL source paths
    dest = Path(STATE["folders"]["destination"])
    dest.mkdir(parents=True, exist_ok=True)

    copied, skipped, copied_paths = [], [], []
    for v in videos:
        src = Path(v)
        if not src.exists():
            skipped.append(v)
            continue
        target = dest / src.name
        # If a file with that name already exists, add a number so we don't clobber.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            n = 1
            while (dest / f"{stem}_{n}{suffix}").exists():
                n += 1
            target = dest / f"{stem}_{n}{suffix}"
        shutil.copy2(src, target)             # copy2 preserves timestamps
        copied.append(str(target))
        # Remember we copied this SOURCE this session, so its tile greys out.
        STATE["copied_this_session"].add(v)
        copied_paths.append(v)
    return {"ok": True, "copied": len(copied), "skipped": len(skipped),
            "copied_videos": copied_paths}


# =============================================================================
#  THE FRONTEND (HTML + CSS + JavaScript), served as one string.
#  This is the page you see and click in Chrome. Comments inline explain the
#  important parts. It talks to the endpoints above using fetch().
# =============================================================================

HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TikTok Sorter</title>
<style>
  /* --- basic dark, compact styling --- */
  body { font-family: system-ui, sans-serif; margin: 0; background:#15171c; color:#e8e8ea; }
  header { padding:14px 18px; background:#1d2027; border-bottom:1px solid #2a2e37; }
  h1 { font-size:18px; margin:0; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:12px 18px; }
  label { font-size:12px; color:#9aa0aa; display:block; margin-bottom:3px; }
  input[type=text]{ background:#0f1115; border:1px solid #2a2e37; color:#e8e8ea;
    padding:7px 9px; border-radius:6px; font-size:13px; }
  .folder input{ width:250px; }
  .pick{ display:flex; gap:4px; }
  button.browse{ background:#2a2e37; padding:7px 10px; font-size:12px; }
  .statusmsg{ font-size:13px; color:#4ade80; align-self:center; }
  button { background:#3b82f6; color:white; border:none; padding:8px 14px;
    border-radius:6px; font-size:13px; cursor:pointer; }
  button.secondary{ background:#2a2e37; }
  /* toggle button in its "on" state gets a subtle accent border */
  button.secondary.active{ background:#26303f; border:1px solid #3b82f6; }
  button:disabled{ opacity:.5; cursor:default; }
  #searchbar{ width:420px; font-size:15px; padding:10px 12px; }

  /* --- collapsible folder-setup section --- */
  /* Header you click to fold the whole setup area away once you're set up. */
  .setup-head{ display:flex; align-items:center; gap:8px; cursor:pointer;
    padding:8px 18px; color:#cfd3da; font-size:13px; user-select:none; }
  .setup-head:hover{ color:#fff; }
  .setup-head .chev{ transition:transform .15s; display:inline-block; }
  .setup-head.collapsed .chev{ transform:rotate(-90deg); }  /* point right when collapsed */
  #setupBody.collapsed{ display:none; }                     /* hide the whole setup body */
  /* Scrollable box around the source rows so 100+ sources don't push the page
     down - it becomes a tidy ~6-row-tall list you scroll inside instead. */
  #sourceList{ max-height:210px; overflow-y:auto; padding-right:4px;
    border:1px solid #2a2e37; border-radius:6px; padding:6px; background:#0f1115; }
  #sourceList:empty{ border:none; padding:0; }

  #progress{ font-size:12px; color:#9aa0aa; padding:0 18px 8px; }
  .bar{ height:6px; background:#2a2e37; border-radius:4px; overflow:hidden; margin-top:4px;}
  .bar > div{ height:100%; background:#3b82f6; width:0%; transition:width .3s; }

  /* --- results grid --- */
  #grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
    gap:10px; padding:14px 18px; }
  .tile{ position:relative; border:3px solid transparent; border-radius:8px;
    overflow:hidden; cursor:pointer; background:#0f1115; transition:box-shadow .12s, border-color .12s; }
  /* selected: bold blue border + a glowing ring so it clearly stands out */
  .tile.selected{ border-color:#3b82f6;
    box-shadow:0 0 0 2px #3b82f6, 0 0 14px 2px rgba(59,130,246,0.55); }
  .tile video{ width:100%; display:block; }
  /* bottom info bar: filename row on top, tags + Open File button below */
  .tile .meta{ font-size:11px; color:#9aa0aa; padding:6px 7px; display:flex;
    flex-direction:column; gap:5px; background:#0f1115; }
  .namerow{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .inforow{ display:flex; align-items:center; gap:5px; }
  .check{ position:absolute; top:6px; right:6px; width:22px; height:22px;
    border-radius:50%; background:#3b82f6; color:white; display:none;
    align-items:center; justify-content:center; font-size:14px; }
  .tile.selected .check{ display:flex; }
  /* already-copied clips: dimmed and not selectable (preview still plays) */
  .tile.copied{ cursor:default; }
  .tile.copied video{ opacity:0.35; }
  .tile.copied .meta{ opacity:0.5; }
  .copiedtag{ position:absolute; top:6px; left:6px; background:#4ade80;
    color:#0f1115; font-size:10px; font-weight:700; padding:2px 6px;
    border-radius:4px; text-transform:uppercase; letter-spacing:.03em; }
  /* pill tags (likeness + duration) now live in the bottom info bar */
  .pill{ font-size:10px; font-weight:600; padding:2px 7px; border-radius:999px;
    color:#fff; background:#2a2e37; white-space:nowrap; }
  .pill.likeness{ background:#3b82f6; }
  .pill.dur{ background:#2a2e37; }
  /* Open File / Open Source buttons on their own row below the tags */
  .btngroup{ display:flex; gap:5px; margin-top:2px; }
  .openbtn{ flex:1; background:#2a2e37; color:#e8e8ea; border:none;
    padding:4px 9px; border-radius:5px; font-size:11px; cursor:pointer;
    white-space:nowrap; text-align:center; }
  .openbtn:hover{ background:#3b82f6; }

  .toolbar{ display:flex; gap:12px; align-items:center; padding:10px 18px;
    background:#1d2027; border-top:1px solid #2a2e37; position:sticky; bottom:0; }
  #count{ font-weight:600; }

  /* --- source-folder list rows --- */
  .srcrow{ display:flex; gap:4px; margin-bottom:6px; align-items:center; }
  .srcrow input{ width:340px; }
  .srcrow .removebtn{ background:#7f1d1d; padding:7px 10px; font-size:12px; }

  /* --- scope tab bar (All folders / single folder) --- */
  .tabs{ display:flex; gap:6px; align-items:center; padding:10px 18px 0; flex-wrap:wrap; }
  .tab{ background:#2a2e37; color:#cfd3da; border:1px solid #2a2e37; padding:7px 14px;
    border-radius:8px 8px 0 0; font-size:13px; cursor:pointer; }
  .tab.active{ background:#3b82f6; color:#fff; border-color:#3b82f6; }
  .tabs select{ background:#0f1115; border:1px solid #2a2e37; color:#e8e8ea;
    padding:7px 9px; border-radius:6px; font-size:13px; margin-left:6px; }
  .scopeinfo{ font-size:12px; color:#9aa0aa; margin-left:8px; }
</style>
</head>
<body>
<header><h1>TikTok Sorter <span style="font-weight:400;color:#9aa0aa;font-size:14px">- by NovaPMV</span></h1></header>

<!-- ===== Folder setup (collapsible) ===== -->
<!-- Click the header to fold this whole area away once you're set up, so the grid
     sits right at the top. The source list below also scrolls internally, so even
     with 100+ sources the page doesn't get pushed down. -->
<div id="setupHead" class="setup-head" onclick="toggleSetup()">
  <span class="chev">&#9660;</span>
  <span id="setupHeadLabel">Folder setup</span>
</div>
<div id="setupBody">

<!-- ===== Folder pickers ===== -->
<!-- Each folder has a "Browse" button (opens a native Windows folder picker via
     the backend) AND a text box you can still type/paste into as a fallback.
     Heads-up: the picker can sometimes open BEHIND this browser window - if you
     click Browse and nothing appears, alt-tab to find it. -->
<div class="row">
  <div class="folder" style="flex:1 1 100%">
    <label>Source folders (one per TikTok account — indexed together, searchable all-at-once or one at a time)</label>
    <div id="sourceList"></div>
    <button class="browse" style="margin-top:6px" onclick="addSourceRow('')">+ Add source folder</button>
  </div>
</div>
<div class="row">
  <div class="folder"><label>Frames folder (default: 1 frame every 2 sec)</label>
    <div class="pick"><input id="f_frames" type="text" placeholder="C:\\clipfinder\\frames">
    <button class="browse" onclick="browse('f_frames')">Browse</button></div></div>
  <div class="folder"><label>Previews folder</label>
    <div class="pick"><input id="f_previews" type="text" placeholder="C:\\clipfinder\\previews">
    <button class="browse" onclick="browse('f_previews')">Browse</button></div></div>
  <div class="folder"><label>Destination folder (copies go here)</label>
    <div class="pick"><input id="f_dest" type="text" placeholder="C:\\clipfinder\\selected">
    <button class="browse" onclick="browse('f_dest')">Browse</button></div></div>
  <div><label>&nbsp;</label><button onclick="saveFolders()">Save folders</button></div>
  <div><label>&nbsp;</label><button class="secondary" onclick="startIndex()">Build / update index</button></div>
  <div><label>&nbsp;</label><button id="cancelbtn" class="secondary" onclick="cancelIndex()" style="display:none;background:#7f1d1d">Cancel</button></div>
  <span id="folderstatus" class="statusmsg"></span>
</div>
</div><!-- /setupBody -->

<!-- ===== Progress bar for indexing ===== -->
<!-- ===== Progress area: status text (with counter + %) and the fill bar ===== -->
<div id="progress">
  <span id="progtext">Idle</span>
  <div class="bar"><div id="barfill"></div></div>
</div>

<!-- ===== Scope tabs: search ALL folders, or a single chosen folder ===== -->
<!-- These filter the SAME index - switching never re-indexes. "All folders"
     pools every account; the single-folder tab narrows to one account via the
     dropdown. Switching scope clears the current selection (like changing pages). -->
<div class="tabs">
  <div id="tabAll" class="tab active" onclick="setScopeAll()">All folders</div>
  <div id="tabOne" class="tab" onclick="setScopeOne()">Single folder</div>
  <select id="sourceSelect" style="display:none" onchange="onSourceSelectChange()"></select>
  <span id="scopeinfo" class="scopeinfo"></span>
</div>

<!-- ===== Combined controls row ===== -->
<!-- Three-part layout so the pagination stays TRULY centered (lining up with the
     bottom bar): search cluster on the left, pagination in the middle, and the
     Per page + Sort view options flush right. The left and right groups both
     flex:1 so they balance and keep the middle centered. -->
<div class="row" id="topnav" style="align-items:center">
  <div style="flex:1; display:flex; gap:10px; align-items:center; min-width:0;">
    <input id="searchbar" type="text" placeholder='Search e.g. "girl wearing jeans"'
           onkeydown="if(event.key==='Enter') runSearch()">
    <button onclick="runSearch()">Search</button>
    <!-- Toggle: whether copied clips show the grey "copied" overlay and are
         blocked from re-selection. Defaults ON (label shows how to turn it OFF). -->
    <button id="showCopiedBtn" class="secondary active" onclick="toggleShowCopied()">Disable Show Copied</button>
  </div>
  <div style="display:flex; gap:12px; align-items:center;">
    <button class="secondary" onclick="prevPage()">&larr; Prev</button>
    <span>page</span>
    <input id="pagenum" type="number" min="1" value="1" style="width:60px"
           onkeydown="if(event.key==='Enter') gotoPage()" onchange="gotoPage()">
    <span id="pageinfo">/ 0</span>
    <button class="secondary" onclick="nextPage()">Next &rarr;</button>
    <!-- Total videos currently in the grid (changes with scope + search). -->
    <span id="totalinfo" class="scopeinfo"></span>
  </div>
  <div style="flex:1; display:flex; justify-content:flex-end; gap:12px; align-items:center;">
    <div><label>Per page</label>
      <input id="perpage" type="range" min="4" max="50" value="24"
             oninput="perpageLabel.textContent=this.value; renderPage()">
      <span id="perpageLabel">24</span></div>
    <div><label>Sort by</label>
      <select id="sortmode" onchange="applySort(); page=0; renderPage();">
        <option value="relevance">Relevance (search)</option>
        <option value="name">Filename (A\u2013Z)</option>
        <option value="name_desc">Filename (Z\u2013A)</option>
        <option value="dur_desc">Duration (long\u2192short)</option>
        <option value="dur">Duration (short\u2192long)</option>
      </select></div>
  </div>
</div>

<!-- ===== Results grid ===== -->
<div id="grid"></div>

<!-- ===== Bottom pagination (mirror of the top controls) ===== -->
<!-- So you don't have to scroll back up to change pages after browsing a long
     grid. These share the same next/prev/goto logic as the top bar; renderPage
     keeps both in sync. -->
<div class="row" id="bottomnav" style="justify-content:center">
  <button class="secondary" onclick="prevPage()">&larr; Prev</button>
  <span>page</span>
  <input id="pagenum_b" type="number" min="1" value="1" style="width:60px"
         onkeydown="if(event.key==='Enter') gotoPageBottom()" onchange="gotoPageBottom()">
  <span id="pageinfo_b">/ 0</span>
  <button class="secondary" onclick="nextPage()">Next &rarr;</button>
  <span id="totalinfo_b" class="scopeinfo"></span>
</div>

<!-- ===== Sticky bottom toolbar: selection count + actions ===== -->
<div class="toolbar">
  <span id="count">0 selected</span>
  <button class="secondary" onclick="selectAllOnPage()">Select all on page</button>
  <button class="secondary" onclick="clearSelection()">Clear selection</button>
  <button onclick="copySelected()">Copy selected to destination</button>
  <span id="copystatus" class="statusmsg"></span>
</div>

<script>
// ---- app state held in the browser ----
let results = [];           // full ranked list from the last search
let page = 0;               // current page index
let selected = new Set();   // set of selected video paths (originals)

// Current view scope: "" (or "all") = every folder; otherwise a source path.
let currentScope = "";
// The last search query, so switching scope re-runs it in the new scope
// instead of dropping you back to browse-all.
let lastQuery = "";

// ---- build the source-folder input rows from a list of paths ----
function renderSourceRows(paths){
  const list = document.getElementById('sourceList');
  list.innerHTML = '';
  if(!paths || !paths.length){ addSourceRow(''); return; }
  paths.forEach(p => addSourceRow(p));
}

// ---- add one source-folder row (with Browse + Remove) ----
function addSourceRow(value){
  const list = document.getElementById('sourceList');
  const row = document.createElement('div');
  row.className = 'srcrow';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'C:\\\\tiktoks\\\\creatorX';
  input.value = value || '';
  const browseBtn = document.createElement('button');
  browseBtn.className = 'browse';
  browseBtn.textContent = 'Browse';
  browseBtn.onclick = async () => {
    const r = await fetch('/browse_folder');
    const d = await r.json();
    if(d.ok && d.path){ input.value = d.path; }
    else if(!d.ok){ showStatus('folderstatus', 'Folder picker unavailable - type the path instead.', true); }
  };
  const removeBtn = document.createElement('button');
  removeBtn.className = 'browse removebtn';
  removeBtn.textContent = 'Remove';
  removeBtn.onclick = () => {
    row.remove();
    // Always keep at least one row so there's somewhere to type.
    if(!document.querySelectorAll('#sourceList .srcrow').length) addSourceRow('');
  };
  row.appendChild(input);
  row.appendChild(browseBtn);
  row.appendChild(removeBtn);
  list.appendChild(row);
  // When the user adds a blank row by hand, scroll the (possibly capped) list to
  // it and focus it, so they can type right away without hunting for it.
  if(!value){ list.scrollTop = list.scrollHeight; input.focus(); }
}

// ---- collapse/expand the folder setup area ----
// Folds the whole setup block away so the grid sits at the top once you're set
// up. When collapsed, the header shows a small summary (how many sources) so you
// still know your setup at a glance.
function toggleSetup(){
  const head = document.getElementById('setupHead');
  const body = document.getElementById('setupBody');
  const collapsed = body.classList.toggle('collapsed');
  head.classList.toggle('collapsed', collapsed);
  const label = document.getElementById('setupHeadLabel');
  if(collapsed){
    const n = getSourcePaths().length;
    label.textContent = 'Folder setup (' + n + (n === 1 ? ' source' : ' sources')
                        + ' — click to edit)';
  } else {
    label.textContent = 'Folder setup';
  }
}

// ---- read the current list of source paths from the rows ----
function getSourcePaths(){
  return Array.from(document.querySelectorAll('#sourceList .srcrow input'))
    .map(i => i.value.trim())
    .filter(v => v.length);
}

// ---- on page load: pre-fill the folder boxes with remembered paths ----
window.addEventListener('load', async () => {
  try {
    const r = await fetch('/get_folders');
    const d = await r.json();
    const f = d.folders || {};
    renderSourceRows(f.sources || []);
    if(f.frames) f_frames.value = f.frames;
    if(f.previews) f_previews.value = f.previews;
    if(f.destination) f_dest.value = f.destination;
  } catch(e) { renderSourceRows([]); }
  await refreshSourceDropdown();   // populate the single-folder dropdown
  // If an index already exists from a past session, show all videos right away
  // so you can browse/copy without searching.
  loadAllVideos();
});

// ---- refresh the single-folder dropdown from what's indexed ----
async function refreshSourceDropdown(){
  try {
    const r = await fetch('/sources');
    const d = await r.json();
    const sel = document.getElementById('sourceSelect');
    const prev = sel.value;
    sel.innerHTML = '';
    (d.sources || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.path; opt.textContent = s.name;
      sel.appendChild(opt);
    });
    // keep the previous choice selected if it still exists
    if(prev && (d.sources||[]).some(s => s.path === prev)) sel.value = prev;
  } catch(e) { /* no sources yet */ }
}

// ---- load indexed videos (no search) into the grid, within current scope ----
async function loadAllVideos(){
  try {
    const r = await fetch('/browse_all?source=' + encodeURIComponent(currentScope));
    const all = await r.json();
    results = all || [];
    results.forEach((it,i) => it._ord = i);   // remember arrival order
    applySort();             // respect the chosen sort order
    page = 0;
    copiedSet = new Set();   // fresh session view
    renderPage();
  } catch(e) { /* no index yet - grid stays empty */ }
}

// ---- scope tab handlers ----
// "All folders": pool every indexed source.
function setScopeAll(){
  currentScope = "";
  document.getElementById('tabAll').classList.add('active');
  document.getElementById('tabOne').classList.remove('active');
  document.getElementById('sourceSelect').style.display = 'none';
  document.getElementById('scopeinfo').textContent = '';
  selected.clear();
  reapplyScope();
}
// "Single folder": narrow to the folder chosen in the dropdown.
async function setScopeOne(){
  await refreshSourceDropdown();
  const sel = document.getElementById('sourceSelect');
  if(!sel.options.length){
    showStatus('folderstatus', 'No indexed folders yet - add sources and build the index first.', true);
    return;
  }
  document.getElementById('tabOne').classList.add('active');
  document.getElementById('tabAll').classList.remove('active');
  sel.style.display = '';
  currentScope = sel.value;
  document.getElementById('scopeinfo').textContent = 'Showing only this folder';
  selected.clear();
  reapplyScope();
}
// Dropdown changed while on the single-folder tab.
function onSourceSelectChange(){
  currentScope = document.getElementById('sourceSelect').value;
  selected.clear();
  reapplyScope();
}
// Re-run whatever the user was doing (a search, or browse-all) in the new scope.
function reapplyScope(){
  if(lastQuery){ runSearch(true); }
  else { loadAllVideos(); }
}

// ---- Open the original file in the computer's default media player ----
async function openFile(videoPath){
  const r = await fetch('/open_file', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: videoPath})});
  const d = await r.json();
  if(!d.ok){ showStatus('copystatus', 'Could not open file: ' + (d.error||''), true); }
}

// ---- Open Source: reveal the original in its folder (for dragging to Premiere) ----
async function revealFile(videoPath){
  const r = await fetch('/reveal_file', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: videoPath})});
  const d = await r.json();
  if(!d.ok){ showStatus('copystatus', 'Could not open folder: ' + (d.error||''), true); }
}

// ---- Browse button: ask the backend to open a native folder picker ----
async function browse(fieldId){
  // (The dialog can open BEHIND the browser - alt-tab if you don't see it.)
  const r = await fetch('/browse_folder');
  const d = await r.json();
  if(d.ok && d.path){
    document.getElementById(fieldId).value = d.path;
  } else if(!d.ok){
    // No GUI available on this machine - user can still type the path.
    showStatus('folderstatus', 'Folder picker unavailable - type the path instead.', true);
  }
}

// ---- a small helper to show an inline status message (auto-fades) ----
function showStatus(elId, text, isError){
  const el = document.getElementById(elId);
  el.textContent = text;
  el.style.color = isError ? '#f87171' : '#4ade80';
  // clear it after a few seconds so it doesn't linger
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.textContent = ''; }, 4000);
}

// ---- folder saving ----
async function saveFolders(){
  const body = {
    sources: getSourcePaths(), frames: f_frames.value,
    previews: f_previews.value, destination: f_dest.value
  };
  const r = await fetch('/set_folders', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json();
  await refreshSourceDropdown();   // new folders become pickable in the dropdown
  showStatus('folderstatus', '\u2713 Folders saved.', false);  // inline, no popup
}

// ---- indexing + progress polling ----
async function startIndex(){
  await saveFolders();                       // make sure paths are current
  showStatus('folderstatus', 'Processing...', false);   // inline green message
  document.getElementById('progtext').textContent = 'Starting indexing\u2026';
  document.getElementById('cancelbtn').style.display = '';  // show Cancel
  await fetch('/index', {method:'POST'});
  pollProgress();                            // begin watching progress
}

// ---- cancel a running index (keeps work already finished) ----
async function cancelIndex(){
  await fetch('/cancel_index', {method:'POST'});
  // the job stops after the current video; pollProgress will show the result
}

async function pollProgress(){
  const r = await fetch('/progress'); const p = await r.json();
  // Percentage of this run's videos done.
  const pct = p.total ? Math.round(100*p.done/p.total) : (p.running?0:0);
  // Status line: the message, plus a live "done/total (pct%)" counter while running.
  let line = p.message;
  if(p.running && p.total){
    line = p.message + '  \u2014  ' + p.done + ' / ' + p.total + ' (' + pct + '%)';
  }
  document.getElementById('progtext').textContent = line;
  barfill.style.width = (p.total ? pct : (p.running?4:0)) + '%';

  if (p.running){
    setTimeout(pollProgress, 500);           // keep polling while it runs
  } else {
    document.getElementById('cancelbtn').style.display = 'none';  // hide Cancel
    // leave the final message (Done / Cancelled) on screen
    // Newly-indexed folders should now be pickable in the single-folder tab.
    refreshSourceDropdown();
    // Show indexed videos now that indexing is done, so you can browse/copy
    // without needing to search first (within the current scope).
    loadAllVideos();
  }
}

// ---- sorting the current results in place ----
// "relevance" keeps the order the backend returned (search rank, or filename
// for browse-all). The other modes sort by name or duration. We read _ord to
// restore the original backend order for the relevance mode.
function applySort(){
  const mode = (document.getElementById('sortmode') || {}).value || 'relevance';
  const byName = (a,b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase());
  const byDur  = (a,b) => (a.duration||0) - (b.duration||0);
  if(mode === 'name')       results.sort(byName);
  else if(mode === 'name_desc') results.sort((a,b)=>byName(b,a));
  else if(mode === 'dur')   results.sort(byDur);
  else if(mode === 'dur_desc') results.sort((a,b)=>byDur(b,a));
  else results.sort((a,b)=>(a._ord||0)-(b._ord||0));   // relevance / arrival order
}

// ---- search ----
async function runSearch(keepQuery){
  // keepQuery=true is used when re-running after a scope switch (uses lastQuery).
  const q = keepQuery ? lastQuery : searchbar.value.trim();
  if(!q){ lastQuery = ""; loadAllVideos(); return; }  // empty search = browse-all
  lastQuery = q;
  const r = await fetch('/search?q=' + encodeURIComponent(q)
                        + '&source=' + encodeURIComponent(currentScope));
  results = await r.json();
  results.forEach((it,i) => it._ord = i);   // arrival order = relevance rank
  applySort();     // relevance by default; or user's chosen name/duration sort
  page = 0;
  renderPage();
}

// ---- rendering the current page of the grid ----
function perPage(){ return parseInt(document.getElementById('perpage').value); }
function pageCount(){ return Math.max(1, Math.ceil(results.length / perPage())); }

// Track which videos are already copied this session (grey + unselectable).
// Seeded from each search result's "copied" flag, and updated after a copy.
let copiedSet = new Set();

// Whether to SHOW the "copied" grey overlay + block re-selecting copied clips.
// Defaults ON (matches the original behavior). When the user toggles it OFF,
// copied clips look and behave like normal clips, so they can be copied again.
// We still keep tracking copies in copiedSet, so flipping it back ON restores
// the greying without needing to re-copy anything.
let showCopied = true;

function toggleShowCopied(){
  showCopied = !showCopied;
  const btn = document.getElementById('showCopiedBtn');
  // Button label reflects what clicking it will DO next.
  btn.textContent = showCopied ? 'Disable Show Copied' : 'Show Copied';
  btn.classList.toggle('active', showCopied);
  renderPage();
}

// Turn seconds into a short tag: "9s" for under a minute, "1:03" above.
function fmtDuration(sec){
  sec = Math.round(sec || 0);
  if(sec < 60) return sec + 's';
  const m = Math.floor(sec/60), s = sec%60;
  return m + ':' + String(s).padStart(2,'0');
}

function renderPage(){
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const start = page * perPage();
  const slice = results.slice(start, start + perPage());

  for(const item of slice){
    // A clip is "copied" only for display purposes when the toggle is ON.
    // With the toggle OFF, copied clips behave like normal clips (selectable,
    // no grey overlay) so they can be deliberately copied again.
    const isCopied = showCopied && (copiedSet.has(item.video) || item.copied);
    const tile = document.createElement('div');
    tile.className = 'tile'
      + (selected.has(item.video) ? ' selected':'')
      + (isCopied ? ' copied':'');            // greyed + unselectable when copied
    // Only allow selecting if NOT already copied this session.
    if(!isCopied){
      tile.onclick = () => toggleSelect(item.video, tile);
    }

    // looping muted autoplaying preview (behaves like a GIF). Plays even when
    // greyed out - greying only blocks selection, not playback.
    const vid = document.createElement('video');
    vid.src = '/preview?path=' + encodeURIComponent(item.preview);
    vid.muted = true; vid.loop = true; vid.autoplay = true;
    vid.playsInline = true;
    tile.appendChild(vid);

    // ----- bottom info bar: filename, then a row of tags + Open File button -----
    const meta = document.createElement('div');
    meta.className = 'meta';

    // filename line
    const nameLine = document.createElement('div');
    nameLine.className = 'namerow';
    nameLine.textContent = item.name;
    meta.appendChild(nameLine);

    // tags + actions row
    const infoRow = document.createElement('div');
    infoRow.className = 'inforow';
    // likeness pill only shows when there's a score (i.e. from a search)
    const likenessHtml = (item.score !== null && item.score !== undefined)
      ? '<span class="pill likeness">likeness: ' + item.score + '</span>' : '';
    infoRow.innerHTML =
      likenessHtml +
      '<span class="pill dur">' + fmtDuration(item.duration) + '</span>';

    // Open File button - opens the ORIGINAL in the default player. We stop the
    // click from bubbling up to the tile so opening doesn't also select it.
    const openBtn = document.createElement('button');
    openBtn.className = 'openbtn';
    openBtn.textContent = 'Open File';
    openBtn.onclick = (e) => { e.stopPropagation(); openFile(item.video); };

    // Open Source button - reveals the ORIGINAL in its folder (highlighted) so
    // you can drag it straight into Premiere. Also stop the click bubbling.
    const revealBtn = document.createElement('button');
    revealBtn.className = 'openbtn';
    revealBtn.textContent = 'Open Source';
    revealBtn.onclick = (e) => { e.stopPropagation(); revealFile(item.video); };

    // Group both buttons into their own row BELOW the tags. The tags row can
    // hold several pills (likeness + duration), so giving the buttons a full
    // line of their own keeps 'Open Source' from getting cut off.
    const btnGroup = document.createElement('div');
    btnGroup.className = 'btngroup';
    btnGroup.appendChild(openBtn);
    btnGroup.appendChild(revealBtn);

    meta.appendChild(infoRow);   // tags row (likeness + duration)
    meta.appendChild(btnGroup);  // buttons row (Open File + Open Source)
    tile.appendChild(meta);

    // selection checkmark (shown when selected)
    const check = document.createElement('div');
    check.className = 'check'; check.textContent = '\u2713';
    tile.appendChild(check);

    // small "copied" tag in the corner when already copied this session
    if(isCopied){
      const tag = document.createElement('div');
      tag.className = 'copiedtag'; tag.textContent = 'copied';
      tile.appendChild(tag);
    }

    grid.appendChild(tile);
  }
  // update the page-number boxes and the "/ N" total on BOTH nav bars
  const curPage = results.length ? (page+1) : 0;
  const totPages = results.length ? pageCount() : 0;
  const n = results.length;
  const totalLabel = (n === 1 ? '1 video' : n + ' videos');
  document.getElementById('pagenum').value = curPage;
  document.getElementById('pagenum_b').value = curPage;
  document.getElementById('pageinfo').textContent = '/ ' + totPages;
  document.getElementById('pageinfo_b').textContent = '/ ' + totPages;
  document.getElementById('totalinfo').textContent = totalLabel;
  document.getElementById('totalinfo_b').textContent = totalLabel;
  // Hide the bottom bar entirely when there's nothing (or only one page) to page.
  document.getElementById('bottomnav').style.display = (totPages > 1) ? '' : 'none';
  updateCount();
}

// ---- selection handling ----
function toggleSelect(video, tile){
  if(selected.has(video)){ selected.delete(video); tile.classList.remove('selected'); }
  else { selected.add(video); tile.classList.add('selected'); }
  updateCount();
}
function updateCount(){ document.getElementById('count').textContent = selected.size + ' selected'; }
function selectAllOnPage(){
  const start = page*perPage();
  // Skip clips already copied this session ONLY when the toggle is on; with it
  // off, copied clips are selectable again so they can be re-copied.
  results.slice(start, start+perPage())
    .filter(i => !(showCopied && (copiedSet.has(i.video) || i.copied)))
    .forEach(i => selected.add(i.video));
  renderPage();
}
function clearSelection(){ selected.clear(); renderPage(); }

// ---- pagination ----
// Changing pages clears the current selection (so selections don't silently
// carry across pages). The "Clear selection" button still works any time too.
// After changing pages we also scroll back to the top of the grid so each page
// starts at its first tile (and the last row never hides under the toolbar).
function scrollToGridTop(){
  const grid = document.getElementById('grid');
  // Scroll so the grid's top sits just below the sticky search controls.
  const y = grid.getBoundingClientRect().top + window.scrollY - 12;
  window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
}
function nextPage(){
  if(page < pageCount()-1){ page++; selected.clear(); renderPage(); scrollToGridTop(); }
}
function prevPage(){
  if(page > 0){ page--; selected.clear(); renderPage(); scrollToGridTop(); }
}
// Jump straight to a typed page number. Shared by both nav bars; pass the id
// of whichever page-number box triggered it.
function gotoPageFrom(boxId){
  let n = parseInt(document.getElementById(boxId).value);
  if(isNaN(n)) return;
  n = Math.max(1, Math.min(pageCount(), n));   // clamp into valid range
  page = n - 1;
  selected.clear();
  renderPage();
  scrollToGridTop();
}
function gotoPage(){ gotoPageFrom('pagenum'); }         // top bar
function gotoPageBottom(){ gotoPageFrom('pagenum_b'); } // bottom bar

// ---- copying selected originals ----
async function copySelected(){
  if(selected.size === 0){
    showStatus('copystatus', 'Nothing selected.', true);
    return;
  }
  const r = await fetch('/copy', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({videos: Array.from(selected)})});
  const d = await r.json();
  // Mark the just-copied videos as copied (grey + unselectable this session),
  // and remove them from the current selection.
  (d.copied_videos || []).forEach(v => { copiedSet.add(v); selected.delete(v); });
  renderPage();
  // Inline message next to the button - no popup.
  let msg = '\u2713 Copied ' + d.copied + ' file(s).';
  if(d.skipped) msg += ' Skipped ' + d.skipped + '.';
  showStatus('copystatus', msg, false);
}
</script>
</body>
</html>
"""


# =============================================================================
#  STARTUP
#  The launcher (.bat) opens the browser; here we load settings + start server.
# =============================================================================

if __name__ == "__main__":
    load_settings()  # pre-fill the four folder boxes with last-used paths
    # If we have remembered folders, also try to load any existing index now,
    # so search works immediately on launch without re-indexing.
    if STATE["folders"].get("frames"):
        load_index()
    print("[ClipFinder] Starting…  open http://localhost:%d if it doesn't open on its own." % APP_PORT)
    # uvicorn is the web server that actually runs our FastAPI 'app'.
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT, log_level="warning")

'''
Title:        End-to-End-Sign-Language-AI-Translation-System
Description:  End-to-end system for sign language translation using AI models across Edge and Cloud.
Licence:      Creative Commons Attribution (CC BY 4.0) License

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots included in this project, please cite
the following paper:

https://doi.org/10.3390/math13233759">Nada Shahin and Leila Ismail, "Towards Trustworthy Sign Language Translation System: 
A Privacy-Preserving Edge–Cloud–Blockchain Approach",
Mathematics 2025

'''


"""
   Edge-Cloud pipeline:
   - Capture frames, extract MediaPipe keypoints.
   - Maintain sliding window (SEQ_LEN).
   - Save each full window as one sample on disk until MAX_SAMPLES (rolling).
   - When MAX_SAMPLES is reached, upload all saved samples to cloud.
   - Download latest model from cloud and cache locally.
   - Run local preprocessing + inference on each full window.
   - Log:
   * t_preprocess_ms (preprocessing time)
   * t_infer_ms (inference time)
   * t_upload_ms (edge-cloud video upload time)
   * t_download_ms (cloud-edge model download time)
"""

import os, io, time, json, csv, uuid, shutil
from time import perf_counter
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np
import requests
import mediapipe as mp
import torch
import sentencepiece as spm
import hashlib
from transformer_model import Keypoint3DCNNExtractor, Sign2GlossTransformerEncoderOnly, Decoder
 
## --- Config ---- ##
CLOUD_UPLOAD_URL = "http://IP:PORT/upload_keypoints" # replace with cloud IP and port
CLOUD_MODEL_URL = "http://IP:PORT/get_model" # replace with cloud IP and port
REQUEST_TIMEOUT = 60.0
SEQ_LEN = 30 # sliding window length (frames)
LOG_TO_CSV = True
CSV_PATH = "runtime_log.csv"

# Local storage
SAMPLES_DIR = "saved_keypoints" # store window videos as .npy (shape [SEQ_LEN, 1662])
MAX_SAMPLES = 50 # keep up to 50 samples
MODEL_PATH = "model/medasl_transformer_model.pth" # local model filename
META_JSON = "pending_translation_meta.json" # maps npy basename and sentence
TRANSLATIONS_CSV_NAME = "translations.csv" # the CSV we send with the NPZ

# Decoder/text assets on edge
SPECIALS_PATH = "model/special_ids.json"
SPM_PATH = "model/medasl_bpe.model"

MIN_WINDOWS_BEFORE_DISPLAY = 3 # warm-up windows after first full buffer
CONF_THRESH = 0.15 # average per-timestep top-1 prob threshold

RUN_LOCAL_INFERENCE = True # local inference
SHOW_METRICS = False # keep False to hide BUF/PRE/INF/E2E

windows_seen = 0    # counter of full windows processed (len(seq_buffer) == SEQ_LEN)
display_text = ""   # last accepted text shown in HUD (debounced by confidence)

_sp = None # used for text detokenization on the edge (BPE to string)
SPECIALS_HASH = None  # module-level cache of the last-written hash

class RuntimeConfig:
    def __init__(self, embed_dim:int):
        self.embed_dim = embed_dim

## ---- Utils ---- ##
def ms(dt_s: float) -> float:
    """Convert seconds to milliseconds."""
    return dt_s * 1000.0
 
def ensure_dir(path: str):
    """Create directory if it doesn’t exist (no error if already exists)."""
    os.makedirs(path, exist_ok=True)
 
def _load_meta() -> dict:
    """Load metadata JSON (maps sample filename -> translation)."""
    try:
        with open(META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} # Return empty dict if not readable/missing
 
def _save_meta(meta: dict):
    """Write metadata dict back to META_JSON (ignore failures)."""
    try:
        with open(META_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass

## --- Text-related --- ##
def _load_special_ids(path: str) -> dict | None:
   """
   Load the special token ID mappings used for gloss/text decoding.
   Reads a JSON file.
   Returns the dictionary if successful, or None if the file is missing
   or cannot be parsed (fallback to default IDs).
   """
   try:
      with open(path, "r", encoding="utf-8") as f:
         return json.load(f)
   except Exception:
      return None

def _ctc_collapse(ids_1d: np.ndarray, blank_id: int) -> list[int]:
   """
    Collapse a raw sequence of CTC (Connectionist Temporal Classification)
    token predictions into a final discrete sequence by removing
    repeated tokens and blanks.
    This is used to convert frame-level gloss logits into gloss IDs.
   """
   out = []
   prev = None
   for t in ids_1d.tolist():
      if t != blank_id and t != prev:
         out.append(int(t))
      prev = t
   return out

def _sp_load(model_path: str):
    """
    Load the SentencePiece tokenizer model used for detokenizing
    translated text (BPE to human-readable sentence). If not already
    loaded in memory.
    Initializes a global SentencePieceProcessor for later use.    
    """
    global _sp
    if _sp is None:
        if not os.path.isfile(model_path):
            # try to fetch it from the cloud once
            try_download_spm(verbose=True)
        if not os.path.isfile(model_path):
            # still not there—bail gracefully (detok stays empty)
            print(f"[spm] still missing at {model_path}; detokenization disabled.")
            return  
        _sp = spm.SentencePieceProcessor()
        _sp.load(model_path)


def bpe_ids_to_text(id_seq: np.ndarray, pad_id: int, sos_id: int, eos_id: int) -> str:
   """Convert 1D int array to text using SentencePiece."""
   _sp_load(SPM_PATH)
   valid = [int(t) for t in id_seq.tolist() if t not in (pad_id, sos_id, eos_id)]
   pieces = [_sp.id_to_piece(i) for i in valid]
   # fuse whitespace pieces like training
   fused = []
   i = 0
   while i < len(pieces):
      if pieces[i] == '▁' and i + 1 < len(pieces):
         fused.append('▁' + pieces[i+1]); i += 2
      else:
         fused.append(pieces[i]); i += 1
   text = _sp.decode_pieces(fused)
   return text.strip()


## ---- Save and Send Keypoints ---- ##
def rolling_filelist(root: str, ext: str = '.npy') -> list:
    """
    Return sorted list of files by modification time (oldest->newest).
    """
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(ext)]
    files.sort(key=os.path.getmtime) # Sort by modified time ascending
    return files
 
def save_window_sample(window_np: np.ndarray, root: str, max_keep: int, translation: str | None = None) -> str:
    """
    Save one window of keypoints (.npy) + optional translation sidecar.

    Args:
        window_np (np.ndarray): Shape [SEQ_LEN, 1662].
        root (str): Directory to store samples.
        max_keep (int): Maximum number of samples to retain.
        translation (str|None): Optional caption to associate.

    Returns:
        str: Final path of saved .npy file.
    """
    ensure_dir(root) # Ensure dir exists
    base = f"{int(time.time()*1e6)}_{uuid.uuid4().hex[:6]}.npy" # Timestamp+UUID filename
    fpath = os.path.join(root, base)
    tmp = fpath + ".tmp" # Temp file for atomic save

    # atomic save: open explicitly so np.save doesn't add .npy
    with open(tmp, "wb") as f:
        np.save(f, window_np.astype(np.float32))

    os.replace(tmp, fpath)   # Atomically rename tmp -> final

    # record translation in META_JSON sidecar
    meta = _load_meta()
    meta[os.path.basename(fpath)] = (translation or "").strip() # Map file->translation
    _save_meta(meta)

    # enforce max_keep: prune oldest .npy files + sidecar entries
    files = rolling_filelist(root, ".npy")
    while len(files) > max_keep:
        try:
            os.remove(files[0]) # Delete oldest file
        except OSError:
            pass
        meta.pop(os.path.basename(files[0]), None) # Remove from metadata
        _save_meta(meta)
        files.pop(0) # Drop from list
    return fpath

def pack_npz_from_dir(root: str) -> tuple[bytes, list[str]]:
    """
    Pack valid .npy samples from directory into compressed .npz.

    Args:
        root (str): Directory containing saved .npy samples.

    Returns:
        tuple(bytes, list[str]):
            - Bytes of .npz archive (field `samples`: [N, SEQ_LEN, 1662]).
            - List of filenames included.
    """
    files = rolling_filelist(root, ".npy") # Oldest to newest .npy files
    valid_arrays = [] # Collected sample arrays
    valid_names = []  # Corresponding filenames
    for p in files:
        try:
            a = np.load(p, allow_pickle=False) # Load files
        except Exception:
            # Skip unreadable file
            continue
        if a.ndim == 2 and a.shape == (SEQ_LEN, 1662): # Must match expected shape
            if a.dtype != np.float32: # Enforce float32 dtype
                a = a.astype(np.float32)
            valid_arrays.append(a)
            valid_names.append(os.path.basename(p))
        else: # Skip bad shape
            pass

    if not valid_arrays: # No valid samples
        return b"", []

    batch = np.stack(valid_arrays, axis=0).astype(np.float32)  # Stack into [N, SEQ_LEN, 1662]
    bio = io.BytesIO() # In-memory buffer
    np.savez_compressed(bio, samples=batch) # Save compressed .npz
    return bio.getvalue(), valid_names # Return bytes + file list

## --- Translation Caption --- ##
def draw_caption(
    img,
    text,
    *,
    max_w_ratio=0.85,          # max width as % of frame
    margin=18,                 # distance from bottom/edges
    pad_x=14, pad_y=10,        # inner padding
    alpha=0.55,                # box opacity (0=transparent, 1=opaque)
    font=cv2.FONT_HERSHEY_DUPLEX, # font style
    font_scale=0.9,            # relative font size
    thickness=1               # text stroke thickness
):
    """
    Draw a semi-transparent caption box with word-wrapped text at bottom of image.

    Args:
        img (np.ndarray): Image to draw on (BGR).
        text (str): Caption text.
        max_w_ratio (float): Max width relative to image width.
        margin (int): Margin from image edges.
        pad_x, pad_y (int): Padding inside the caption box.
        alpha (float): Box opacity.
        font (int): OpenCV font type.
        font_scale (float): Font size scaling factor.
        thickness (int): Font thickness.
    """
    if not text: # nothing to draw
        return

    h, w = img.shape[:2] # get image height/width
    max_w = int(w * max_w_ratio) # maximum text width in pixels

    # --- Word-wrap: split text into lines that fit max_w ---
    words = text.split()
    lines = [] # store wrapped lines
    cur = "" # current line buffer
    for wd in words:
        trial = (cur + " " + wd).strip()  # try appending word
        (tw, th), _ = cv2.getTextSize(trial, font, font_scale, thickness) # size of trial line
        if tw <= max_w - 2*pad_x or not cur: # fits in box OR first word
            cur = trial # keep building line
        else:
            lines.append(cur)  # line full -> push
            cur = wd # start new line
    if cur: # push last line
        lines.append(cur)

    # --- Compute box dimensions based on text lines ---
    line_sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]  # width/height of each line
    text_w = max(sz[0] for sz in line_sizes) if line_sizes else 0                     # widest line
    text_h = sum(sz[1] for sz in line_sizes)                                         # total text height
    line_gap = int(0.45 * line_sizes[0][1]) if line_sizes else 0                     # spacing between lines
    total_h = text_h + (len(lines)-1)*line_gap + 2*pad_y                             # full box height
    total_w = text_w + 2*pad_x                                                       # full box width

    # --- Compute coordinates of caption box (bottom-left anchored) ---
    x1 = margin                                 # left
    x2 = min(w - margin, x1 + total_w)          # right (clamped to image)
    y2 = h - margin                             # bottom
    y1 = y2 - total_h                           # top


    # --- Draw semi-transparent rectangle ---
    overlay = img.copy()                        # copy for alpha blending
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)   # dark background box
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)        # blend box into image

    # --- Draw each line of text with shadow ---
    y_text = y1 + pad_y + line_sizes[0][1] if line_sizes else y1 + pad_y + 16
    for s in lines:
        # shadow (black, slightly offset)
        cv2.putText(img, s, (x1 + pad_x + 1, y_text + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # main text (light gray)
        cv2.putText(img, s, (x1 + pad_x, y_text),         font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y_text += line_sizes[0][1] + line_gap # move down for next line

## --- MediaPipe ---- ##
def landmarks_to_np(lm_list, expected_n, include_z=True):
    """
    Convert Mediapipe landmarks to NumPy array.

    Args:
        lm_list: Mediapipe landmark list object (or None).
        expected_n (int): Number of landmarks to fill.
        include_z (bool): Whether to include z-coordinate.

    Returns:
        np.ndarray: Shape [expected_n, 3] (or [expected_n, 2] if include_z=False).
    """
    dims = 3 if include_z else 2
    out = np.zeros((expected_n, dims), dtype=np.float32)  # Pre-fill with zeros
    if lm_list is None: return out # Return zeros if missing
    for i, lm in enumerate(lm_list.landmark[:expected_n]):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        if include_z:
            out[i, 2] = getattr(lm, "z", 0.0) # Default z=0.0 if missing
    return out

def landmarks_visibility(lm_list, expected_n):
    """
    Extract visibility scores from Mediapipe landmarks.

    Args:
        lm_list: Mediapipe landmark list object (or None).
        expected_n (int): Number of landmarks to fill.

    Returns:
        np.ndarray: Shape [expected_n,], visibility ∈ [0,1].
    """
    out = np.zeros((expected_n,), dtype=np.float32)  # Pre-fill with zeros
    if lm_list is None: return out
    for i, lm in enumerate(lm_list.landmark[:expected_n]):
        out[i] = getattr(lm, "visibility", 0.0)  # Default vis=0.0 if missing
    return out

def mediapipe_detection(image_bgr, model):
    """
    Run Mediapipe Holistic on a BGR frame and return new frame + results.

    Args:
        image_bgr (np.ndarray): Input image in BGR color.
        model: Mediapipe Holistic model instance.

    Returns:
        tuple: (processed_image_bgr, results)
            - processed_image_bgr: Converted back to BGR (for drawing/overlay).
            - results: Mediapipe landmarks (face, hands, pose).
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR->RGB
    img.flags.writeable = False # Speed-up by disabling writes
    results = model.process(img) # Run Mediapipe inference
    img.flags.writeable = True
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert back to BGR
    return out, results

## ---- Preprocessing --- ##
def preprocess(
          window_np: np.ndarray,
          *,
          seq_len: int,
          feature_dim: int = 1662,
          mean: Optional[np.ndarray] = None, # shape (feature_dim,) if used -- Recompute per-feature mean/std from training data
          std: Optional[np.ndarray] = None, # shape (feature_dim,) if used -- Recompute per-feature mean/std from training data
          clip_range: Optional[Tuple[float, float]] = None,  # Optional clipping
          enforce_contiguous: bool = True # Force contiguous memory layout
) -> Tuple[np.ndarray, float]:
   """
   Edge-side preprocessing of one full window of keypoint features.

   Args:
        window_np (np.ndarray): Input array [T, feature_dim].
        seq_len (int): Expected number of time steps (T).
        feature_dim (int): Expected number of features (default=1662).
        mean, std (np.ndarray | None): Optional per-feature normalization stats.
        clip_range (tuple | None): Optional value clipping (lo, hi).
        enforce_contiguous (bool): Ensure C-contiguous memory for speed.

   Returns:
        tuple:
            - x (np.ndarray): Preprocessed array [1, seq_len, feature_dim] (float32).
            - t_preprocess_ms (float): Preprocessing time in milliseconds.
   """
   t0 = perf_counter() # start timer
    
   # ---- shape checks ----
   if window_np.ndim != 2: # must be 2D
      raise ValueError(f"Expected 2D array [T,{feature_dim}], got shape {window_np.shape}")
   T, F = window_np.shape
   if F != feature_dim:  # wrong feature dimension
      raise ValueError(f"Bad feature_dim: got {F}, expected {feature_dim}")
   if T != seq_len: # wrong sequence length
      raise ValueError(f"Bad seq_len: got {T}, expected {seq_len}")
    
   # ---- dtype + sanitization ----
   x = window_np.astype(np.float32, copy=False) # force float32
   if not np.isfinite(x).all(): # replace NaN/±Inf with zeros
      x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
   # ---- optional normalization (only if mean/std are provided) ----
   if mean is not None and std is not None: # Normalize
      if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
         raise ValueError(f"mean/std must have shape ({feature_dim},)")

      safe_std = std.copy().astype(np.float32) # copy std to avoid mutating input
      safe_std[safe_std == 0] = 1.0  # avoid divide-by-zero
      x = (x - mean.astype(np.float32)) / safe_std # broadcast normalize [T,F]
       
   # ---- optional clipping after normalization ----
   if clip_range is not None: # clip values if requested
      lo, hi = clip_range
      x = np.clip(x, lo, hi, out=x)
    
   # ---- memory layout ----
   if enforce_contiguous: # ensure contiguous memory
      x = np.ascontiguousarray(x)
    
   # ---- add batch dim ----
   x = x[None, ...] # reshape [T,F] -> [1, T, F] == [1, seq_len, feature_dim]
    
   t1 = perf_counter()
   t_preprocess_ms = ms(t1 - t0)  # convert to ms
   return x, t_preprocess_ms 
 
## --- Transmission to cloud ---- ##
def _sha256_bytes(b: bytes) -> str:
    """
    Compute the SHA-256 hash of a byte sequence in memory.
    Used to generate a unique fingerprint of model files, metadata,
    or transmitted data blocks for integrity verification.

    Returns a 64-character hexadecimal string.
    """
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sha256_file(path: str) -> str | None:
    """
    Compute the SHA-256 hash of a file on disk, reading it in
    64-KB chunks to support large files efficiently.
    Used to detect whether a local file differs from
    the one on the cloud by comparing hash digests.
    
    Returns the hex digest string, or None if the file is missing.
    """
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def try_download_model() -> Optional[float]:
   """
   Download model from CLOUD_MODEL_URL and save to MODEL_PATH.
   Returns t_download_ms on success, else None.
   """
   t0 = perf_counter()
   try:
      r = requests.get(CLOUD_MODEL_URL, stream=True, timeout=REQUEST_TIMEOUT)
   except requests.exceptions.RequestException:
      return None
   if not r.ok:
      return None
   ensure_dir(os.path.dirname(MODEL_PATH) or ".")
   tmp_path = MODEL_PATH + ".tmp"
   with open(tmp_path, "wb") as f:
      for chunk in r.iter_content(chunk_size=65536):
         if chunk:
            f.write(chunk)
   shutil.move(tmp_path, MODEL_PATH)
   t1 = perf_counter()
   t_download_ms = ms(t1 - t0)

   # reload runner if needed
   model_runner.reload_if_updated()
   try_download_specials(verbose=False)
   # one-time warm-up
   _ = model_runner.infer(np.zeros((1, SEQ_LEN, 1662), dtype=np.float32))
   return t_download_ms

def try_download_specials(verbose: bool = True) -> bool:
    """
    Fetch specials JSON from the server and write it to SPECIALS_PATH
    only if content changed. Returns True if reachable (even if unchanged).
    """
    global SPECIALS_HASH
    url = CLOUD_MODEL_URL.replace("/get_model", "/get_specials")
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            if verbose:
                print(f"[specials] server responded {r.status_code}")
            return False

        new_bytes = r.content
        new_hash = _sha256_bytes(new_bytes)

        # initialize cache from disk on first run
        if SPECIALS_HASH is None:
            SPECIALS_HASH = _sha256_file(SPECIALS_PATH)

        # unchanged -> do nothing, stay quiet
        if new_hash == SPECIALS_HASH:
            return True

        ensure_dir(os.path.dirname(SPECIALS_PATH) or ".")
        tmp = SPECIALS_PATH + ".tmp"
        with open(tmp, "wb") as f:
            f.write(new_bytes)
        os.replace(tmp, SPECIALS_PATH)

        SPECIALS_HASH = new_hash
        if verbose:
            print(f"[specials] updated {SPECIALS_PATH}")
        return True

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[specials] download failed: {e}")
        return False
    
def try_download_spm(verbose: bool = True) -> bool:

    """
    Attempt to download the SentencePiece tokenizer model (BPE)
    from the cloud and save it locally to SPM_PATH.
    
    - The cloud endpoint is derived from CLOUD_MODEL_URL
    - If the download succeeds, the file is atomically written
    - If the download or network request fails, the function returns False.
    
    Args:
        verbose (bool): If True, prints download status messages.
    
    Returns:
        bool: True if the model was successfully downloaded and saved,
              False if the request failed or returned a non-OK response.
    """
    url = CLOUD_MODEL_URL.replace("/get_model", "/get_spm")
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            if verbose: print(f"[spm] server responded {r.status_code}")
            return False
        ensure_dir(os.path.dirname(SPM_PATH) or ".")
        tmp = SPM_PATH + ".tmp"
        with open(tmp, "wb") as f: f.write(r.content)
        os.replace(tmp, SPM_PATH)
        if verbose: print(f"[spm] updated {SPM_PATH}")
        return True
    except requests.exceptions.RequestException as e:
        if verbose: print(f"[spm] download failed: {e}")
        return False
   
def try_upload_saved_samples() -> Optional[float]:
    """
    Collect all locally saved keypoint samples (.npy) from SAMPLES_DIR,
    package them into a compressed .npz file, and upload the batch along
    with its translation metadata (CSV) to the cloud server.

    Workflow:
    1. Load all valid [SEQ_LEN, 1662] keypoint samples from disk.
    2. Generate a CSV that maps each sample filename to its translation text.
    3. Compute SHA-256 hashes of both the NPZ and CSV for integrity verification.
    4. POST both files to CLOUD_UPLOAD_URL using multipart form-data.
    5. If upload succeeds (HTTP 200), delete only the samples that were sent
       and update the local metadata JSON accordingly.

    Returns:
        Optional:
            The total upload duration in milliseconds (ms) if successful,
            or None if the upload failed or timed out.
    """
    data_bytes, file_basenames = pack_npz_from_dir(SAMPLES_DIR)
    if not data_bytes or not file_basenames:
        return None

    meta = _load_meta()
    csv_bio = io.StringIO()
    w = csv.writer(csv_bio)
    w.writerow(["sample_npy", "translation"])
    for bn in file_basenames:
        w.writerow([bn, meta.get(bn, "")])
    csv_bytes = csv_bio.getvalue().encode("utf-8")

    # Hash the files before transmitting to cloud

    data_hash = _sha256_bytes(data_bytes)
    csv_hash = _sha256_bytes(csv_bytes)

    files = {
        "file": ("samples.npz", data_bytes, "application/octet-stream"),
        "meta": (TRANSLATIONS_CSV_NAME, csv_bytes, "text/csv"),
    }

    headers = {
        "X-File-SHA256": data_hash,
        "X-Meta-SHA256": csv_hash,
    }

    t0 = perf_counter()
    try:
        r = requests.post(CLOUD_UPLOAD_URL, files=files, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        return None
    t1 = perf_counter()
    if not r.ok:
        return None

    # delete only the ones we actually sent
    for bn in file_basenames:
        p = os.path.join(SAMPLES_DIR, bn)
        try:
            os.remove(p)
        except OSError:
            pass
        meta.pop(bn, None)
    _save_meta(meta)

    return ms(t1 - t0)


## ---- Run Model ---- ##
class RuntimeEnd2End(torch.nn.Module):
   """
   x [1,T,1662] -> feat -> (1,T,512) -> s2g -> gloss_logits
   -> CTC greedy -> <SOS>+pred -> decoder -> text_ids
   """
   def __init__(self, in_dim:int, embed_dim:int, num_heads:int,
               hidden_units:int, num_encoders:int,
               gloss_vocab_size:int, text_vocab_size:int,
               max_gloss_len:int, max_text_len:int,
               special_ids:dict|None):
      super().__init__()
      self.specials = special_ids or {}
      self.max_text_len = max_text_len

      self.feat = Keypoint3DCNNExtractor(in_dim=in_dim, out_dim=512)

      cfg = RuntimeConfig(embed_dim=embed_dim)
      self.s2g = Sign2GlossTransformerEncoderOnly(
         config=cfg,
         gloss_vocab_size=gloss_vocab_size,
         num_encoders=num_encoders,
         hidden_units=hidden_units,
         num_heads=num_heads,
         inv_gloss_vocab_dict=None
      )

      # embeddings for gloss memory (match training names/shapes)
      self.gloss_token_emb = torch.nn.Embedding(
         gloss_vocab_size, embed_dim,
         padding_idx=self.specials.get("GLOSS_PAD_ID", 0)
      )
      self.gloss_pos_emb = torch.nn.Embedding(max_gloss_len, embed_dim)

      # minimal config shim for your Decoder
      class _Cfg: pass
      _c = _Cfg(); _c.embed_dim = embed_dim; _c.num_heads = num_heads
      _c.hidden_units = hidden_units; _c.decoder_layers = 1
      _c.text_vocab_size = text_vocab_size
      _c.max_text_length = max_text_len
      self.dec = Decoder(_c)

   def _embed_gloss_memory(self, gloss_ids: torch.Tensor) -> torch.Tensor:
      B, Sg = gloss_ids.size()
      pos = torch.arange(Sg, device=gloss_ids.device).unsqueeze(0).expand(B, Sg)
      return self.gloss_token_emb(gloss_ids) + self.gloss_pos_emb(pos)

   @torch.no_grad()
   def forward(self, x_btF: torch.Tensor):
      """
      x_btF: torch.FloatTensor [1,T,1662]
      Returns dict with numpy-ready tensors.
      """
      gloss_pad = int(self.specials.get("GLOSS_PAD_ID", 0))
      gloss_sos = int(self.specials.get("GLOSS_SOS_ID", 1))
      gloss_eos = int(self.specials.get("GLOSS_EOS_ID", 2))
      gloss_blank = int(self.specials.get("GLOSS_BLANK_ID", 3))

      text_pad = int(self.specials.get("TEXT_PAD_ID", 0))
      text_sos = int(self.specials.get("TEXT_SOS_ID", 1))
      text_eos = int(self.specials.get("TEXT_EOS_ID", 2))

      feats = self.feat(x_btF) # (1,T,512)
      gloss_logits = self.s2g(feats) # (1,T,Vg)

      # --- CTC greedy on CPU numpy for collapse
      logp = torch.log_softmax(gloss_logits, dim=-1)[0] # (T,Vg)
      pred_ids = logp.argmax(dim=-1).cpu().numpy() # (T,)
      gloss_seq = _ctc_collapse(pred_ids, blank_id=gloss_blank) # list[int]

      # Build gloss memory: <SOS> + pred (crop/pad to max_gloss_len)
      Sg = self.gloss_pos_emb.num_embeddings
      seq = [gloss_sos] + gloss_seq
      seq = seq[:Sg] + [gloss_pad] * max(0, Sg - len(seq))
      gloss_inp = torch.tensor([seq], dtype=torch.long, device=x_btF.device) # (1,Sg)
      gloss_mem = self._embed_gloss_memory(gloss_inp)
      gloss_pad_mask = (gloss_inp == gloss_pad)

      # Decode text ids with your Decoder (greedy)
      text_ids = self.dec.predict(
         gloss_memory=gloss_mem,
         sos_token_id=text_sos,
         pad_token_id=text_pad,
         eos_token_id=text_eos,
         max_len=self.max_text_len,
         gloss_pad_mask=gloss_pad_mask
      ) # (1, St)

      # Optionally also produce logits for first token (not strictly needed)
      text_logits = None

      return {
         "gloss_logits": gloss_logits, # torch tensor
         "text_logits": text_logits,
         "text_ids": text_ids
      }

def build_model():
   """
   Rebuilds feat+s2g+dec and loads checkpoint.
   Uses SPECIALS_PATH for IDs; asserts vocab sizes against checkpoint shapes.
   """
   ckpt_path = MODEL_PATH
   if not os.path.isfile(ckpt_path):
      raise FileNotFoundError(f"MODEL_PATH not found: {ckpt_path}")

   sd = torch.load(ckpt_path, map_location="cpu")

   # extract flat state_dict
   if isinstance(sd, dict) and "model_state" in sd and isinstance(sd["model_state"], dict):
      state = sd["model_state"]
   elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
      state = sd["state_dict"]
   elif isinstance(sd, dict):
      state = sd
   else:
      return sd # full module

   # infer sizes from shapes
   if "s2g.gloss_classifier.weight" not in state:
      raise RuntimeError("Missing s2g.gloss_classifier.weight in checkpoint.")
   gloss_vocab_size = int(state["s2g.gloss_classifier.weight"].shape[0])
   embed_dim = int(state["s2g.gloss_classifier.weight"].shape[1])

   if "dec.output_proj.weight" not in state:
      raise RuntimeError("Missing dec.output_proj.weight in checkpoint.")
   text_vocab_size = int(state["dec.output_proj.weight"].shape[0])

   max_gloss_len = int(state.get("gloss_pos_emb.weight", torch.empty(0)).shape[0]) if "gloss_pos_emb.weight" in state else 64
   max_text_len = int(state.get("dec.positional_encoding.weight", torch.empty(0)).shape[0]) if "dec.positional_encoding.weight" in state else 64

   # encoder layers
   # --- robustly infer num_encoders from state_dict keys
   enc_indices = []
   for k in state.keys():
      if not k.startswith("s2g.encoder_layers."):
         continue
      parts = k.split(".") # e.g. ["s2g","encoder_layers","0","conv_half","weight"]
      if len(parts) >= 3:
         token = parts[2]
         if token.isdigit():
            enc_indices.append(int(token))
         else:
            # keys look like "s2g.encoder_layers.conv_half.weight" (single module, no index)
            enc_indices.append(0)

   num_encoders = (max(enc_indices) + 1) if enc_indices else 1
   if num_encoders <= 0:
      num_encoders = 1

   num_heads = 8
   hidden_units = 512
   in_dim = 1662

   specials = _load_special_ids(SPECIALS_PATH) or {}
   if not os.path.isfile(SPECIALS_PATH):
      print(f"[warn] specials not found at {SPECIALS_PATH}. "
          "Using default IDs; decoding may be suboptimal.")

   # sanity: warn if TEXT_VOCAB_SIZE mismatches checkpoint
   if specials.get("TEXT_VOCAB_SIZE") and specials["TEXT_VOCAB_SIZE"] != text_vocab_size:
      print(f"[warn] TEXT_VOCAB_SIZE in JSON ({specials['TEXT_VOCAB_SIZE']}) != checkpoint ({text_vocab_size}). Using checkpoint size.")

   model = RuntimeEnd2End(
         in_dim=in_dim,
         embed_dim=embed_dim,
         num_heads=num_heads,
         hidden_units=hidden_units,
         num_encoders=num_encoders,
         gloss_vocab_size=gloss_vocab_size,
         text_vocab_size=text_vocab_size,
         max_gloss_len=max_gloss_len,
         max_text_len=max_text_len,
         special_ids=specials
   )

   # remove shape-mismatched tensors from checkpoint ---
   model_sd = model.state_dict()
   to_drop = []
   for k, v in state.items():
      if k in model_sd and model_sd[k].shape != v.shape:
         to_drop.append(k)

   if to_drop:
      for k in to_drop:
         state.pop(k)

   # load what fits; the dropped params will keep their random init
   missing, unexpected = model.load_state_dict(state, strict=False)

   model.eval()
   return model

def ctc_confidence(gloss_logits_np: np.ndarray) -> float:
    """
    gloss_logits_np: (1, T, Vg) float32 logits
    Returns mean(max softmax prob per timestep) in [0,1].
    """
    if gloss_logits_np is None:
        return 0.0
    logp = torch.log_softmax(torch.from_numpy(gloss_logits_np[0]), dim=-1)  # (T, Vg)
    p_top = torch.exp(logp).max(dim=-1).values                               # (T,)
    return float(p_top.mean().item())

class ModelRunner:
   """
   Model Prediction/Inference
   Assumes the cloud serves a model with a single input matching shape [1, SEQ_LEN, 1662] (float32)
   Returns a dict with keys: "gloss_logits", "text_ids" (optional), etc.
   """
   def __init__(self, model_path: str, model_ctor=None, map_location="cpu"):
      self.model_path = model_path                 # Filesystem path to checkpoint (.pth)
      self.model_ctor = model_ctor                 # Callable returning an nn.Module ready to run
      self.map_location = map_location             # Torch map_location for loading
      self.model = None                            # Will hold the nn.Module (eval mode)
      self.seq_len = None                          # (Reserved for future, not used here)
      self.feature_dim = 1662                      # Fixed feature dimension
      self._init_model_if_exists()                 # Try to load at construction

   def _init_model_if_exists(self):
      """
      Load model if the checkpoint file exists.

      If model_ctor is provided, it should build the architecture and
      load/filter the state dict internally.
      """
      if not os.path.isfile(self.model_path):  # Bail out if checkpoint missing
         self.model = None
         return

      if self.model_ctor is not None:
         # model_ctor() (i.e., build_model) already loads & filters the state dict
         self.model = self.model_ctor()   # Construct + load model
         self.model.eval() # Inference mode
         return

      # Fallback path only if no ctor was provided: try to load a full module
      sd = torch.load(self.model_path, map_location=self.map_location)
      if isinstance(sd, torch.nn.Module): # Directly serialized module
         self.model = sd
         self.model.eval()
      elif isinstance(sd, dict):
      # If this branch is reached without a ctor, must construct an architecture and load the dict
         raise RuntimeError("State dict found but model_ctor is None; cannot construct model.")
      else:
         raise RuntimeError("Unsupported checkpoint format.")

   def reload_if_updated(self):
      """Reinitialize the model after a new checkpoint is downloaded."""
      self._init_model_if_exists()

   @torch.no_grad()
   def infer(self, x: np.ndarray):
      """
      Run a forward pass on one batch and postprocess outputs.

      Args:
          x (np.ndarray): Input array [1, T, 1662] (float32).

      Returns:
          tuple(dict|None, float|None):
              - outputs dict (or None if model missing):
                  {
                    "gloss_logits": np.ndarray (1, T, Vg) or None,
                    "text_ids":     np.ndarray (1, <=St) int64 or None,
                    "text_str":     str or None
                  }
              - t_infer_ms (float): Inference latency in milliseconds (or None)
      """
      if self.model is None: # No model loaded yet
         return None, None
      if not isinstance(x, np.ndarray) or x.ndim != 3 or x.shape[0] != 1:
         # Input contract check
         raise ValueError(f"Expected np.ndarray [1, T, F], got {type(x)} shape {getattr(x, 'shape', None)}")

      t0 = perf_counter() # Start timing
      xt = torch.from_numpy(x) # Convert np to torch [1, T, F]
      out = self.model(xt) # Forward pass: expected dict of tensors
      t1 = perf_counter() # End timing
      t_infer_ms = ms(t1 - t0) # Convert to milliseconds

      # numpy conversions
      gloss_logits_np = out["gloss_logits"].detach().cpu().numpy() if out["gloss_logits"] is not None else None
      text_ids_np = out["text_ids"].detach().cpu().numpy() if out["text_ids"] is not None else None

      # optional detokenize to string
      text_str = None
      try:
         specials = _load_special_ids(SPECIALS_PATH) or {} # {TEXT_PAD_ID, TEXT_SOS_ID, TEXT_EOS_ID}
         if text_ids_np is not None:
            text_str = bpe_ids_to_text(
               id_seq=text_ids_np[0], # (St,)
               pad_id=int(specials.get("TEXT_PAD_ID", 0)),
               sos_id=int(specials.get("TEXT_SOS_ID", 1)),
               eos_id=int(specials.get("TEXT_EOS_ID", 2))
            )
      except Exception:
         text_str = None # Robust to missing/malformed specials

      res = {
         "gloss_logits": gloss_logits_np,  # (1, T, Vg) or None
         "text_ids": text_ids_np,  # (1, St) or None
         "text_str": text_str # Detokenized text or None
      }
      return res, t_infer_ms

## --- Mediapipe --- ##
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

## --- Application --- ##
cap = cv2.VideoCapture(0) # Open default camera (index 0, change index as needed)
cv2.namedWindow("Sign Language Translation", cv2.WINDOW_NORMAL) # Resizable window
cv2.resizeWindow("Sign Language Translation", 1280, 720) # Initial window size

# Sliding window + CSV
seq_buffer = deque(maxlen=SEQ_LEN)  # Holds last SEQ_LEN frames of keypoint vectors
csv_file = open(CSV_PATH, "a", newline="") if LOG_TO_CSV else None # Append or disable
csv_writer = csv.writer(csv_file) if LOG_TO_CSV else None # CSV writer if logging enabled
if LOG_TO_CSV and csv_file.tell() == 0: # If new file, write header
   csv_writer.writerow([
   "frame_id",
   "mp_ms","t_preprocess_ms","t_infer_ms","end2end_ms",
   "t_upload_ms","t_download_ms",
   "saved_local"
   ])

ensure_dir(SAMPLES_DIR) # Make sure local samples dir exists
try_download_specials(verbose=False)   # Fetch special_ids.json
try_download_spm(verbose=False) # Fetch SentencePiece model
model_runner = ModelRunner(MODEL_PATH, model_ctor=build_model, map_location="cpu") # Prepare model runner

try:
   while True: # Main frame loop
       ok, frame = cap.read() # Grab a frame from camera
       if not ok: # Exit on capture failure
           break

       t_capture = perf_counter() # Start timing this iteration
       image, results = mediapipe_detection(frame, holistic) #  Run MediaPipe Holistic

       # --- keypoint extraction ---
       face_np  = landmarks_to_np(results.face_landmarks,        468, include_z=True)
       lh_np    = landmarks_to_np(results.left_hand_landmarks,     21, include_z=True)
       rh_np    = landmarks_to_np(results.right_hand_landmarks,    21, include_z=True)
       pose_np  = landmarks_to_np(results.pose_landmarks,          33, include_z=True)
       pose_vis = landmarks_visibility(results.pose_landmarks,     33)

       kp_concat = np.concatenate([lh_np, rh_np, face_np, pose_np], axis=0) # stacked order
       seq_xyz   = kp_concat.reshape(-1) # Reshape
       sequence  = np.concatenate([seq_xyz, pose_vis]).astype(np.float32) # 1662 (add pose visibilities)

       # safety: guarantee finite values
       if not np.isfinite(sequence).all(): # Replace NaN/±Inf with zeros to keep model stable
           sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

       seq_buffer.append(sequence.astype(np.float32)) # Push this frame vector into sliding window
       mp_ms = ms(perf_counter() - t_capture) # Time from capture to keypoints in milliseconds (ms)

       # --- optional drawing ---
       if results.face_landmarks is not None: # Draw face mesh if detected
           mp_drawing.draw_landmarks(
               image=image,
               landmark_list=results.face_landmarks,
               connections=mp_holistic.FACEMESH_TESSELATION,
               landmark_drawing_spec=None,
               connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
           )
       if results.left_hand_landmarks is not None: # Draw left hand landmarks if detected
           mp_drawing.draw_landmarks(
               image=image, landmark_list=results.left_hand_landmarks,
               connections=mp_holistic.HAND_CONNECTIONS,
               landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
               connection_drawing_spec=mp_styles.get_default_hand_connections_style()
           )
       if results.right_hand_landmarks is not None: # Draw right hand landmarks if detected
           mp_drawing.draw_landmarks(
               image=image, landmark_list=results.right_hand_landmarks,
               connections=mp_holistic.HAND_CONNECTIONS,
               landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
               connection_drawing_spec=mp_styles.get_default_hand_connections_style()
           )
       if results.pose_landmarks is not None: # Draw pose landmarks if detected
           mp_drawing.draw_landmarks(
               image=image, landmark_list=results.pose_landmarks,
               connections=mp_holistic.POSE_CONNECTIONS,
               landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
               connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
           )

       # ---------- when window is full: save, preprocess, infer ----------
       t_preprocess_ms = None # Per-window preprocess timing (ms)
       t_infer_ms = None  # Per-window inference timing (ms)
       t_upload_ms = None # Upload timing (ms) when triggered
       t_download_ms = None  # Model download timing (ms) when triggered
       sentence = "" # Candidate/translated text for this iteration
       saved_local = None # Whether NPZ is saved locally
       overlay = [] # display text metrics lines

       if len(seq_buffer) == SEQ_LEN: # Only act once a window is full, get short unique ID for row/event, and batch for preprocess
           frame_id = str(uuid.uuid4())[:8]
           seq_batch = np.stack(seq_buffer, axis=0)  # [SEQ_LEN, 1662]

           # Local preprocessing + inference
           x, t_preprocess_ms = preprocess(seq_batch, seq_len=SEQ_LEN) # Feature prep + timing
           t0_sentence = perf_counter() # Measure inference latency externally
           outs, _ = model_runner.infer(x)   # Inference; ignore internal timing
           t_sentence_ms = ms(perf_counter() - t0_sentence) # Per-window inference timing (ms)
           candidate = "" # Raw decoded text candidate/translation
           conf = 0.0 # CTC confidence initialization
           if isinstance(outs, dict): # Expect a dictionary with outputs
               candidate = (outs.get("text_str") or "").strip() # Best text hypothesis
               conf = ctc_confidence(outs.get("gloss_logits")) # Confidence from logits

           accepted = False
           windows_seen += 1 # Increase windows processed
           if windows_seen >= MIN_WINDOWS_BEFORE_DISPLAY and candidate and conf >= CONF_THRESH: # Debounce: wait a few windows and until confidence is accepted
               display_text = candidate # Display translated text
               accepted = True

           # Never show raw candidate; only show last accepted
           sentence = display_text

           # Report per-sentence timing only if accepted
           t_infer_ms = t_sentence_ms if accepted else None

           # Save sample with translation sidecar as .npy(+txt)
           save_window_sample(seq_batch, SAMPLES_DIR, MAX_SAMPLES, translation=sentence)
           saved_local = True

           # Try upload if reached MAX_SAMPLES
           if len(rolling_filelist(SAMPLES_DIR, ".npy")) >= MAX_SAMPLES: # Batch trigger for upload
               t_upload_ms = try_upload_saved_samples() # Push NPZ/CSVs to cloud
               if t_upload_ms is not None: # If upload happened: pull latest model if newer, refresh specials and SentencePiece model
                   t_download_ms = try_download_model()
                   try_download_specials()
                   try_download_spm(verbose=False)

           # Metrics overlay text
           if SHOW_METRICS:  # Build HUD lines
               overlay.append(f"BUF:{len(seq_buffer)}/{SEQ_LEN}")
               if t_preprocess_ms is not None: overlay.append(f"PRE:{t_preprocess_ms:.1f}ms")
               if t_infer_ms is not None:      overlay.append(f"INF:{t_infer_ms:.1f}ms")
               if t_upload_ms is not None:     overlay.append(f"UP:{t_upload_ms:.1f}ms")
               if t_download_ms is not None:   overlay.append(f"GET-M:{t_download_ms:.1f}ms")
               if t_preprocess_ms is not None and t_infer_ms is not None:
                   overlay.append(f"E2E:{(t_preprocess_ms + t_infer_ms):.1f}ms")

           # CSV logging
           if LOG_TO_CSV: # Append a row per full window
               end2end_ms = (t_preprocess_ms or 0.0) + (t_infer_ms or 0.0)
               csv_writer.writerow([
                   frame_id,
                   mp_ms, t_preprocess_ms, t_infer_ms, end2end_ms,
                   t_upload_ms, t_download_ms,
                   int(bool(saved_local)) if saved_local is not None else None
               ])


       # --- Optional translation display with metrics (only if SHOW_METRICS) ---
       if SHOW_METRICS and overlay: # Draw translucent black panel + text
           box_w = 600; row_h = 22
           rows = (len(overlay) + 1) // 2
           cv2.rectangle(image, (10, 10), (10 + box_w, 10 + rows*row_h), (0, 0, 0), -1)
           x_text, y_text = 20, 28
           for i, txt in enumerate(overlay):
               cv2.putText(image, txt, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
               if i % 2 == 1:
                   y_text += row_h; x_text = 20
               else:
                   x_text = 320

       # ---- One caption draw per frame (translation only) ----
       caption_text = (sentence.strip() if sentence and sentence.strip() else "translating...") # Stable caption
       if len(seq_buffer) < SEQ_LEN: # Hide until we have a full window
           caption_text = "translating..."
       draw_caption(image, caption_text) # Render caption at bottom

       # ---- show + key handling ----
       cv2.imshow("Sign Language Translation", image) # Show current frame
       key = cv2.waitKey(1) & 0xFF # Poll a key with small delay
       if key == ord('q'): # Quit
           break
       elif key == ord('u'): # Manual upload + model fetch
           t_up = try_upload_saved_samples()
           t_md = try_download_model()

           if LOG_TO_CSV and csv_writer is not None:  # Log manual event
               csv_writer.writerow([
                   f"manual-{uuid.uuid4().hex[:6]}",
                   None, None, None, None,
                   t_up,
                   t_md,
                   None
               ])
               try:
                   csv_file.flush()
               except Exception:
                   pass

           if t_up:
               cv2.displayOverlay("Sign Language Translation", f"Uploaded samples in {t_up:.1f} ms", 1500)
           if t_md:
               cv2.displayOverlay("Sign Language Translation", f"Downloaded model in {t_md:.1f} ms", 1500)

finally:
    cap.release() # Release camera
    cv2.destroyAllWindows() # Close UI windows
    if LOG_TO_CSV and csv_file:  # Flush/close CSV if used
        try:
            csv_file.flush(); csv_file.close()
        except Exception:
            pass
    holistic.close() # Close MediaPipe resources


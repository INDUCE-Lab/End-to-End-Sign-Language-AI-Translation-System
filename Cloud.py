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

from flask import Flask, request, jsonify, send_file
import os, time, hashlib, io
import numpy as np

## ---  Config --- ##

MODEL_DIR  = "path/to/model"
SPM_PATH  = "path/to/model/medasl_bpe.model"
UPLOAD_DIR = "path/to/uploaded_keypoints"
PORT = 0000  # add port
MAX_KEEP = 50 # how many video files to keep on edge before transmitting to cloud (adjustable)
SPECIALS_FILENAME = "special_ids.json"
SPECIALS_PATH = os.path.join(MODEL_DIR, SPECIALS_FILENAME)

## ---  Application --- ##

app = Flask(__name__) # Create the Flask application instance

os.makedirs(MODEL_DIR,  exist_ok=True) # Ensure model directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure upload directory exists

@app.get("/health")
def health():
    """Liveness probe endpoint."""
    return jsonify(status="ok"), 200 # Return health status

## ---  Utils --- ##
def sha256_bytes(b: bytes) -> str:
    """
    Compute the SHA-256 hash (hex digest) of a given byte sequence.

    Args:
        b (bytes): The raw bytes to hash.

    Returns:
        str: 64-character hexadecimal string representing the SHA-256 digest.
    """
    h = hashlib.sha256()    # Create a new SHA-256 hasher instance
    h.update(b)             # Feed the input bytes into the hasher
    return h.hexdigest()    # Return the final hash as a lowercase hex string

def header_hex(name: str) -> str:
    """
    Retrieve and normalize a specific HTTP header value as lowercase text.

    Args:
        name (str): The header name to extract (e.g., 'X-File-SHA256').

    Returns:
        str | None: Lowercased header value if present and a string, otherwise None.
    """
    v = request.headers.get(name)            # Look up the header in the current Flask request
    return v.lower() if isinstance(v, str) else None  # Lowercase it for consistent comparison


def sha256_file(path, chunk_size=1<<20):
    """Compute SHA-256 checksum of a file in streaming chunks.

    Args:
        path (str): File path to hash.
        chunk_size (int): Bytes per read (default 1 MiB).

    Returns:
        str: Hex digest of the file content.
    """ 
    # Initialize SHA-256 hasher
    h = hashlib.sha256()
    
    # Open file in binary mode, read until end of fixed-size chunks, feed chunk to the hasher, and return hex string digest
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _prune(dirpath, pattern=".npz", max_keep=50):
    """Keep only the most recent 'max_keep' files matching pattern in a directory.

    Args:
        dirpath (str): Directory to clean.
        pattern (str): Filename suffix to match (default '.npz').
        max_keep (int): Maximum files to retain.
    """
    # Build sorted list of matching files by time (oldest first)
    files = sorted(
        [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(pattern)],
        key=os.path.getmtime
    )
    # While over retention limit, remove oldest file, and drop it from the list
    while len(files) > max_keep:
        try:
            os.remove(files[0])
        except OSError:
            pass
        files.pop(0)

## ---  Model --- ##

def latest_model_info():
    """Return metadata for the most recently modified .pth model in MODEL_DIR.

    Returns:
        dict | None: Contains version, path, size, sha256, and updated_at if found, else None.
    """
    pths = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    # Collect all .pth files in model directory
    if not pths: # Return None if no model files exist
        return None
    path = max(pths, key=os.path.getmtime) # Pick the latest file by modification time
    return {
        "version": os.path.basename(path), # File name acts as model version
        "path": path, # Absolute file path
        "size_bytes": os.path.getsize(path), # File size in bytes
        "sha256": sha256_file(path), # SHA-256 checksum for integrity/cache validation
        "updated_at": os.path.getmtime(path), # Last modified timestamp
    }

@app.get("/get_spm")
def get_spm():
    """Serve the SentencePiece model with HTTP cache validators (ETag/Last-Modified)."""
    # SPM_PATH must be defined/imported, print message if missing
    if not os.path.isfile(SPM_PATH):
        print("SPM file not found")
    return send_file( # Stream the file to the client
        SPM_PATH,
        mimetype="application/octet-stream", # Generic binary type
        as_attachment=True, # Force download
        download_name="medasl2000_bpe.model", # Download filename
        conditional=True, # Enables conditional GET handling
        max_age=3600  # Cache max-age in seconds
    )

@app.get("/get_model")
def get_model():
    """Serve the latest .pth model with manual ETag/Last-Modified handling."""
    m = latest_model_info() # Inspect the newest model
    if not m: # No model available
        return jsonify(error="no model available"), 404

    etag = m["sha256"] # Use SHA-256 as strong ETag
    last_mod = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(m["updated_at"])) # Format Last-Modified as HTTP-date
    inm = request.headers.get("If-None-Match")  # Edge's ETag precondition
    ims = request.headers.get("If-Modified-Since") # Edge's Last-Modified precondition
    if inm == etag or ims == last_mod: # If edge already has current version, no modifications
        return ("", 304, {"ETag": etag, "Last-Modified": last_mod})

    resp = send_file(m["path"], mimetype="application/octet-stream", as_attachment=False, download_name=m["version"]) # Otherwise, send the model file
    resp.headers["ETag"] = etag # Attach ETag header
    resp.headers["Last-Modified"] = last_mod # Attach Last-Modified header
    return resp # Return response

@app.get("/get_specials")
def get_specials():
    """Serve special_ids.json with cache validators similar to /get_model."""
    # SPECIALS_PATH must be defined/imported, print message if missing
    if not os.path.isfile(SPECIALS_PATH):
        return jsonify(error=f"{SPECIALS_FILENAME} not found on server"), 404

    stat = os.stat(SPECIALS_PATH) # Stat the file for mtime
    etag = sha256_file(SPECIALS_PATH) # Compute hash as ETag
    last_mod = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stat.st_mtime)) # HTTP-date representation of mtime
    inm = request.headers.get("If-None-Match") # Edge's ETag precondition
    ims = request.headers.get("If-Modified-Since") # Edge's Last-Modified precondition
    if inm == etag or ims == last_mod: # Short-circuit if unchanged
        return ("", 304, {"ETag": etag, "Last-Modified": last_mod})

    resp = send_file(SPECIALS_PATH, mimetype="application/json", as_attachment=False, # Stream JSON without forcing download
                     download_name=SPECIALS_FILENAME)
    resp.headers["ETag"] = etag # Attach ETag
    resp.headers["Last-Modified"] = last_mod # Attach Last-Modified
    return resp # Return response

## ---  Transmit keypoints --- ##

@app.post("/upload_keypoints")
def upload_keypoints():
    """Accept NPZ of keypoint samples (…, 1662) and optional CSV meta; verify SHA-256 if provided; save and prune.

    Expects:
        - form-data field "file": NPZ with array `samples` of shape [N, T, 1662].
        - optional field "meta": CSV bytes (saved alongside with timestamped name).
    Returns:
        JSON with saved filenames.
    """
    # NPZ payload under field "file" must be defined/imported, print message if missing
    if "file" not in request.files:
        return jsonify(error="missing 'file'"), 400
    f = request.files["file"] # FileStorage object
    raw = f.read() # Read entire payload into memory

    # Integrity headers from edge
    file_sha_hdr = header_hex("X-File-SHA256")     # hex string or None
    meta_sha_hdr = header_hex("X-Meta-SHA256")     # hex string or None
    file_sha_srv = sha256_bytes(raw)               # server-computed

    # If client provided a hash, verify it exactly
    if file_sha_hdr is not None and file_sha_hdr != file_sha_srv:
        return jsonify(error="file SHA256 mismatch",
                       expected=file_sha_hdr, got=file_sha_srv), 400

    # validate NPZ contains 'samples' with [..., 1662]
    try:
        buf = io.BytesIO(raw) # Wrap bytes for np.load
        z = np.load(buf) # Load NPZ archive
        samples = z["samples"] # Extract the 'samples' array
        if samples.ndim != 3 or samples.shape[-1] != 1662: # Shape check
            return jsonify(error=f"bad samples shape {samples.shape}"), 400
    except Exception as e: # Robust parse error handling
        return jsonify(error=f"npz parse failed: {e}"), 400

    ts = int(time.time() * 1e6) # Microsecond timestamp for unique names
    base = f"samples_{ts}" # Base filename prefix
    npz_path = os.path.join(UPLOAD_DIR, f"{base}.npz") # Destination path for NPZ
    with open(npz_path, "wb") as out: # Save raw NPZ bytes to disk
        out.write(raw)

    # Capture the translations CSV if present (field "meta")
    meta_name = None # Default: no meta saved
    if "meta" in request.files: # If edge sent CSV metadata, access the file, read its bytes, construct CSV filename, and write in CSV
        m = request.files["meta"]
        meta_bytes = m.read()
        meta_sha_srv = sha256_bytes(meta_bytes)
        if meta_sha_hdr is not None and meta_sha_hdr != meta_sha_srv:
            # Clean up npz we just wrote to avoid orphan on failure
            try: os.remove(npz_path)
            except OSError: pass
            return jsonify(error="meta SHA256 mismatch",
                           expected=meta_sha_hdr, got=meta_sha_srv), 400
        meta_name = f"{base}_translations.csv"
        csv_path = os.path.join(UPLOAD_DIR, meta_name)
        with open(csv_path, "wb") as out:
            out.write(meta_bytes)

    _prune(UPLOAD_DIR, ".npz", MAX_KEEP) # Prune old NPZs (MAX_KEEP must be defined/imported)
    return jsonify(ok=True, saved_npz=os.path.basename(npz_path), saved_meta=meta_name,
        file_sha256=file_sha_srv,
        meta_sha256=meta_sha_srv), 200 # Return success response

if __name__ == "__main__": # Run the app only when executed directly
    # Place the latest:
    #   - model .pth in MODEL_DIR
    #   - special_ids.json in MODEL_DIR (filename must match SPECIALS_FILENAME)
    app.run(host="0.0.0.0", port=PORT) # Start Flask server on configured port

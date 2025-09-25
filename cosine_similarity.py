# Create a single "universal_cosine.py" script that can compare many file types.
# Supported:
# - Images: PNG/JPG/BMP/WebP... -> grayscale resize -> standardized -> L2 unit vector
# - PDFs: first/selected page via PyMuPDF (optional) -> image pipeline
# - DOCX: extract text via python-docx -> TF-IDF
# - Text / CSV / Code: read as text; optional simple comment stripping for code -> TF-IDF
# - Audio: mel-spectrogram via librosa -> image pipeline
# - Video: sample frames via OpenCV -> average pooled image vectors
# - Binary: byte histogram (256 bins) OR hashed 2-gram (4096 bins)
#
# Usage examples are printed by --help.
#
# Note: Optional deps: pymupdf (PDF), python-docx (DOCX), librosa (audio), opencv-python (video)

from pathlib import Path
import argparse
import io
import math
from typing import List, Tuple, Optional

import numpy as np

# Lazy imports for optional modules
PIL_Image = None
def _require_pil():
    global PIL_Image
    if PIL_Image is None:
        from PIL import Image as PIL_Image
    return PIL_Image

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False).ravel()
    n = np.linalg.norm(vec) + 1e-12
    return vec / n

# ---------------------- Image (and image-like) ----------------------
def image_to_unit_vec(img, size: int = 160) -> np.ndarray:
    Image = _require_pil()
    img = img.convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Standardize (zero mean, unit variance) for contrast / brightness robustness
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return _l2_normalize(arr)

def load_image_file(path: str):
    Image = _require_pil()
    return Image.open(path)

# ---------------------- PDF ----------------------
def load_pdf_page_as_image(path: str, page_index: int = 0, dpi: int = 200):
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PDF 지원을 위해 'pymupdf'를 설치하세요: pip install pymupdf") from e
    doc = fitz.open(path)
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"PDF 페이지 범위를 벗어남: 0 <= page_index < {len(doc)}")
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    from PIL import Image as PIL_Image  # ensure PIL present
    img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# ---------------------- Text / TF-IDF ----------------------
def read_text_file(path: str, encoding: Optional[str] = "utf-8") -> str:
    p = Path(path)
    data = p.read_bytes()
    if encoding is None:
        # try utf-8 then fallback latin-1
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    else:
        return data.decode(encoding, errors="ignore")

def strip_code_comments(text: str, file_ext: str) -> str:
    # Very simple heuristic comment stripper for popular languages
    # - Python/Shell: lines starting with #
    # - C/C++/Java/JS/TS/C#: // and /* ... */
    # - SQL: -- comments
    import re
    ext = file_ext.lower()
    t = text
    if ext in {".py", ".sh", ".rb", ".pl"}:
        t = "\n".join([line for line in t.splitlines() if not line.strip().startswith("#")])
        return t
    if ext in {".sql"}:
        t = "\n".join([line for line in t.splitlines() if not line.strip().startswith("--")])
        return t
    if ext in {".c", ".cpp", ".cc", ".h", ".hpp", ".java", ".js", ".ts", ".cs", ".go", ".php"}:
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)  # block comments
        t = "\n".join([re.sub(r"//.*$", "", line) for line in t.splitlines()])
        return t
    return t

def cosine_text(a_text: str, b_text: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
    X = vec.fit_transform([a_text, b_text])
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)

# ---------------------- DOCX ----------------------
def read_docx_text(path: str) -> str:
    try:
        import docx  # python-docx
    except Exception as e:
        raise RuntimeError("DOCX 지원을 위해 'python-docx'를 설치하세요: pip install python-docx") from e
    doc = docx.Document(path)
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
    # tables
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)

# ---------------------- Audio -> mel-spectrogram ----------------------
def audio_to_melspec_image(path: str, sr: int = 16000, n_mels: int = 128, hop_length: int = 512):
    try:
        import librosa
    except Exception as e:
        raise RuntimeError("오디오 지원을 위해 'librosa'를 설치하세요: pip install librosa") from e
    y, sr = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("오디오가 비어있습니다.")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Convert to PIL image-like by normalizing to 0-255
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    S_img = (S_norm * 255.0).astype(np.uint8)
    from PIL import Image as PIL_Image
    img = PIL_Image.fromarray(S_img)  # (n_mels, time)
    return img

# ---------------------- Video -> sampled frame vectors ----------------------
def video_to_vector(path: str, frame_interval: float = 1.0, size: int = 160) -> np.ndarray:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("비디오 지원을 위해 'opencv-python'를 설치하세요: pip install opencv-python") from e
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    step = max(int(frame_interval * fps), 1)

    Image = _require_pil()
    vecs = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            vecs.append(image_to_unit_vec(img, size=size))
        frame_idx += 1
    cap.release()
    if not vecs:
        raise RuntimeError("샘플링된 프레임이 없습니다.")
    # Average pool vectors -> single vector
    arr = np.vstack(vecs).mean(axis=0)
    return _l2_normalize(arr)

# ---------------------- Binary -> hist / hashed 2-gram ----------------------
def binary_to_histogram(path: str) -> np.ndarray:
    data = Path(path).read_bytes()
    if not data:
        return np.zeros(256, dtype=np.float32)
    hist = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
    hist /= hist.sum()
    return _l2_normalize(hist)

def binary_to_hashed_2gram(path: str, bins: int = 4096) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) < 2:
        return np.zeros(bins, dtype=np.float32)
    v = np.zeros(bins, dtype=np.float32)
    prev = data[0]
    for b in data[1:]:
        two = (prev << 8) | b  # 0..65535
        idx = (two * 1315423911) & 0xFFFFFFFF
        idx = idx % bins
        v[idx] += 1.0
        prev = b
    v /= (v.sum() + 1e-12)
    return _l2_normalize(v)

# ---------------------- Dispatcher ----------------------
def detect_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
        return "image"
    if ext in {".pdf"}:
        return "pdf"
    if ext in {".docx"}:
        return "docx"
    if ext in {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".yml", ".yaml", ".ini", ".cfg", ".log"}:
        return "text"
    if ext in {".py", ".c", ".cpp", ".cc", ".h", ".hpp", ".java", ".js", ".ts", ".cs", ".go", ".php", ".rb", ".sh", ".ps1", ".sql"}:
        return "code"
    if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        return "audio"
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    return "binary"

def file_to_vector(path: str,
                   kind: Optional[str] = None,
                   size: int = 160,
                   pdf_page: int = 0,
                   audio_sr: int = 16000,
                   video_interval: float = 1.0,
                   binary_mode: str = "hist",
                   strip_comments: bool = False) -> np.ndarray:
    if kind is None:
        kind = detect_kind(path)

    if kind == "image":
        img = load_image_file(path)
        return image_to_unit_vec(img, size=size)

    if kind == "pdf":
        img = load_pdf_page_as_image(path, page_index=pdf_page, dpi=200)
        return image_to_unit_vec(img, size=size)

    if kind == "docx":
        text = read_docx_text(path)
        return text_to_vector(text)

    if kind in {"text", "code"}:
        text = read_text_file(path, encoding=None)
        if kind == "code" and strip_comments:
            text = strip_code_comments(text, Path(path).suffix)
        return text_to_vector(text)

    if kind == "audio":
        img = audio_to_melspec_image(path, sr=audio_sr)
        return image_to_unit_vec(img, size=size)

    if kind == "video":
        return video_to_vector(path, frame_interval=video_interval, size=size)

    if kind == "binary":
        if binary_mode == "ngram2":
            return binary_to_hashed_2gram(path, bins=4096)
        return binary_to_histogram(path)

    raise ValueError(f"알 수 없는 kind: {kind}")

def text_to_vector(text: str) -> np.ndarray:
    # Use TF-IDF -> dense unit vector
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
    X = vec.fit_transform([text])
    # Convert sparse to dense safely
    dense = X.toarray()[0].astype(np.float32)
    return _l2_normalize(dense)

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # For unit vectors, cosine is dot product
    return float(np.dot(vec1, vec2))

def cosine_files(path1: str, path2: str, **kwargs) -> float:
    v1 = file_to_vector(path1, **kwargs)
    v2 = file_to_vector(path2, **kwargs)
    return cosine_sim(v1, v2)

def main():
    parser = argparse.ArgumentParser(
        description="Universal cosine similarity for images, PDFs, DOCX, text/code, audio, video, binary files."
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--kind", choices=["auto","image","pdf","docx","text","code","audio","video","binary"], default="auto",
                        help="강제 모드 설정 (기본 auto 감지)")
    parser.add_argument("--size", type=int, default=160, help="이미지/스펙트로그램/프레임 리사이즈 한 변 길이")
    parser.add_argument("--pdf-page-a", type=int, default=0, help="PDF일 때 file1 페이지 (0부터)")
    parser.add_argument("--pdf-page-b", type=int, default=0, help="PDF일 때 file2 페이지 (0부터)")
    parser.add_argument("--audio-sr", type=int, default=16000, help="오디오 리샘플링 주파수")
    parser.add_argument("--video-interval", type=float, default=1.0, help="비디오 프레임 샘플링 간격(초)")
    parser.add_argument("--binary-mode", choices=["hist","ngram2"], default="hist", help="바이너리 벡터화 방식")
    parser.add_argument("--strip-comments", action="store_true", help="code 모드에서 주석 제거")
    parser.add_argument("--debug", action="store_true", help="중간 정보 출력")

    args = parser.parse_args()

    try:
        kwargs = {
            "kind": None if args.kind == "auto" else args.kind,
            "size": args.size,
            "pdf_page": args.pdf_page_a,  # default for file1; override per file below
            "audio_sr": args.audio_sr,
            "video_interval": args.video_interval,
            "binary_mode": args.binary_mode,
            "strip_comments": args.strip_comments,
        }

        # Handle separate pdf pages for each file
        if (kwargs["kind"] in (None, "pdf")):
            v1 = file_to_vector(args.file1, **kwargs)
            kwargs2 = dict(kwargs)
            kwargs2["pdf_page"] = args.pdf_page_b
            v2 = file_to_vector(args.file2, **kwargs2)
        else:
            v1 = file_to_vector(args.file1, **kwargs)
            v2 = file_to_vector(args.file2, **kwargs)

        sim = cosine_sim(v1, v2)
        if args.debug:
            print(f"[DEBUG] kind={kwargs['kind'] or 'auto'} size={args.size} audio_sr={args.audio_sr} video_interval={args.video_interval} binary_mode={args.binary_mode}")
            print(f"[DEBUG] file1={args.file1} -> vec_dim={v1.size}")
            print(f"[DEBUG] file2={args.file2} -> vec_dim={v2.size}")
        print(f"{sim:.6f}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
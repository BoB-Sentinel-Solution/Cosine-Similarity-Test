
from pathlib import Path
import argparse
import math
import numpy as np

def _require_pil():
    from PIL import Image
    return Image

def _pad_square_and_resize(gray_img, size: int = 512):
    Image = _require_pil()
    if gray_img.mode != "L":
        gray_img = gray_img.convert("L")
    w, h = gray_img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 255)
    canvas.paste(gray_img, ((s - w)//2, (s - h)//2))
    return canvas.resize((size, size), Image.Resampling.BICUBIC)

def load_image(path: str, size: int = 512):
    Image = _require_pil()
    img = Image.open(path).convert("L")
    return _pad_square_and_resize(img, size=size)

def load_pdf_page(path: str, page_index: int = 0, size: int = 512):
    import fitz  # PyMuPDF
    from PIL import Image as PIL_Image
    doc = fitz.open(path)
    if not (0 <= page_index < len(doc)):
        raise IndexError(f"PDF page index out of range: 0..{len(doc)-1}")
    page = doc.load_page(page_index)
    mat = fitz.Matrix(200/72, 200/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
    doc.close()
    return _pad_square_and_resize(img, size=size)

def audio_to_image(path: str, size: int = 512, sr: int = 16000):
    import librosa
    from PIL import Image as PIL_Image
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    img = PIL_Image.fromarray((S_norm * 255).astype(np.uint8))
    return _pad_square_and_resize(img, size=size)

def video_to_images(path: str, size: int = 512, interval: float = 1.0):
    import cv2
    from PIL import Image as PIL_Image
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    step = max(int(fps * interval), 1)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = PIL_Image.fromarray(gray)
            frames.append(_pad_square_and_resize(img, size=size))
        idx += 1
    cap.release()
    return frames

def detect_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in {".wav",".mp3",".flac",".ogg",".m4a",".aac"}:
        return "audio"
    if ext in {".mp4",".avi",".mov",".mkv",".webm"}:
        return "video"
    return "image"

def ssim_score(A: np.ndarray, B: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(A, B, data_range=255))

def compare_files_ssim(path1: str, path2: str, kind: str = "auto", size: int = 512,
                       pdf_page_a: int = 0, pdf_page_b: int = 0, audio_sr: int = 16000,
                       video_interval: float = 1.0) -> float:
    if kind == "auto":
        kind = detect_kind(path1)
    if kind == "image":
        from PIL import Image
        A = load_image(path1, size=size)
        B = load_image(path2, size=size)
        return ssim_score(np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32))
    if kind == "pdf":
        A = load_pdf_page(path1, page_index=pdf_page_a, size=size)
        B = load_pdf_page(path2, page_index=pdf_page_b, size=size)
        return ssim_score(np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32))
    if kind == "audio":
        A = audio_to_image(path1, size=size, sr=audio_sr)
        B = audio_to_image(path2, size=size, sr=audio_sr)
        return ssim_score(np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32))
    if kind == "video":
        frames1 = video_to_images(path1, size=size, interval=video_interval)
        frames2 = video_to_images(path2, size=size, interval=video_interval)
        n = min(len(frames1), len(frames2))
        if n == 0:
            raise RuntimeError("No frames sampled from the videos.")
        scores = []
        for i in range(n):
            A = np.asarray(frames1[i], dtype=np.float32)
            B = np.asarray(frames2[i], dtype=np.float32)
            scores.append(ssim_score(A, B))
        return float(np.mean(scores))
    raise ValueError(f"Unsupported kind: {kind}")

def main():
    import argparse
    p = argparse.ArgumentParser(description="SSIM-based similarity for images, PDFs, audio (via spectrogram), and video.")
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--kind", choices=["auto","image","pdf","audio","video"], default="auto")
    p.add_argument("--size", type=int, default=512, help="resize square edge length")
    p.add_argument("--pdf-page-a", type=int, default=0)
    p.add_argument("--pdf-page-b", type=int, default=0)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--video-interval", type=float, default=1.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    try:
        sim = compare_files_ssim(args.file1, args.file2, kind=args.kind, size=args.size,
                                 pdf_page_a=args.pdf_page_a, pdf_page_b=args.pdf_page_b,
                                 audio_sr=args.audio_sr, video_interval=args.video_interval)
        if args.debug:
            print(f"[DEBUG] kind={args.kind} size={args.size} pages=({args.pdf_page_a},{args.pdf_page_b}) sr={args.audio_sr} v_interval={args.video_interval}")
        print(f"{sim:.6f}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

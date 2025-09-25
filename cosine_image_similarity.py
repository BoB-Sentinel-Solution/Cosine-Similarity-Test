from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def _to_vector(p: Path, size: int = 128) -> np.ndarray:
    img = Image.open(p).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    vec = arr.ravel().astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)

def cosine_image_similarity(img1_path: str, img2_path: str, size: int = 128) -> float:
    v1 = _to_vector(Path(img1_path), size=size)
    v2 = _to_vector(Path(img2_path), size=size)
    return float(np.dot(v1, v2))

def main():
    ap = argparse.ArgumentParser(description="Cosine similarity between two images")
    ap.add_argument("img1")
    ap.add_argument("img2")
    ap.add_argument("--size", type=int, default=128, help="resize edge length (default: 128)")
    ap.add_argument("--debug", action="store_true", help="print debug info")
    args = ap.parse_args()

    try:
        sim = cosine_image_similarity(args.img1, args.img2, size=args.size)
        if args.debug:
            print(f"[DEBUG] size={args.size}")
            print(f"[DEBUG] img1={args.img1}")
            print(f"[DEBUG] img2={args.img2}")
        print(f"{sim:.6f}")
    except Exception as e:
        # 어떤 오류라도 콘솔에 보이도록
        print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

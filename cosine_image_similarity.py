from pathlib import Path
import numpy as np
from PIL import Image

def _to_vector(p: Path, size: int = 128) -> np.ndarray:
    # 그레이스케일 → 리사이즈 → 표준화 → 평탄화 → L2 정규화
    img = Image.open(p).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    vec = arr.ravel().astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)

def cosine_image_similarity(img1_path: str, img2_path: str, size: int = 128) -> float:
    v1 = _to_vector(Path(img1_path), size=size)
    v2 = _to_vector(Path(img2_path), size=size)
    return float(np.dot(v1, v2))

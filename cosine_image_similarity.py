# pdf_or_image_cosine.py
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

def load_page_as_pil(pdf_path: str, page_index: int = 0, dpi: int = 200) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def to_unit_vec(img: Image.Image, size: int = 128) -> np.ndarray:
    img = img.convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)/255.0
    arr = (arr - arr.mean())/(arr.std() + 1e-8)
    v = arr.ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def cosine_img(img1: Image.Image, img2: Image.Image, size: int = 128) -> float:
    v1, v2 = to_unit_vec(img1, size), to_unit_vec(img2, size)
    return float(np.dot(v1, v2))

def cosine_file(a: str, b: str, size: int = 128, page_a: int = 0, page_b: int = 0) -> float:
    def load_any(p, page):
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            return load_page_as_pil(p, page)
        else:
            return Image.open(p)
    return cosine_img(load_any(a, page_a), load_any(b, page_b), size)

if __name__ == "__main__":
    # 예: 첫 페이지끼리 비교
    print(cosine_file("a.pdf", "b.pdf", size=160, page_a=0, page_b=0))

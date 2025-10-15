
# similarity_classifier_strict.py
# -*- coding: utf-8 -*-
"""
이미지 유사도 '모든 단계'를 반드시 수행하는 엄격 모드 구현 (LPIPS 입력 크기 맞춤 패치 포함).

필수 단계:
  1) pHash 해밍거리
  2) CLIP 임베딩 코사인 유사도
  3) ORB 특징점 + RANSAC 정합(inlier 비율/개수)
  4) LPIPS (작을수록 유사)  ← 크기/종횡비 차이 보정
  5) SSIM  (클수록 유사)

어느 한 단계라도 수행 불가하면 즉시 명시적 예외를 발생시킵니다.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np
import json
import os
import sys
import io
import warnings

# 경고(QuickGELU mismatch 등) 숨기기 선택사항
warnings.filterwarnings("ignore", message="QuickGELU mismatch")

# ---- Hard imports (모두 필수) ----
from PIL import Image as PILImage
from imagehash import phash as _phash

import cv2
from skimage.metrics import structural_similarity as ssim

import torch
import lpips
import open_clip


@dataclass
class Thresholds:
    # Hash
    hamming_th_strong: int = 5
    hamming_th_maybe: int = 10  # 참고값
    # Embedding (cosine)
    cos_th_high: float = 0.55
    cos_th_mid_low: float = 0.40
    # Geometric (ORB + RANSAC)
    inlier_ratio_th: float = 0.20
    inlier_count_th: int = 30
    # Perceptual
    lpips_th: float = 0.25
    ssim_th: float = 0.85


class StrictSimilarityClassifier:
    def __init__(self, device: str | None = None, thresholds: Thresholds | None = None):
        self.th = thresholds or Thresholds()

        # device 결정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device not in ("cuda", "cpu"):
                raise ValueError("device must be 'cuda' or 'cpu'")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            self.device = device

        # LPIPS 모델
        self.lpips_net = lpips.LPIPS(net='alex').to(self.device).eval()

        # CLIP 모델
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.clip_model = self.clip_model.eval().to(self.device)

        # ORB 매처
        self.orb = cv2.ORB_create(5000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # ---- 파일/이미지 읽기 (한글 경로 안전) ----

    @staticmethod
    def _ensure_file(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _read_rgb(self, path: str) -> PILImage.Image:
        # BytesIO 경로로 PIL 읽기 → 유니코드 경로 안전
        data = self._read_bytes(path)
        return PILImage.open(io.BytesIO(data)).convert("RGB")

    def _read_gray_cv(self, path: str):
        # cv2.imdecode 경유 → 유니코드 경로 안전
        data = np.frombuffer(self._read_bytes(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"OpenCV failed to decode image (grayscale): {path}")
        return img

    # ---- 개별 지표 계산 ----

    def compute_phash_hamming(self, imgA_path: str, imgB_path: str) -> int:
        A = self._read_rgb(imgA_path)
        B = self._read_rgb(imgB_path)
        return int(_phash(A) - _phash(B))

    def compute_orb_ransac_inliers(self, imgA_path: str, imgB_path: str) -> Tuple[float, int]:
        img1 = self._read_gray_cv(imgA_path)
        img2 = self._read_gray_cv(imgB_path)

        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            # 엄격 모드: 이 경우도 실패로 간주
            raise RuntimeError("Not enough ORB keypoints/descriptors for geometric matching.")

        matches = self.bf.match(des1, des2)
        if len(matches) < 8:
            raise RuntimeError("Not enough BF matches for RANSAC.")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            raise RuntimeError("RANSAC failed to produce inlier mask.")
        inliers = int(mask.sum())
        ratio = inliers / max(len(matches), 1)
        return float(ratio), int(inliers)

    def compute_ssim(self, imgA_path: str, imgB_path: str) -> float:
        img1 = self._read_gray_cv(imgA_path)
        img2 = self._read_gray_cv(imgB_path)
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        if h < 8 or w < 8:
            raise RuntimeError("Images too small for SSIM after size alignment.")
        img1r = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        img2r = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        val = float(ssim(img1r, img2r))
        return val

    def compute_lpips(self, imgA_path: str, imgB_path: str) -> float:
        """
        LPIPS는 두 입력 텐서의 H,W가 동일해야 하므로
        - 짧은 변을 256으로 맞춰 종횡비 유지 리사이즈
        - 두 결과의 공통 최소 크기로 '중앙 크롭'하여 H,W 동일 보장
        """
        def resize_min_keep_ratio(im: PILImage.Image, short=256):
            w, h = im.size
            scale = short / min(w, h)
            return im.resize((int(round(w*scale)), int(round(h*scale))), PILImage.BICUBIC)

        def center_crop_same(a: PILImage.Image, b: PILImage.Image) -> tuple[PILImage.Image, PILImage.Image]:
            wa, ha = a.size
            wb, hb = b.size
            W = min(wa, wb)
            H = min(ha, hb)
            def cc(im, W, H):
                w, h = im.size
                left = (w - W) // 2
                top  = (h - H) // 2
                return im.crop((left, top, left+W, top+H))
            return cc(a, W, H), cc(b, W, H)

        A = resize_min_keep_ratio(self._read_rgb(imgA_path), short=256)
        B = resize_min_keep_ratio(self._read_rgb(imgB_path), short=256)
        A, B = center_crop_same(A, B)  # H,W 동일 보장

        def to_tensor(im: PILImage.Image) -> torch.Tensor:
            x = torch.from_numpy(np.array(im).transpose(2,0,1)).float() / 127.5 - 1.0
            return x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            d = self.lpips_net(to_tensor(A), to_tensor(B)).item()
        return float(d)

    def compute_clip_cosine(self, imgA_path: str, imgB_path: str) -> float:
        def emb(path: str) -> np.ndarray:
            im = self._read_rgb(path)
            with torch.no_grad():
                x = self.clip_preprocess(im).unsqueeze(0).to(self.device)
                feat = self.clip_model.encode_image(x)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.detach().cpu().numpy()[0]

        v1, v2 = emb(imgA_path), emb(imgB_path)
        return float(np.dot(v1, v2))

    # ---- 최종 판정 ----

    def classify(self, imgA_path: str, imgB_path: str) -> Dict[str, Any]:
        # 파일 존재검사
        self._ensure_file(imgA_path)
        self._ensure_file(imgB_path)

        # 1) pHash
        Hd = self.compute_phash_hamming(imgA_path, imgB_path)

        # 2) CLIP cosine
        S = self.compute_clip_cosine(imgA_path, imgB_path)

        # 3) Geometric (ORB + RANSAC)
        rin, nin = self.compute_orb_ransac_inliers(imgA_path, imgB_path)

        # 4) LPIPS
        lpips_val = self.compute_lpips(imgA_path, imgB_path)

        # 5) SSIM
        ssim_val = self.compute_ssim(imgA_path, imgB_path)

        metrics: Dict[str, Any] = {
            "phash_hamming": Hd,
            "cosine_clip": S,
            "inlier_ratio": rin,
            "inlier_count": nin,
            "lpips": lpips_val,
            "ssim": ssim_val,
            "device": self.device,
        }

        th = self.th

        # ---- 의사결정 규칙 ----
        if Hd <= th.hamming_th_strong:
            clazz, conf, note = "A_near_duplicate", 0.98, "pHash strong match"
        elif S >= th.cos_th_high:
            if rin >= th.inlier_ratio_th:
                clazz, conf, note = "B_same_scene", 0.95, "high cosine; geom inliers OK"
            elif (lpips_val <= th.lpips_th) or (ssim_val >= th.ssim_th):
                clazz, conf, note = "C_semantic_similar", 0.90, "high cosine; perceptual OK"
            else:
                clazz, conf, note = "C_semantic_similar", 0.75, "high cosine; weak geom/perceptual"
        elif th.cos_th_mid_low <= S < th.cos_th_high:
            if rin >= max(th.inlier_ratio_th, 0.25):
                clazz, conf, note = "B_same_scene", 0.85, "mid cosine; strong geom"
            elif (lpips_val <= (th.lpips_th - 0.03)) or (ssim_val >= (th.ssim_th + 0.03)):
                clazz, conf, note = "C_semantic_similar", 0.80, "mid cosine; strong perceptual"
            else:
                clazz, conf, note = "uncertain", 0.55, "mid cosine; weak geom/perceptual"
        else:
            clazz, conf, note = "D_dissimilar", 0.95, "cosine low; no strong hash match"

        return {
            "class": clazz,
            "confidence": conf,
            "metrics": metrics,
            "thresholds": asdict(th),
            "notes": [note],
        }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Strict Image Similarity Classifier (all checks required, LPIPS shape fix)")
    p.add_argument("imgA")
    p.add_argument("imgB")
    p.add_argument("--device", choices=["cpu","cuda"], default=None,
                   help="강제 디바이스 선택 (미지정 시 자동)")
    args = p.parse_args()

    try:
        clf = StrictSimilarityClassifier(device=args.device)
        res = clf.classify(args.imgA, args.imgB)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

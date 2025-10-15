# similarity_classifier_strict.py
# -*- coding: utf-8 -*-
"""
모든 임계/가드/모델/런타임 설정은 반드시 YAML에서만 로드합니다.
내장 기본값 없음. --config 미지정·필드 누락 시 즉시 에러.

필수 단계:
- pHash 해밍거리
- CLIP 임베딩 코사인 유사도
- ORB + RANSAC (inlier 비율/개수)
- LPIPS (중앙 크롭으로 H,W 맞춤)
- SSIM

보강:
- 유니코드 경로 안전 읽기(BytesIO, cv2.imdecode)
- QuickGELU 경고 숨김
- '강한 비유사' 가드룰(설정에서 주입)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import json
import os
import sys
import io
import warnings

warnings.filterwarnings("ignore", message="QuickGELU mismatch")

# 필수 외부 라이브러리
from PIL import Image as PILImage
from imagehash import phash as _phash
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import open_clip
import yaml  # 반드시 설치 필요: pip install pyyaml


# -----------------------------
# YAML 스키마 (모든 필드 필수)
# -----------------------------
@dataclass
class Thresholds:
    # Hash
    hamming_th_strong: int
    hamming_th_maybe: int
    # Embedding
    cos_th_high: float
    cos_th_mid_low: float
    # Geometry
    inlier_ratio_th: float
    inlier_count_th: int
    # Perceptual
    lpips_th: float
    ssim_th: float
    # Guards (hard dissimilar)
    guard_phash_min: int
    guard_lpips_min: float
    guard_inlier_ratio_max: float
    guard_inlier_count_max: int


@dataclass
class ModelConfig:
    clip_arch: str
    clip_pretrained: str
    lpips_net: str


@dataclass
class RuntimeConfig:
    device: str               # "auto" | "cpu" | "cuda"
    orb_max_features: int
    ransac_reproj_thresh: float
    lpips_resize_short: int
    lpips_center_crop_same: bool


# -----------------------------
# YAML 로딩 & 검증
# -----------------------------
REQUIRED_KEYS = {
    "models": ["clip.arch", "clip.pretrained", "lpips.net"],
    "stages.hash": ["hamming_th_strong", "hamming_th_maybe"],
    "stages.embedding": ["cosine_thresholds.high", "cosine_thresholds.mid_low"],
    "stages.geometry": ["inlier_ratio_th", "inlier_count_th", "orb_max_features", "ransac_reproj_thresh"],
    "stages.perceptual": ["ssim_th", "lpips_th", "lpips_resize_short", "lpips_center_crop_same"],
    "guards": ["phash_min", "lpips_min", "inlier_ratio_max", "inlier_count_max"],
    "device": [],
}

def get(cfg: Dict[str, Any], path: str):
    cur = cfg
    for k in path.split("."):
        if k not in cur:
            raise KeyError(path)
        cur = cur[k]
    return cur

def load_yaml_config(path: str) -> Tuple[RuntimeConfig, ModelConfig, Thresholds]:
    if not path:
        raise RuntimeError("--config <yaml> 을 반드시 지정해야 합니다.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 필수 키 검사
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        try:
            if section != "device":
                _ = get(cfg, section)  # 섹션 존재 검사
        except KeyError:
            missing.append(section)
        for k in keys:
            try:
                _ = get(cfg, f"{section}.{k}") if section else get(cfg, k)
            except KeyError:
                missing.append(f"{section}.{k}" if section else k)
    # device는 루트 키여야 함
    if "device" not in cfg:
        missing.append("device")
    if missing:
        raise KeyError("YAML 누락 키: " + ", ".join(sorted(set(missing))))

    # Dataclass 채우기
    rt = RuntimeConfig(
        device=str(cfg["device"]),
        orb_max_features=int(get(cfg, "stages.geometry.orb_max_features")),
        ransac_reproj_thresh=float(get(cfg, "stages.geometry.ransac_reproj_thresh")),
        lpips_resize_short=int(get(cfg, "stages.perceptual.lpips_resize_short")),
        lpips_center_crop_same=bool(get(cfg, "stages.perceptual.lpips_center_crop_same")),
    )

    mc = ModelConfig(
        clip_arch=str(get(cfg, "models.clip.arch")),
        clip_pretrained=str(get(cfg, "models.clip.pretrained")),
        lpips_net=str(get(cfg, "models.lpips.net")),
    )

    th = Thresholds(
        hamming_th_strong=int(get(cfg, "stages.hash.hamming_th_strong")),
        hamming_th_maybe=int(get(cfg, "stages.hash.hamming_th_maybe")),
        cos_th_high=float(get(cfg, "stages.embedding.cosine_thresholds.high")),
        cos_th_mid_low=float(get(cfg, "stages.embedding.cosine_thresholds.mid_low")),
        inlier_ratio_th=float(get(cfg, "stages.geometry.inlier_ratio_th")),
        inlier_count_th=int(get(cfg, "stages.geometry.inlier_count_th")),
        lpips_th=float(get(cfg, "stages.perceptual.lpips_th")),
        ssim_th=float(get(cfg, "stages.perceptual.ssim_th")),
        guard_phash_min=int(get(cfg, "guards.phash_min")),
        guard_lpips_min=float(get(cfg, "guards.lpips_min")),
        guard_inlier_ratio_max=float(get(cfg, "guards.inlier_ratio_max")),
        guard_inlier_count_max=int(get(cfg, "guards.inlier_count_max")),
    )

    return rt, mc, th


# -----------------------------
# 분류기
# -----------------------------
class StrictSimilarityClassifier:
    def __init__(self, rt: RuntimeConfig, mc: ModelConfig, th: Thresholds):
        self.th = th

        # device
        if rt.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif rt.device in ("cpu", "cuda"):
            if rt.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA 요청됨 그러나 사용 불가.")
            self.device = rt.device
        else:
            raise ValueError("device는 'auto' | 'cpu' | 'cuda' 중 하나여야 합니다.")

        # LPIPS 모델
        self.lpips_net = lpips.LPIPS(net=mc.lpips_net).to(self.device).eval()

        # CLIP 모델
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            mc.clip_arch, pretrained=mc.clip_pretrained
        )
        self.clip_model = self.clip_model.eval().to(self.device)

        # ORB/RANSAC
        self.orb = cv2.ORB_create(rt.orb_max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_reproj_thresh = rt.ransac_reproj_thresh

        # LPIPS 리사이즈 정책
        self.lpips_resize_short = rt.lpips_resize_short
        self.lpips_center_crop_same = rt.lpips_center_crop_same

    # ---- 파일/이미지 읽기 ----
    @staticmethod
    def _ensure_file(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _read_rgb(self, path: str) -> PILImage.Image:
        data = self._read_bytes(path)
        return PILImage.open(io.BytesIO(data)).convert("RGB")

    def _read_gray_cv(self, path: str) -> np.ndarray:
        data = np.frombuffer(self._read_bytes(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"OpenCV failed to decode image (grayscale): {path}")
        return img

    # ---- 지표 계산 ----
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
            raise RuntimeError("Not enough ORB keypoints/descriptors for geometric matching.")

        matches = self.bf.match(des1, des2)
        if len(matches) < 8:
            raise RuntimeError("Not enough BF matches for RANSAC.")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_thresh)
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
        return float(ssim(img1r, img2r))

    def compute_lpips(self, imgA_path: str, imgB_path: str) -> float:
        def resize_min_keep_ratio(im: PILImage.Image, short=256):
            w, h = im.size
            scale = short / min(w, h)
            return im.resize((int(round(w*scale)), int(round(h*scale))), PILImage.BICUBIC)

        def center_crop_same(a: PILImage.Image, b: PILImage.Image):
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

        A = resize_min_keep_ratio(self._read_rgb(imgA_path), short=self.lpips_resize_short)
        B = resize_min_keep_ratio(self._read_rgb(imgB_path), short=self.lpips_resize_short)
        if self.lpips_center_crop_same:
            A, B = center_crop_same(A, B)

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
        self._ensure_file(imgA_path)
        self._ensure_file(imgB_path)

        Hd = self.compute_phash_hamming(imgA_path, imgB_path)
        S = self.compute_clip_cosine(imgA_path, imgB_path)
        rin, nin = self.compute_orb_ransac_inliers(imgA_path, imgB_path)
        lpips_val = self.compute_lpips(imgA_path, imgB_path)
        ssim_val = self.compute_ssim(imgA_path, imgB_path)

        th = self.th
        metrics: Dict[str, Any] = {
            "phash_hamming": Hd,
            "cosine_clip": S,
            "inlier_ratio": rin,
            "inlier_count": nin,
            "lpips": lpips_val,
            "ssim": ssim_val,
            "device": self.device,
        }

        # 강한 비유사 가드
        hard_dissimilar = (
            (Hd >= th.guard_phash_min) and
            (lpips_val >= th.guard_lpips_min) and
            ((rin < th.guard_inlier_ratio_max) or (nin < th.guard_inlier_count_max))
        )
        if hard_dissimilar:
            return {
                "class": "D_dissimilar",
                "confidence": 0.97,
                "metrics": metrics,
                "thresholds": th.__dict__,
                "notes": ["guard: high pHash + high LPIPS + poor geometry"],
            }

        # 의사결정 규칙
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
            "thresholds": th.__dict__,
            "notes": [note],
        }


# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Strict Image Similarity Classifier (YAML-only config)")
    p.add_argument("--config", required=True, help="YAML 설정 파일 경로 (필수)")
    p.add_argument("imgA")
    p.add_argument("imgB")
    args = p.parse_args()

    try:
        rt, mc, th = load_yaml_config(args.config)
        clf = StrictSimilarityClassifier(rt, mc, th)
        res = clf.classify(args.imgA, args.imgB)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

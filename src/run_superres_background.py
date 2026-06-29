#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_superres_background.py

PNG Port Moresby 해양 관측자료의 공간 해상도를 8×4 km에서 1×1 km로 증강하기 위한
background 실행용 스크립트입니다.

주요 기능
---------
1. 입력 자료
   - 해수 온도 및 염도 자료: T0M, S0M, T10M, S10M
   - 기본 입력 경로: <repo>/data/raw

2. 해상도 증강 기법
   - Uniform: 저해상도 격자 값을 고해상도 격자에 반복 복제하는 기준선 방법
   - Bilinear: 인접 격자 간 선형 보간으로 연속적인 공간 분포를 생성
   - Kriging: 공간 자기상관성을 반영하여 미관측 격자 값을 추정
   - DIP: Bilinear 결과를 입력으로 사용하고 Kriging 결과를 pseudo GT로 활용하는
          Kriging-supervised 딥러닝 기반 refinement 방법

3. 자원 활용
   - Uniform, Bilinear, Kriging은 CPU 병렬 처리로 실행
   - DIP는 CUDA > MPS > CPU 순으로 사용 가능한 장치를 자동 선택
   - CPU thread oversubscription 방지를 위해 주요 수치 연산 thread 수를 기본 1로 제한

4. 출력 구조
   - 결과는 method별 하위 디렉토리에 저장됩니다.

   OUT_DIR/
   ├── uniform/T0M_uniform_1km.csv ...
   ├── bilinear/T0M_bilinear_1km.csv ...
   ├── kriging/T0M_kriging_1km.csv ...
   └── dip/T0M_dip_1km.csv ...

실행 예시
---------
# 전체 방법 실행
nohup python -u run_superres_background.py \
  > superres_background_max_resource.log 2>&1 &

# CPU 기반 방법만 실행
nohup python -u run_superres_background.py \
  --methods uniform bilinear kriging \
  > superres_background_cpu.log 2>&1 &

# 딥러닝 기반 방법만 실행
nohup python -u run_superres_background.py \
  --methods dip --dl-iters 300 \
  > superres_background_dl.log 2>&1 &
"""

# ============================================================
# 0) CPU thread 설정
#    병렬 처리 시 수치 연산 라이브러리의 thread 과다 사용을 방지합니다.
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# 1) 라이브러리
# ============================================================

import argparse
import re
import time
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 2) 기본 경로 및 실행 설정
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "data"
INPUT_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "superres_1km"

DEFAULT_METHODS = ["uniform", "bilinear", "kriging", "dip"]
DEFAULT_VARIABLES = ["T0M", "S0M", "T10M", "S10M"]

# 입력 격자(8×4 km)를 1×1 km 격자로 증강하기 위한 배율
DEFAULT_SCALE_Y = 4
DEFAULT_SCALE_X = 8

# CPU 기반 방법은 기본적으로 전체 CPU core를 활용
DEFAULT_CPU_WORKERS = os.cpu_count() or 1

# 딥러닝 기반 refinement 설정
DEFAULT_DL_ITERS = 300
DEFAULT_DL_LR = 1e-3
DEFAULT_DL_BATCH_SIZE = 1  # 시점별 격자 단위 학습
DEFAULT_OVERWRITE = False


# ============================================================
# 3) 로그 및 진행률 출력
# ============================================================

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def log(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def progress_iter(iterable, total=None, desc=""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


# ============================================================
# 4) 입력 경로 처리
# ============================================================

def resolve_existing_path(candidates, name="file"):
    if isinstance(candidates, (str, Path)):
        candidates = [Path(candidates)]

    for p in candidates:
        p = Path(p)
        if p.exists():
            return p

    raise FileNotFoundError(f"{name}을 찾지 못했습니다. candidates={candidates}")


def build_input_paths(input_dir: Path):
    input_dir = Path(input_dir)

    paths = {
        "T0M": [
            input_dir / "PNG_Port_Moresby_해수온도_0m.csv",
            input_dir / "PNG_Port_Moresby_온도_0m.csv",
        ],
        "T10M": [
            input_dir / "PNG_Port_Moresby_해수온도_10m.csv",
            input_dir / "PNG_Port_Moresby_온도_10m.csv",
        ],
        "S0M": [
            input_dir / "PNG_Port_Moresby_해수염도_0m.csv",
            input_dir / "PNG_Port_Moresby_염도_0m.csv",
        ],
        "S10M": [
            input_dir / "PNG_Port_Moresby_해수염도_10m.csv",
            input_dir / "PNG_Port_Moresby_염도_10m.csv",
        ],
    }

    resolved = {}
    for tag, candidates in paths.items():
        resolved[tag] = resolve_existing_path(candidates, name=tag)

    return resolved


# ============================================================
# 5) CSV 입출력 처리
# ============================================================

def _find_time_col(cols):
    for c in cols:
        if c.strip().lower() == "time":
            return c
    return None


def _point_sort_key(c: str):
    m = re.match(r"Point_R(\d+)_C(\d+)$", c)
    if m:
        return (0, int(m.group(1)), int(m.group(2)))

    m = re.match(r"Point_(\d+)$", c)
    if m:
        return (1, int(m.group(1)))

    return (9, c)


def _best_factor_pair(K: int):
    best = None
    for h in range(1, int(np.sqrt(K)) + 1):
        if K % h == 0:
            w = K // h
            if best is None or abs(h - w) < abs(best[0] - best[1]):
                best = (h, w)
    return best


def infer_hw(point_cols):
    rc = [re.match(r"Point_R(\d+)_C(\d+)$", c) for c in point_cols]
    if all(m is not None for m in rc):
        rows = sorted(set(int(m.group(1)) for m in rc))
        cols = sorted(set(int(m.group(2)) for m in rc))
        return len(rows), len(cols)

    K = len(point_cols)
    best = _best_factor_pair(K)
    if best is None:
        raise ValueError(f"Point 개수 K={K}에 대해 H,W 추정 실패")
    return best


def load_wide_to_matrices(path: Path, fill_nan=True):
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    time_col = _find_time_col(df.columns)
    if time_col is None:
        raise ValueError(f"[{path}] Time 컬럼 없음")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    point_cols = [c for c in df.columns if c.startswith("Point_")]
    if not point_cols:
        raise ValueError(f"[{path}] Point_* 컬럼 없음")

    point_cols = sorted(point_cols, key=_point_sort_key)

    if fill_nan:
        df[point_cols] = df[point_cols].interpolate(limit_direction="both")
        med = df[point_cols].median(numeric_only=True)
        df[point_cols] = df[point_cols].fillna(med).fillna(0.0)

    H, W = infer_hw(point_cols)
    values = df[point_cols].to_numpy(np.float32)
    mats = values.reshape(len(df), H, W)

    return df[time_col].to_numpy(), mats, point_cols, (H, W)


def matrices_to_wide_csv_rc(times, mats_thw, out_path: Path, one_indexed=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T, H, W = mats_thw.shape

    r_start = 1 if one_indexed else 0
    c_start = 1 if one_indexed else 0

    cols = []
    for r in range(r_start, r_start + H):
        for c in range(c_start, c_start + W):
            cols.append(f"Point_R{r}_C{c}")

    flat = mats_thw.reshape(T, H * W)
    out = pd.DataFrame(flat, columns=cols)
    out.insert(0, "Time", pd.to_datetime(times))
    out.to_csv(out_path, index=False)

    return out_path


# ============================================================
# 6) 해상도 증강 기법
# ============================================================

def upscale_uniform(mat_hw, sy, sx):
    return np.kron(mat_hw, np.ones((sy, sx), dtype=np.float32))


def upscale_bilinear(mat_hw, sy, sx):
    H, W = mat_hw.shape
    H2, W2 = H * sy, W * sx

    ys = np.linspace(0, H - 1, H2)
    xs = np.linspace(0, W - 1, W2)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = (ys - y0).reshape(-1, 1).astype(np.float32)
    wx = (xs - x0).reshape(1, -1).astype(np.float32)

    Ia = mat_hw[y0[:, None], x0[None, :]]
    Ib = mat_hw[y0[:, None], x1[None, :]]
    Ic = mat_hw[y1[:, None], x0[None, :]]
    Id = mat_hw[y1[:, None], x1[None, :]]

    return (1 - wy) * (1 - wx) * Ia + (1 - wy) * wx * Ib + wy * (1 - wx) * Ic + wy * wx * Id


def upscale_kriging_gpr(mat_hw, sy, sx, random_state=0):
    H, W = mat_hw.shape
    H2, W2 = H * sy, W * sx

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X_train = np.column_stack([yy.ravel(), xx.ravel()])
    y_train = mat_hw.ravel()

    if np.isnan(y_train).any():
        y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))

    yy2, xx2 = np.meshgrid(
        np.linspace(0, H - 1, H2),
        np.linspace(0, W - 1, W2),
        indexing="ij",
    )
    X_pred = np.column_stack([yy2.ravel(), xx2.ravel()])

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(1.0, (1e-2, 1e2))
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=random_state,
    )
    gpr.fit(X_train, y_train)

    y_pred = gpr.predict(X_pred).astype(np.float32)
    return y_pred.reshape(H2, W2)


def _worker_upscale(args):
    """Uniform, Bilinear, Kriging을 시점 단위로 병렬 처리합니다."""
    idx, mat_hw, method, sy, sx = args

    if method == "uniform":
        out = upscale_uniform(mat_hw, sy, sx)
    elif method == "bilinear":
        out = upscale_bilinear(mat_hw, sy, sx)
    elif method == "kriging":
        out = upscale_kriging_gpr(mat_hw, sy, sx, random_state=0)
    else:
        raise ValueError(f"Unsupported CPU method: {method}")

    return idx, out


def run_cpu_parallel(mats, method, sy, sx, cpu_workers):
    """CPU 기반 해상도 증강을 병렬 실행하고 원래 시계열 순서로 복원합니다."""
    T = mats.shape[0]
    results = [None] * T

    worker_args = [(i, mats[i], method, sy, sx) for i in range(T)]

    if cpu_workers <= 1:
        for args in progress_iter(worker_args, total=T, desc=f"{method}"):
            idx, out = _worker_upscale(args)
            results[idx] = out
    else:
        with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
            futures = [ex.submit(_worker_upscale, args) for args in worker_args]

            for fut in progress_iter(as_completed(futures), total=T, desc=f"{method}"):
                idx, out = fut.result()
                results[idx] = out

    return np.stack(results, axis=0).astype(np.float32)


# ============================================================
# 7) Kriging-supervised 딥러닝 기반 해상도 증강
# ============================================================

class SRRefinementNet(nn.Module):
    """Bilinear HR 입력을 Kriging HR pseudo GT에 맞게 보정하는 CNN refinement network입니다."""

    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 입력장을 보존하면서 잔차 성분을 학습
        return x + self.net(x)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_for_device(device):
    """선택된 장치에 맞게 PyTorch 실행 환경을 설정합니다."""
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        log(f"[GPU] CUDA enabled: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        log("[GPU] Apple MPS enabled")
    else:
        cpu_threads = os.cpu_count() or 1
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
        log(f"[CPU] Torch CPU threads set to {cpu_threads}")


def standardize_np(x, eps=1e-6):
    x = x.astype(np.float32)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < eps:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def upscale_kriging_supervised_dl(
    mat_hw,
    sy,
    sx,
    pseudo_hr=None,
    iters=300,
    lr=1e-3,
    seed=0,
    device=None,
    verbose=False,
):
    """Bilinear HR을 입력으로, Kriging HR을 pseudo GT로 사용하여 고해상도장을 보정합니다."""
    if device is None:
        device = pick_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_hr = upscale_bilinear(mat_hw, sy, sx).astype(np.float32)

    if pseudo_hr is None:
        pseudo_hr = upscale_kriging_gpr(mat_hw, sy, sx, random_state=seed).astype(np.float32)
    else:
        pseudo_hr = pseudo_hr.astype(np.float32)

    x_std, _, _ = standardize_np(x_hr)
    y_std, y_mu, y_sd = standardize_np(pseudo_hr)

    x_t = torch.tensor(x_std, dtype=torch.float32)[None, None, :, :].to(device)
    y_t = torch.tensor(y_std, dtype=torch.float32)[None, None, :, :].to(device)

    net = SRRefinementNet(base=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for t in range(iters):
        opt.zero_grad(set_to_none=True)
        pred = net(x_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        opt.step()

        if verbose and (t + 1) % max(1, iters // 5) == 0:
            log(f"  DL [{device}] {t+1}/{iters} loss={loss.item():.6f}")

    with torch.no_grad():
        pred_std = net(x_t).detach().cpu().numpy()[0, 0]

    pred_hr = pred_std * y_sd + y_mu
    return pred_hr.astype(np.float32)


def run_dl_serial(mats, sy, sx, dl_iters, dl_lr, device, dl_verbose):
    """각 시점별로 Kriging pseudo GT를 생성하고 딥러닝 refinement를 순차 실행합니다."""
    T = mats.shape[0]
    results = []

    configure_torch_for_device(device)

    for t in progress_iter(range(T), total=T, desc="kriging-supervised-dl"):
        pseudo_hr = upscale_kriging_gpr(mats[t], sy, sx, random_state=0)

        pred_hr = upscale_kriging_supervised_dl(
            mat_hw=mats[t],
            sy=sy,
            sx=sx,
            pseudo_hr=pseudo_hr,
            iters=dl_iters,
            lr=dl_lr,
            seed=0,
            device=device,
            verbose=dl_verbose,
        )
        results.append(pred_hr)

        if (t + 1) % 100 == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return np.stack(results, axis=0).astype(np.float32)


# ============================================================
# 8) 변수별 해상도 증강 실행
# ============================================================

def superres_and_save_one(
    path_in: Path,
    var_tag: str,
    method: str,
    sy: int,
    sx: int,
    out_dir: Path,
    overwrite=False,
    cpu_workers=DEFAULT_CPU_WORKERS,
    dl_iters=DEFAULT_DL_ITERS,
    dl_lr=DEFAULT_DL_LR,
    dl_verbose=False,
    device=None,
):
    method = method.lower()
    if method == "deep":
        method = "dip"

    out_path = Path(out_dir) / method / f"{var_tag}_{method}_1km.csv"

    if out_path.exists() and not overwrite:
        log(f"[SKIP] exists: {out_path}")
        return out_path

    times, mats, point_cols, (H, W) = load_wide_to_matrices(path_in, fill_nan=True)
    log(f"[INFO] {var_tag} ({method}) LR grid={H}x{W}, T={mats.shape[0]}")

    if method in ["uniform", "bilinear", "kriging"]:
        log(f"[CPU] Running {method} with workers={cpu_workers}")
        mats_hr = run_cpu_parallel(
            mats=mats,
            method=method,
            sy=sy,
            sx=sx,
            cpu_workers=cpu_workers,
        )

    elif method == "dip":
        if device is None:
            device = pick_device()
        log(f"[DL] Running Kriging-supervised DL on device={device}")
        mats_hr = run_dl_serial(
            mats=mats,
            sy=sy,
            sx=sx,
            dl_iters=dl_iters,
            dl_lr=dl_lr,
            device=device,
            dl_verbose=dl_verbose,
        )

    else:
        raise ValueError("method must be one of: uniform, bilinear, kriging, dip/deep")

    matrices_to_wide_csv_rc(times, mats_hr, out_path)
    log(f"[SAVE] {out_path} | HR shape={mats_hr.shape}")

    return out_path


# ============================================================
# 9) 실행 인자 설정
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PNG ocean-data super-resolution from 8×4 km to 1×1 km."
    )

    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))

    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Super-resolution methods: uniform bilinear kriging dip. 'deep' is treated as dip.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="Input variables to process: T0M S0M T10M S10M.",
    )

    parser.add_argument("--scale-y", type=int, default=DEFAULT_SCALE_Y)
    parser.add_argument("--scale-x", type=int, default=DEFAULT_SCALE_X)

    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Number of CPU workers for uniform/bilinear/kriging. Default: all available cores.",
    )

    parser.add_argument("--dl-iters", type=int, default=DEFAULT_DL_ITERS)
    parser.add_argument("--dl-lr", type=float, default=DEFAULT_DL_LR)
    parser.add_argument("--dl-verbose", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cuda", "mps", "cpu"],
        help="Device for dip/deep. Default: automatic selection.",
    )

    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)

    return parser.parse_args()


def normalize_methods(methods):
    out = []
    for m in methods:
        m = m.lower()
        if m == "deep":
            m = "dip"
        if m not in ["uniform", "bilinear", "kriging", "dip"]:
            raise ValueError(f"Unknown method: {m}")
        out.append(m)
    return out


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = normalize_methods(args.methods)
    variables = [v.upper() for v in args.variables]

    device = args.device or pick_device()

    log("============================================================")
    log("PNG ocean-data super-resolution runner")
    log("============================================================")
    log(f"BASE_DIR    = {BASE_DIR}")
    log(f"input_dir   = {input_dir}")
    log(f"out_dir     = {out_dir}")
    log(f"methods     = {methods}")
    log(f"variables   = {variables}")
    log(f"scale_y/x   = {args.scale_y}/{args.scale_x}")
    log(f"cpu_workers = {args.cpu_workers}")
    log(f"device      = {device}")
    log(f"dl_iters    = {args.dl_iters}")
    log(f"dl_lr       = {args.dl_lr}")
    log(f"overwrite   = {args.overwrite}")

    input_paths = build_input_paths(input_dir)

    summary = []
    t0 = time.time()

    for method in methods:
        for var_tag in variables:
            if var_tag not in input_paths:
                raise ValueError(f"Unknown variable: {var_tag}")

            start = time.time()
            try:
                out_path = superres_and_save_one(
                    path_in=input_paths[var_tag],
                    var_tag=var_tag,
                    method=method,
                    sy=args.scale_y,
                    sx=args.scale_x,
                    out_dir=out_dir,
                    overwrite=args.overwrite,
                    cpu_workers=args.cpu_workers,
                    dl_iters=args.dl_iters,
                    dl_lr=args.dl_lr,
                    dl_verbose=args.dl_verbose,
                    device=device,
                )

                elapsed = time.time() - start
                summary.append({
                    "method": method,
                    "variable": var_tag,
                    "input": str(input_paths[var_tag]),
                    "output": str(out_path),
                    "status": "done",
                    "elapsed_sec": elapsed,
                })

            except Exception as e:
                elapsed = time.time() - start
                log(f"[ERROR] method={method}, variable={var_tag}: {e}")
                summary.append({
                    "method": method,
                    "variable": var_tag,
                    "input": str(input_paths.get(var_tag, "")),
                    "output": "",
                    "status": f"error: {e}",
                    "elapsed_sec": elapsed,
                })

                summary_df = pd.DataFrame(summary)
                summary_df.to_csv(out_dir / "superres_run_summary_partial.csv", index=False)
                raise

    total_elapsed = time.time() - t0

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "superres_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log("============================================================")
    log("DONE")
    log(f"summary       = {summary_path}")
    log(f"total_elapsed = {total_elapsed:.1f} sec")
    log("============================================================")


if __name__ == "__main__":
    main()

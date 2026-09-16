"""DS2 启发式主干: 近窗流行度 + dst recency 动量, 历史交互/冷候选降权 (严格 t'<t)

调参:   python scripts/heuristic_ds2.py --sample 30000
全量验证: python scripts/heuristic_ds2.py --full
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .heuristic_core import (DAY, EventIndex, candidate_features,
                             hash_tiebreak, mrr_ties, repeat_flags)

# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]
WINDOWS = [30, 60, 90, 180]

DEFAULT_PARAMS = dict(W=90, alpha=0.001, beta=5.0, tau=30, rep_mult=0.01, cold_mult=0.01)


def build_indices(train_csv):
    tr = pd.read_csv(train_csv)
    t = tr['time'].values.astype(np.int64)
    dst_idx = EventIndex(tr['dst'].values, t)
    src_idx = EventIndex(tr['src'].values, t, payload=tr['dst'].values)
    return dst_idx, src_idx


def compute_feats(dst_idx, src_idx, srcs, cands, ts, windows=WINDOWS):
    total, last, recent = candidate_features(dst_idx, cands, ts, windows)
    rep = repeat_flags(src_idx, srcs, cands, ts)
    return dict(total=total, last=last, recent=recent, rep=rep,
                ts=np.asarray(ts, dtype=np.int64), tb=hash_tiebreak(cands))


def score(f, W, alpha, beta, tau, rep_mult, cold_mult):
    gap = f['ts'][:, None].astype(np.float64) - f['last']
    rec = np.where(f['last'] > -(10 ** 17), np.exp(-np.maximum(gap, 0) / (tau * DAY)), 0.0)
    sc = f['recent'][W].astype(np.float64) + alpha * f['total'] + beta * rec
    sc = np.where(f['rep'], sc * rep_mult, sc)
    sc = np.where(f['total'] == 0, sc * cold_mult, sc)
    return sc + f['tb']


def grid_search(f):
    grid_W = WINDOWS
    grid_alpha = [0.0, 0.001, 0.01]
    grid_beta = [0.0, 1.0, 5.0, 20.0]
    grid_tau = [7, 30, 90]
    grid_rep = [0.01]
    grid_cold = [0.01, 0.001]
    results = []
    for W, alpha, beta, tau, rep_m, cold_m in itertools.product(
            grid_W, grid_alpha, grid_beta, grid_tau, grid_rep, grid_cold):
        if beta == 0.0 and tau != grid_tau[0]:
            continue
        m = mrr_ties(score(f, W, alpha, beta, tau, rep_m, cold_m))
        results.append((m, dict(W=W, alpha=alpha, beta=beta, tau=tau,
                                rep_mult=rep_m, cold_mult=cold_m)))
    results.sort(key=lambda x: -x[0])
    return results


def main():
    """网格搜索 DS2 启发式参数（需要预生成的硬负样本缓存）。"""
    ap = argparse.ArgumentParser(description='DS2 启发式参数搜索')
    ap.add_argument('--sample', type=int, default=30000)
    ap.add_argument('--full', action='store_true', help='用已保存参数在全量硬负集验证')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data')
    ap.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs' / 'cache')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_paths = {
        'cands': cache_dir / 'hard_val_dataset2.npy',
        'srcs': cache_dir / 'hard_val_dataset2_src.npy',
        'ts': cache_dir / 'hard_val_dataset2_t.npy',
    }
    train_path = Path(args.data_root) / 'dataset2' / 'train.csv'
    missing = [
        str(path) for path in (*cache_paths.values(), train_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            'missing hard-validation inputs: ' + ', '.join(missing) +
            '; check --data-root / --cache-dir and see data/README.md'
        )
    params_path = Path(args.output_dir) / 'heuristic_params_ds2.json'

    cands = np.load(cache_paths['cands']).astype(np.int64)
    srcs = np.load(cache_paths['srcs']).astype(np.int64)
    ts = np.load(cache_paths['ts']).astype(np.int64)
    print(f'hard_val ds2: {cands.shape}', flush=True)

    t0 = time.time()
    dst_idx, src_idx = build_indices(train_path)
    print(f'index built in {time.time()-t0:.1f}s', flush=True)

    if args.full:
        params = DEFAULT_PARAMS
        if params_path.exists():
            params = json.loads(params_path.read_text())['params']
        W = params['W']
        t0 = time.time()
        f = compute_feats(dst_idx, src_idx, srcs, cands, ts, windows=[W])
        print(f'full feats in {time.time()-t0:.1f}s', flush=True)
        m = mrr_ties(score(f, **params))
        print(f'FULL hard-val MRR ds2 = {m:.4f}  params={params}')
        out = json.loads(params_path.read_text()) if params_path.exists() else {'params': params}
        out['full_mrr'] = m
        params_path.write_text(json.dumps(out, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cands), min(args.sample, len(cands)), replace=False)
    t0 = time.time()
    f = compute_feats(dst_idx, src_idx, srcs[idx], cands[idx], ts[idx])
    print(f'sample feats in {time.time()-t0:.1f}s', flush=True)

    results = grid_search(f)
    print('top 10:')
    for m, p in results[:10]:
        print(f'  {m:.4f}  {p}')
    best_m, best_p = results[0]
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps({'params': best_p, 'sample_mrr': best_m,
                                       'sample': int(args.sample)}, indent=2))
    print(f'best saved to {params_path}: MRR={best_m:.4f}')


if __name__ == '__main__':
    main()

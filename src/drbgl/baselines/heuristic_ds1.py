"""DS1 启发式主干: 无向 pair 频次 + pair recency 衰减 + 节点活跃度兜底 (严格 t'<t)

调参:   python scripts/heuristic_ds1.py --sample 30000
全量验证: python scripts/heuristic_ds1.py --full
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .heuristic_core import (DAY, EventIndex, candidate_features,
                             hash_tiebreak, mrr_ties)

# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]
ACT_WINDOWS = [7, 30, 90]
BASE = 100.0

DEFAULT_PARAMS = dict(w_cnt=1.0, w_rec=20.0, tau=7, Wf=30, eps=0.001)


def build_indices(train_csv):
    tr = pd.read_csv(train_csv)
    s = tr['src'].values.astype(np.int64)
    d = tr['dst'].values.astype(np.int64)
    t = tr['time'].values.astype(np.int64)
    M = int(max(s.max(), d.max())) + 1
    assert M < 2 ** 31, 'pair key would overflow int64'
    # 无向: 两个方向都建
    pair_keys = np.concatenate([s * M + d, d * M + s])
    tt = np.concatenate([t, t])
    pair_idx = EventIndex(pair_keys, tt)
    act_idx = EventIndex(np.concatenate([s, d]), tt)
    return pair_idx, act_idx, M


def compute_feats(pair_idx, act_idx, M, srcs, cands, ts, act_windows=ACT_WINDOWS):
    srcs = np.asarray(srcs, dtype=np.int64)
    cands = np.asarray(cands, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    pk = srcs[:, None] * M + cands
    pcnt, plast, _ = candidate_features(pair_idx, pk, ts, windows=[])
    _, _, act_recent = candidate_features(act_idx, cands, ts, windows=act_windows)
    return dict(pcnt=pcnt, plast=plast, act=act_recent, ts=ts, tb=hash_tiebreak(cands))


def score(f, w_cnt, w_rec, tau, Wf, eps):
    gap = f['ts'][:, None].astype(np.float64) - f['plast']
    rec = np.where(f['plast'] > -(10 ** 17), np.exp(-np.maximum(gap, 0) / (tau * DAY)), 0.0)
    hit = f['pcnt'] > 0
    pair_part = np.where(hit, BASE + w_cnt * f['pcnt'] + w_rec * rec, 0.0)
    return pair_part + eps * np.minimum(f['act'][Wf], 50000.0) + f['tb']


def grid_search(f):
    results = []
    for w_cnt, w_rec, tau, Wf, eps in itertools.product(
            [0.2, 0.5, 1.0, 2.0], [5.0, 20.0, 50.0], [3, 7, 14, 30],
            ACT_WINDOWS, [1e-4, 1e-3]):
        m = mrr_ties(score(f, w_cnt, w_rec, tau, Wf, eps))
        results.append((m, dict(w_cnt=w_cnt, w_rec=w_rec, tau=tau, Wf=Wf, eps=eps)))
    results.sort(key=lambda x: -x[0])
    return results


def main():
    """网格搜索 DS1 启发式参数（需要预生成的硬负样本缓存）。"""
    ap = argparse.ArgumentParser(description='DS1 启发式参数搜索')
    ap.add_argument('--sample', type=int, default=30000)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data')
    ap.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs' / 'cache')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_paths = {
        'cands': cache_dir / 'hard_val_dataset1.npy',
        'srcs': cache_dir / 'hard_val_dataset1_src.npy',
        'ts': cache_dir / 'hard_val_dataset1_t.npy',
    }
    train_path = Path(args.data_root) / 'dataset1' / 'train.csv'
    missing = [
        str(path) for path in (*cache_paths.values(), train_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            'missing hard-validation inputs: ' + ', '.join(missing) +
            '; check --data-root / --cache-dir and see data/README.md'
        )
    params_path = Path(args.output_dir) / 'heuristic_params_ds1.json'

    cands = np.load(cache_paths['cands']).astype(np.int64)
    srcs = np.load(cache_paths['srcs']).astype(np.int64)
    ts = np.load(cache_paths['ts']).astype(np.int64)
    print(f'hard_val ds1: {cands.shape}', flush=True)

    t0 = time.time()
    pair_idx, act_idx, M = build_indices(train_path)
    print(f'index built in {time.time()-t0:.1f}s', flush=True)

    if args.full:
        params = DEFAULT_PARAMS
        if params_path.exists():
            params = json.loads(params_path.read_text())['params']
        t0 = time.time()
        f = compute_feats(pair_idx, act_idx, M, srcs, cands, ts,
                          act_windows=[params['Wf']])
        print(f'full feats in {time.time()-t0:.1f}s', flush=True)
        m = mrr_ties(score(f, **params))
        print(f'FULL hard-val MRR ds1 = {m:.4f}  params={params}')
        out = json.loads(params_path.read_text()) if params_path.exists() else {'params': params}
        out['full_mrr'] = m
        params_path.write_text(json.dumps(out, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cands), min(args.sample, len(cands)), replace=False)
    t0 = time.time()
    f = compute_feats(pair_idx, act_idx, M, srcs[idx], cands[idx], ts[idx])
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

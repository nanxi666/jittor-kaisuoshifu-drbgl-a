"""生成启发式提交 (含 B 榜自适应策略探测器)

用法: python scripts/gen_heuristic_submission.py [--name h1_heuristic] [--datasets dataset1 dataset2]

策略自动探测 (AB 榜算法一致):
  - 对 train 做时间切分 (前90%/后10%), 测 holdout 重复率 (dst 是否在 src 先前历史中)
  - 重复率 > 0.3  -> memory 策略 (pair 频次 + recency, ds1 型)
  - 重复率 ~ 0    -> depop 策略 (近窗流行度 + recency 动量 + 剔重/剔冷 + item-CF, ds2 型)
  - 非二部图 (src/dst 集合重叠) 时 pair 历史按无向处理
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import heuristic_ds1 as H1
from . import heuristic_ds2 as H2
from .heuristic_core import EventIndex

# 仓库根目录（本文件位于 <repo>/src/drbgl/baselines/ 下），仅用作脚本默认路径，
# 实际路径一律由命令行参数或配置文件传入。
ROOT = Path(__file__).resolve().parents[3]
CAND_COLS = [f'c{i}' for i in range(1, 101)]


def detect_strategy(tr):
    s = tr['src'].values.astype(np.int64)
    d = tr['dst'].values.astype(np.int64)
    t = tr['time'].values.astype(np.int64)
    order = np.argsort(t, kind='mergesort')
    s, d = s[order], d[order]
    cut = int(len(s) * 0.9)
    src_set = set(np.unique(s).tolist())
    dst_set = set(np.unique(d).tolist())
    overlap = len(src_set & dst_set) / max(min(len(src_set), len(dst_set)), 1)
    bipartite = overlap < 0.01
    from collections import defaultdict
    hist = defaultdict(set)
    for u, v in zip(s[:cut], d[:cut]):
        hist[u].add(v)
        if not bipartite:
            hist[v].add(u)
    tail_s, tail_d = s[cut:], d[cut:]
    n = min(len(tail_s), 200000)
    rep = np.fromiter((tail_d[i] in hist[tail_s[i]] for i in range(n)), bool, n)
    rate = float(rep.mean())
    strategy = 'memory' if rate > 0.3 else 'depop'
    return strategy, bipartite, rate


def score_memory(tr, te, params, bipartite, cn_cfg=None):
    s = tr['src'].values.astype(np.int64)
    d = tr['dst'].values.astype(np.int64)
    t = tr['time'].values.astype(np.int64)
    M = int(max(s.max(), d.max(), te[CAND_COLS].values.max(), te['src'].values.max())) + 1
    assert M < 2 ** 31
    if bipartite:
        pair_idx = EventIndex(s * M + d, t)
        act_idx = EventIndex(d, t)
    else:
        pair_idx = EventIndex(np.concatenate([s * M + d, d * M + s]), np.concatenate([t, t]))
        act_idx = EventIndex(np.concatenate([s, d]), np.concatenate([t, t]))
    srcs = te['src'].values.astype(np.int64)
    cands = te[CAND_COLS].values.astype(np.int64)
    ts = te['time'].values.astype(np.int64)
    f = H1.compute_feats(pair_idx, act_idx, M, srcs, cands, ts, act_windows=[params['Wf']])
    sc = H1.score(f, **params)
    if cn_cfg:
        cn = common_neighbor_scores(s, d, srcs, cands)
        sc = sc + cn_cfg['gamma'] * np.log1p(cn) * (f['pcnt'] == 0)
    return sc


def common_neighbor_scores(s, d, srcs, cands):
    """无向共同邻居数 |N(src) ∩ N(c)|, 按 src 缓存"""
    nodes = np.unique(np.concatenate([s, d, srcs, cands.ravel()]))
    n = len(nodes)
    ui = np.searchsorted(nodes, s)
    vi = np.searchsorted(nodes, d)
    A = sparse.csr_matrix((np.ones(2 * len(s), np.float32),
                           (np.concatenate([ui, vi]), np.concatenate([vi, ui]))),
                          shape=(n, n))
    A.data[:] = 1.0
    Si = np.searchsorted(nodes, srcs)
    Ci = np.searchsorted(nodes, cands)
    cn = np.zeros(cands.shape, np.float32)
    cache = {}
    for r in range(len(srcs)):
        y = cache.get(Si[r])
        if y is None:
            y = (A[Si[r]] @ A).toarray().ravel()
            cache[Si[r]] = y
        cn[r] = y[Ci[r]]
    return cn


def item_cf_scores(tr, te, K=20, use_iuf=False, item_decay=0):
    """item-item 共现 CF: cf(c) = Σ_{i∈src最近K个item} w_i·|users(i) ∩ users(c)| (去除 src 自身贡献)
    use_iuf: 用户按 1/log2(deg+2) 降权; item_decay: src 物品按位置指数衰减 (越近权重越高)"""
    us = tr['src'].values.astype(np.int64)
    it = tr['dst'].values.astype(np.int64)
    t = tr['time'].values.astype(np.int64)
    uu, ui = np.unique(us, return_inverse=True)
    iu, ii = np.unique(it, return_inverse=True)
    R_csc = sparse.csr_matrix((np.ones(len(tr), np.float32), (ui, ii)),
                              shape=(len(uu), len(iu)))
    R_csc.data[:] = 1.0
    Rt = R_csc.T.tocsr()
    iuf = (1.0 / np.log2(np.asarray(R_csc.sum(1)).ravel() + 2.0)).astype(np.float32)
    R_csc = R_csc.tocsc()
    src_idx = EventIndex(us, t, payload=it)

    srcs = te['src'].values.astype(np.int64)
    cands = te[CAND_COLS].values.astype(np.int64)
    icols = np.searchsorted(iu, np.minimum(cands, iu.max()))
    icols = np.where((iu[np.minimum(icols, len(iu) - 1)] == cands), icols, -1)

    cf = np.zeros(cands.shape, np.float32)
    y_cache = {}
    t0 = time.time()
    for r in range(len(srcs)):
        sid = srcs[r]
        y = y_cache.get(sid)
        if y is None:
            lo, hi = src_idx.locate(sid)
            if hi <= lo:
                y_cache[sid] = np.zeros(0)
                continue
            raw = src_idx.payload[hi - min(K, hi - lo):hi]
            items, first_pos = np.unique(raw, return_index=True)
            if item_decay:
                w_it = np.exp((first_pos - len(raw) + 1) / float(item_decay)).astype(np.float32)
            else:
                w_it = np.ones(len(items), np.float32)
            cols = np.searchsorted(iu, items)
            ok = (cols < len(iu)) & (iu[np.minimum(cols, len(iu) - 1)] == items)
            cols, w_it = cols[ok], w_it[ok]
            if len(cols) == 0:
                y_cache[sid] = np.zeros(0)
                continue
            z = np.asarray(R_csc[:, cols] @ w_it).ravel()
            j = np.searchsorted(uu, sid)
            if j < len(uu) and uu[j] == sid:
                z[j] = 0.0
            if use_iuf:
                z = z * iuf
            y = Rt @ z
            y_cache[sid] = y
        if len(y):
            v = icols[r]
            m = v >= 0
            cf[r, m] = y[v[m]]
        if r % 20000 == 0:
            print(f'  cf row {r}/{len(srcs)} ({time.time()-t0:.0f}s)', flush=True)
    return cf


def score_depop(tr, te, params, cf_cfg):
    t = tr['time'].values.astype(np.int64)
    dst_idx = EventIndex(tr['dst'].values, t)
    src_idx = EventIndex(tr['src'].values, t, payload=tr['dst'].values)
    srcs = te['src'].values.astype(np.int64)
    cands = te[CAND_COLS].values.astype(np.int64)
    ts = te['time'].values.astype(np.int64)
    f = H2.compute_feats(dst_idx, src_idx, srcs, cands, ts, windows=[params['W']])
    sc = H2.score(f, **params)
    if cf_cfg:
        cf = item_cf_scores(tr, te, K=cf_cfg.get('K', 20),
                            use_iuf=cf_cfg.get('use_iuf', False),
                            item_decay=cf_cfg.get('item_decay', 0))
        mult = np.where(f['rep'], params['rep_mult'], 1.0) * \
            np.where(f['total'] == 0, params['cold_mult'], 1.0)
        sc = sc + cf_cfg.get('gamma', 1.0) * np.log1p(cf) * mult
    return sc


def normalize_rows(sc):
    # rank 输出 (值域 0.01..1.00): MRR 只看行内排序, 8 位小数舍入下保证无并列
    from scipy.stats import rankdata
    r = rankdata(sc, method='ordinal', axis=1)
    return r / 100.0


def validate_csv(path, n_rows):
    df = pd.read_csv(path, header=None)
    assert len(df) == n_rows, f'{path}: rows {len(df)} != {n_rows}'
    assert df.shape[1] == 100, f'{path}: cols {df.shape[1]} != 100'
    v = df.values
    assert v.min() >= 0.0 and v.max() <= 1.0, f'{path}: values out of [0,1]'
    print(f'  OK {path.name}: {len(df)} rows x 100 cols, range [{v.min():.4f}, {v.max():.4f}]')


def main():
    """按策略自动生成启发式提交压缩包。"""
    ap = argparse.ArgumentParser(description='启发式提交生成器')
    ap.add_argument('--name', default='h1_heuristic')
    ap.add_argument('--datasets', nargs='+', default=['dataset1', 'dataset2'])
    ap.add_argument('--no-cf', action='store_true')
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / 'heuristic_submission'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = []
    for ds in args.datasets:
        print(f'== {ds} ==', flush=True)
        tr_path = Path(args.data_root) / ds / 'train.csv'
        te_path = Path(args.data_root) / ds / 'test.csv'
        missing = [str(p) for p in (tr_path, te_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f'missing {ds} data: ' + ', '.join(missing) +
                '; check --data-root and see data/README.md'
            )
        tr = pd.read_csv(tr_path)
        te = pd.read_csv(te_path)
        strategy, bipartite, rate = detect_strategy(tr)
        print(f'  detector: repeat_rate={rate:.4f} bipartite={bipartite} -> {strategy}', flush=True)
        pfile = Path(args.output_dir) / f'heuristic_params_{"ds1" if strategy == "memory" else "ds2"}.json'
        if not pfile.is_file():
            raise FileNotFoundError(
                f'missing tuned parameters: {pfile}; '
                f'run drbgl.baselines.heuristic_ds1 / heuristic_ds2 first'
            )
        cfg = json.loads(pfile.read_text())
        params = cfg['params']
        t0 = time.time()
        if strategy == 'memory':
            sc = score_memory(tr, te, params, bipartite, cn_cfg=cfg.get('cn'))
        else:
            cf_cfg = None if args.no_cf else cfg.get('cf')
            sc = score_depop(tr, te, params, cf_cfg)
        print(f'  scored in {time.time()-t0:.0f}s', flush=True)
        prob = normalize_rows(sc)
        path = out_dir / f'{ds}.csv'
        np.savetxt(path, prob, fmt='%.8f', delimiter=',')
        validate_csv(path, len(te))
        csv_paths.append((ds, path))

    zip_path = Path(args.output_dir) / 'submission_candidates' / f'{args.name}.zip'
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ds, p in csv_paths:
            zf.write(p, arcname=f'{ds}.csv')
    print(f'submission written: {zip_path}')


if __name__ == '__main__':
    main()

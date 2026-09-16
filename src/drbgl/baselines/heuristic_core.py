"""共享时序计数结构: 严格 t' < t 的历史特征查询 (heuristic_ds1/ds2 与提交生成复用)"""
import numpy as np

DAY = 86400


class EventIndex:
    """按节点分组的事件时间索引 (CSR), times 组内升序, payload 可携带对端节点"""

    def __init__(self, node_arr, time_arr, payload=None):
        node_arr = np.asarray(node_arr, dtype=np.int64)
        time_arr = np.asarray(time_arr, dtype=np.int64)
        order = np.lexsort((time_arr, node_arr))
        nodes = node_arr[order]
        self.ids, starts = np.unique(nodes, return_index=True)
        self.off = np.append(starts, len(nodes)).astype(np.int64)
        self.times = time_arr[order]
        self.payload = np.asarray(payload)[order] if payload is not None else None

    def locate(self, node):
        j = np.searchsorted(self.ids, node)
        if j < len(self.ids) and self.ids[j] == node:
            return int(self.off[j]), int(self.off[j + 1])
        return 0, 0


def candidate_features(idx, cands, ts, windows):
    """对每个 (行, 候选) 返回: 严格 t'<t 的总次数 total、最近活跃时间 last、各窗口内次数 recent[W]"""
    n, m = cands.shape
    flat_c = cands.ravel().astype(np.int64)
    flat_t = np.repeat(np.asarray(ts, dtype=np.int64), m)
    total = np.zeros(n * m, np.float32)
    last = np.full(n * m, -(10 ** 18), np.int64)
    recent = {w: np.zeros(n * m, np.float32) for w in windows}
    uniq, inv = np.unique(flat_c, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    di = np.searchsorted(idx.ids, uniq)
    di_c = np.minimum(di, max(len(idx.ids) - 1, 0))
    exists = (idx.ids[di_c] == uniq) if len(idx.ids) else np.zeros(len(uniq), bool)
    for k in np.flatnonzero(exists):
        q = order[bounds[k]:bounds[k + 1]]
        a = idx.times[idx.off[di[k]]:idx.off[di[k] + 1]]
        qt = flat_t[q]
        hi = np.searchsorted(a, qt)  # side='left': 严格 < t
        total[q] = hi
        has = hi > 0
        last[q[has]] = a[hi[has] - 1]
        for w in windows:
            lo = np.searchsorted(a, qt - w * DAY)
            recent[w][q] = hi - lo
    return (total.reshape(n, m), last.reshape(n, m),
            {w: recent[w].reshape(n, m) for w in windows})


def repeat_flags(src_idx, srcs, cands, ts):
    """候选是否在 src 的严格历史 (t'<t) 中出现过"""
    n, m = cands.shape
    rep = np.zeros((n, m), bool)
    for i in range(n):
        lo, hi = src_idx.locate(srcs[i])
        if hi > lo:
            k = lo + int(np.searchsorted(src_idx.times[lo:hi], ts[i]))
            if k > lo:
                rep[i] = np.isin(cands[i], src_idx.payload[lo:k])
    return rep


def mrr_ties(scores):
    """列 0 为正样本, ties 取平均排名"""
    pos = scores[:, :1]
    rank = 1 + (scores > pos).sum(1) + 0.5 * ((scores == pos).sum(1) - 1)
    return float(np.mean(1.0 / rank))


def hash_tiebreak(cands):
    """确定性微小扰动, 打破同分 (量级 <1e-7, 不影响真实分差)"""
    h = (cands.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(100003)
    return h.astype(np.float64) / 1e12

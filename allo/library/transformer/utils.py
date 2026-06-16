from collections import defaultdict


def compose_stages(s, stages):
    seen = defaultdict(int)
    for sc, nm in stages:
        k = seen[nm]
        s.compose(sc) if k == 0 else s.compose(sc, id=str(k))
        seen[nm] += 1

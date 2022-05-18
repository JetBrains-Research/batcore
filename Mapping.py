from collections import deque, Counter
from pathlib import Path

import numpy as np

_MISSED_ID = -999


class Mapping:
    def __init__(self, item2id, id2item):
        self.item2id = item2id
        self.id2item = id2item
        self.mask = np.zeros_like(id2item, dtype=np.bool_)

    def set_mask(self, df, col):
        self.mask = np.zeros_like(self.mask)
        self.mask[df[col]] = True

    def transform(self, X, y=None, **kwargs):
        ids = np.asarray([value if self.mask[value] else _MISSED_ID for value in X])
        return np.ma.masked_where(ids == _MISSED_ID, ids, copy=False)


class MappingWithFallback(Mapping):
    def transform(self, X, y=None, **kwargs):
        ids = []
        values = self.id2item[self.mask]
        for value in X:
            if self.mask[value]:
                ids.append(value)
            else:  # fallback
                fallback = self._lc_tokens(self.id2item[value], values)
                if len(fallback):
                    ids.extend([self.item2id[f] for f in fallback
                                if self.item2id[f] not in ids])
                else:
                    ids.append(_MISSED_ID)

        ids = np.array(ids)
        return np.ma.masked_where(ids == _MISSED_ID, ids, copy=False)

    @staticmethod
    def _lc_tokens(target: str, paths: list):
        parts = [Path(path).parts for path in paths]
        parts_target = Path(target).parts

        # longest common, longest prefix, index
        longest_common_tokens = deque([(_MISSED_ID, _MISSED_ID, _MISSED_ID)], maxlen=2)
        for ind, _parts in enumerate(parts):
            len_common_tokens = len(Counter(_parts) & Counter(parts_target))
            prefix = []
            for p1, p2 in zip(parts_target, _parts):
                if p1 != p2:
                    break
                prefix.append(p1)
            len_common_prefix = len(prefix)

            if len_common_tokens > 0 and len_common_tokens >= longest_common_tokens[0][0] \
                    and len_common_prefix >= longest_common_tokens[0][1]:
                longest_common_tokens.appendleft((len_common_tokens, len_common_prefix, ind))

        return [paths[ind] for _, _, ind in longest_common_tokens if ind != _MISSED_ID]

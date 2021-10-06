import numpy as np

class AbstractOnePeriodTriangle:
    _oneptri = np.zeros((1, 1))
    def __init__(self, val=np.zeros(0), ori=np.zeros(0), dev=np.zeros(0), tri_val=np.zeros(0), **kwargs):
        if val.size > 0:
            self._load_mov_data(val, ori, dev, **kwargs)
        elif tri_val.ndim == 2:
            self._oneptri = tri_val
        else:
            self._oneptri = np.zeros((kwargs.get('ori_size', 1), kwargs.get('dev_size', 1)))
    def _load_mov_data(self, val: np.array, ori: np.array, dev: np.array, **kwargs):
        if not (val.size == ori.size == dev.size):
            raise
        max_dev = max(ori + dev)
        ori_size = kwargs.get('ori_size', max(ori) + 1)
        dev_size = kwargs.get('dev_size', max_dev + 1)
        self._oneptri = np.zeros((ori_size, dev_size))
        for v, o, d in zip(val, ori, dev):
            if o > ori_size or o + d + 1 > dev_size:
                continue
            self._oneptri[o, d] += v
    def __repr__(self):
        return str(self._oneptri)

class AbstractTriangle(AbstractOnePeriodTriangle):
    @property
    def values(self):
        return self._internaltri
    @property
    def shape(self):
        return self._internaltri.shape

    def __init__(self, val=np.zeros(0), ori=np.zeros(0), dev=np.zeros(0), tri_val=np.zeros(0),
        origin_per=1, dev_per=1, **kwargs):
        super().__init__(val, ori, dev, tri_val, **kwargs)
        self._construct_triangle(origin_per, dev_per)
    def _construct_triangle(self, origin_per, dev_per):
        one_size = self._oneptri.shape
        origin_size = one_size[0] //  origin_per + min(one_size[0] % origin_per, 1)
        dev_size = one_size[1] //  dev_per + min(one_size[1] % dev_per, 1)
        left_overs = one_size[1] % dev_per
        correction = 1 if left_overs > 0 else 0 # If left_overs we must use it to correct the dev months
        self._internaltri = np.zeros((origin_size, dev_size))
        for i in range(one_size[0]):
            for j in range(one_size[1] - i):
                i_new = i // origin_per
                rel_months = i % origin_per + j
                j_new = (rel_months - left_overs) // dev_per + correction
                self._internaltri[i_new, j_new] += self._oneptri[i, j]
        self._left_overs = left_overs
        self._correction = correction
        self.origin_per = origin_per
        self.dev_per = dev_per

    def _maxcol_index(self, row):
        # n_months = (self.shape[1] - self._correction) * self.dev_per + self._left_overs
        dev_ori_ratio = int(self.origin_per / self.dev_per)
        index = self.shape[1] - row * dev_ori_ratio - 1
        return index

    def __repr__(self):
        return str(self._internaltri)

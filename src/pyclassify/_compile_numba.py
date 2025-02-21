from numba.pycc import CC

cc = CC('_numba_module')
# cc.verbose = True

@cc.export('_distance_numba', 'f8(f8[:], f8[:])')
def _distance_numba(x1, x2):
    dist = 0
    for c1, c2 in zip(x1, x2):
        dist += (c1 - c2) ** 2
    return dist

if __name__ == "__main__":
    cc.compile()

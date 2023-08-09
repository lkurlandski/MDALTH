import sys

sys.path.insert(0, ".")

import numpy as np

from mdalt.pool import Pool


p = Pool(10)
print(p)

p.label(np.array([1, 6]))
print(p)

p.unlabel(np.array([6]))
print(p)

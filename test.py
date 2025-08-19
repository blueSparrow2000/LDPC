import numpy as np

for i in range(10):
    np.random.seed(seed=i) # 6: erronous seed
    c = np.random.choice([0, 1], size=(5,), p=[1. / 2, 1. / 2])
    print(c)









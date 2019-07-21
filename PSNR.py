import numpy as np


def PSNR(origin, prediction):
    c, m, n = origin.shape
    mse = ((origin - prediction)**2).sum() / (c * m * n)
    return 10 * np.log10(1 / mse)


if __name__ == "__main__":
    a = np.ones([3, 100, 100])
    b = np.zeros([3, 100, 100])
    print(PSNR(a, b))

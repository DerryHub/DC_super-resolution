import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def PSNR(origin, prediction):
    c, m, n = origin.shape
    mse = ((origin - prediction)**2).sum() / (c * m * n)
    return 10 * np.log10(1 / mse)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def SSIM(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    ssim = []
    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))
    for i in range(3):
        mu1 = filter2(im1[i], window, 'valid')
        mu2 = filter2(im2[i], window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1[i] * im1[i], window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2[i] * im2[i], window, 'valid') - mu2_sq
        sigmal2 = filter2(im1[i] * im2[i], window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim.append(np.mean(np.mean(ssim_map)))

    return sum(ssim) / len(ssim)


if __name__ == "__main__":
    a = np.ones([3, 100, 100])
    b = np.ones([3, 100, 100])
    print(SSIM(a, b))

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
import skimage.color
import matplotlib.pyplot as plt

class HomomorphicFilter:
    def __init__(self, gH=1.5, gL=0.5):
        self.gH = float(gH)
        self.gL = float(gL)

    def __Duv(self, I_shape):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2) ** (1 / 2)).astype(np.dtype('d'))
        return Duv

    def __butterworth_filter(self, I_shape, filter_params):
        Duv = self.__Duv(I_shape)
        n = filter_params[2]
        c = filter_params[1]
        D0 = filter_params[0]
        h = 1 / (1 + ((c * Duv) / D0) ** (2 * n))
        H = (1 - h)
        return H

    def __gaussian_filter(self, I_shape, filter_params):
        Duv = self.__Duv(I_shape)
        c = filter_params[1]
        D0 = filter_params[0]
        h = np.exp((-c * (Duv ** 2) / (2 * (D0 ** 2))))
        H = (1 - h)
        return H

    def __plot_Filter(self, I, H, filter_params):
        I_shape = I.shape
        if I_shape[0] > I_shape[1]:
            plt.plot(self.__Duv(I_shape)[int(I_shape[1] / 2)], H[int(I_shape[1] / 2)])
        else:
            plt.plot(self.__Duv(I_shape)[int(I_shape[0] / 2)], H[int(I_shape[0] / 2)])

    def __apply_filter(self, I, H, params):
        if self.gH < 1 or self.gL >= 1:
            H = H
        else:
            H = ((self.gH - self.gL) * H + self.gL)
        I_filtered = H * I
        return I_filtered

    def apply_filter(self, I, filter_params=(12, 1, 2), filter_='butterworth', H=None):
        if len(I.shape) != 2:
            raise Exception('image not suitable')
        I_log = np.log1p(np.array(I, dtype='d'))
        I_fft = np.fft.fft2(I_log)
        I_fft = np.fft.fftshift(I_fft)
        if filter_ == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter_ == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter_ == 'external':
            if len(H.shape) != 2:
                raise Exception('Invalid ex filter')
        else:
            raise Exception('filter selected was not applied')
        I_fft_filt = self.__apply_filter(I=I_fft, H=H, params=filter_params)
        I_fft_filt = np.fft.fftshift(I_fft_filt)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.expm1(np.real(I_filt))
        Imax = (np.max(I))
        Imin = (np.min(I))
        I = 255 * ((I - Imin) / (Imax - Imin))
        return I

def display_images(images, titles=None):
    num_images = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(num_images)]
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))
    if num_images == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()




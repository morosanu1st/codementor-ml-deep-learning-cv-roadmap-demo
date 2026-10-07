"""2D convolution + max pooling from scratch, to show how CNN feature maps work.

Run:  python -m src.conv_from_scratch
"""
import numpy as np

# Classic hand-made filters. A CNN *learns* filters like these in its first layer.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)  # vertical edges
SOBEL_Y = SOBEL_X.T                                                     # horizontal edges
BOX_BLUR = np.ones((3, 3)) / 9.0


def conv2d(image, kernel, stride=1, padding=0):
    """Cross-correlation (what deep-learning libraries call 'convolution').

    out_size = (in_size + 2*padding - k) // stride + 1
    """
    if padding:
        image = np.pad(image, padding)
    kh, kw = kernel.shape
    h, w = image.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def relu(x):
    return np.maximum(0, x)


def max_pool2d(fmap, size=2, stride=2):
    h, w = fmap.shape
    oh = (h - size) // stride + 1
    ow = (w - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = fmap[i * stride:i * stride + size, j * stride:j * stride + size].max()
    return out


def conv_layer(image, kernels):
    """One CNN layer: several filters -> stack of feature maps (C, H, W)."""
    return np.stack([relu(conv2d(image, k, padding=1)) for k in kernels])


def make_square_image(size=8):
    img = np.zeros((size, size))
    img[2:6, 2:6] = 1.0
    return img


def show(name, a):
    print(f"\n{name}  shape={a.shape}")
    for row in a:
        print(" ".join(f"{v:5.1f}" for v in row))


if __name__ == "__main__":
    img = make_square_image()
    show("input (white square on black)", img)
    maps = conv_layer(img, [SOBEL_X, SOBEL_Y])
    show("feature map 1: ReLU(Sobel-X) -> fires on LEFT->RIGHT dark-to-bright edges", maps[0])
    show("feature map 2: ReLU(Sobel-Y) -> fires on TOP->BOTTOM dark-to-bright edges", maps[1])
    show("max-pooled feature map 1 (2x2, stride 2): smaller, keeps strongest response", max_pool2d(maps[0]))

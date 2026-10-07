import numpy as np
from src.conv_from_scratch import (
    conv2d, max_pool2d, conv_layer, make_square_image, SOBEL_X, SOBEL_Y, BOX_BLUR,
)


def test_output_shape_formula():
    img = np.zeros((8, 8))
    assert conv2d(img, np.ones((3, 3))).shape == (6, 6)
    assert conv2d(img, np.ones((3, 3)), padding=1).shape == (8, 8)
    assert conv2d(img, np.ones((3, 3)), stride=2, padding=1).shape == (4, 4)


def test_matches_manual_cross_correlation():
    img = np.arange(16, dtype=float).reshape(4, 4)
    k = np.array([[1.0, 0.0], [0.0, -1.0]])
    out = conv2d(img, k)
    expected = img[:-1, :-1] - img[1:, 1:]
    assert np.allclose(out, expected)


def test_blur_on_constant_image_is_constant():
    img = np.full((5, 5), 3.0)
    assert np.allclose(conv2d(img, BOX_BLUR), 3.0)


def test_sobel_detects_edges_only():
    img = make_square_image()
    maps = conv_layer(img, [SOBEL_X, SOBEL_Y])
    assert maps.shape == (2, 8, 8)
    # flat interior and flat background give zero response
    assert maps[0][4, 4] == 0 and maps[0][0, 7] == 0
    # left edge of the square fires the vertical-edge filter
    assert maps[0][3:5, 1:3].max() > 0
    # top edge fires the horizontal-edge filter
    assert maps[1][1:3, 3:5].max() > 0


def test_max_pool():
    fmap = np.array([[1, 2, 0, 0], [3, 4, 0, 5], [0, 0, 7, 0], [6, 0, 0, 8]], dtype=float)
    assert np.array_equal(max_pool2d(fmap), np.array([[4, 5], [6, 8]]))

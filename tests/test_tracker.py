"""pixel_to_box / box_center geometry."""

from src.tracker import box_center, pixel_to_box


def test_box_centered_on_interior_pixel():
    assert pixel_to_box(500, 300, 128, 1920, 1080) == (436, 236, 128, 128)


def test_box_clamped_at_frame_corner():
    x, y, w, h = pixel_to_box(5, 5, 128, 1920, 1080)
    assert (x, y) == (0, 0) and (w, h) == (128, 128)
    x, y, w, h = pixel_to_box(1919, 1079, 128, 1920, 1080)
    assert (x + w, y + h) == (1920, 1080)


def test_box_size_clamped_to_small_frame():
    x, y, w, h = pixel_to_box(50, 50, 128, 100, 80)
    assert (w, h) == (80, 80) and x >= 0 and y >= 0


def test_box_center_roundtrip():
    assert box_center((436, 236, 128, 128)) == (500, 300)

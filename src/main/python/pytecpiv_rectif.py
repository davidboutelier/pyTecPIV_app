def find_control_point(img, x, y, dx, dy):
    import numpy as np
    from skimage import io, img_as_float, img_as_uint
    from skimage.feature import corner_harris, corner_subpix, corner_peaks

    S = img.shape
    y1 = int(y - dy)
    y2 = int(y + dy)

    x1 = int(x - dx)
    x2 = int(x + dx)

    if y1 < 0:
        y1 = 0

    if y2 > S[0]:
        y2 = S[0]

    if x1 < 0:
        x1 = 0

    if x2 > S[1]:
        x2 = S[1]

    cropped = img[y1:y2, x1:x2]
    cropped2 = img_as_float(cropped)
    cropped2 = (cropped2 - min(cropped2.flatten())) / (max(cropped2.flatten()) - min(cropped2.flatten()))
    cropped2[cropped2 >= 0.5] = 1
    cropped2[cropped2 < 0.5] = 0

    coords = corner_peaks(corner_harris(cropped2, method='k', k=0.2, eps=1e-06, sigma=1),
                          min_distance=5, threshold_rel=0, num_peaks=1)
    coords_subpix = corner_subpix(cropped2, coords, window_size=13)

    if len(coords_subpix) == 0:
        xp = x - dx
        yp = y - dy
    else:
        xp = x - dx + coords_subpix[0, 1]
        yp = y - dy + coords_subpix[0, 0]
    return xp, yp


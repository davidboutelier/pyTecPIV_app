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

def correction_error(X_target, Y_target, X_mes, Y_mes):
    import numpy as np

    DX = np.abs(X_target - X_mes)
    DY = np.abs(Y_target - Y_mes)

    mean_ex = np.mean(DX)
    mean_ey = np.mean(DY)

    std_ex = np.std(DX)
    std_ey = np.std(DY)

    s_x = 2 * std_ex
    s_y = 2 * std_ey

    return mean_ex, mean_ey, s_x, s_y

def find_error_proj(img, dst, dx, dy, nx, ny):

    import matplotlib.pyplot as plt
    import numpy as np
    p0 = dst[0, :]
    p1 = dst[1, :]
    p2 = dst[2, :]
    p3 = dst[3, :]

    LX_proj = p1[0] - p0[0]
    LY_proj = p1[1] - p2[1]

    DLX_proj = LX_proj / nx
    DLY_proj = LY_proj / ny

    X_pts_proj = p3[0] + np.linspace(0, nx, nx+1) * DLX_proj
    Y_pts_proj = p3[1] + np.linspace(0, ny, ny+1) * DLY_proj

    X_pts_proj, Y_pts_proj = np.meshgrid(X_pts_proj, Y_pts_proj)
    X_pts_mes = np.zeros(X_pts_proj.shape)
    Y_pts_mes = np.zeros(Y_pts_proj.shape)

    for j in range(0, len(X_pts_proj[0, :])):
        for i in range(0, len(Y_pts_proj[:, 0])):
            x = X_pts_proj[i, j]
            y = Y_pts_proj[i, j]
            dx = DLX_proj / 2
            dy = DLY_proj / 2

            [xp, yp] = find_control_point(img, x, y, dx, dy)
            X_pts_mes[i, j] = xp
            Y_pts_mes[i, j] = yp

    PTS = [X_pts_mes, Y_pts_mes]

    mean_ex, mean_ey, s_x, s_y = correction_error(X_pts_proj, Y_pts_proj, X_pts_mes, Y_pts_mes)

    print(mean_ex, mean_ey, s_x, s_y)




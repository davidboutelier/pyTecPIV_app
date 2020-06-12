def rgb2gray(rgb):
    import numpy as np
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convert_dng(frame_num, file, dir_out):
    import os
    import rawpy
    import numpy as np
    import warnings
    from skimage import io, img_as_uint

    with rawpy.imread(file) as raw:
        rgb = raw.postprocess()
        gray_img = rgb2gray(rgb)
        gray_img_max = np.max(gray_img.flatten())
        gray_img_min = np.min(gray_img.flatten())
        gray_img = (gray_img - gray_img_min) / (gray_img_max - gray_img_min)
        warnings.filterwarnings("ignore", category=UserWarning)
        bit_16_gray_img = img_as_uint(gray_img)
        io.imsave(os.path.join(dir_out, 'IMG_' + str(frame_num + 1).zfill(4) + '.tif'), bit_16_gray_img)
        message = '> ' + file + ' -> ' + os.path.join(dir_out, 'IMG_' + str(frame_num + 1).zfill(4) + '.tif')
        return message
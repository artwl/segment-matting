from pymatting import cutout
import numpy as np
import cv2

def mask_to_trimap(mask, dilation_size=10, erosion_size=10):
    trimap = np.where(mask == 0, 0, 255)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    dilated_trimap = cv2.dilate(trimap.astype(np.uint8), dilation_kernel, iterations=1)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
    final_trimap = cv2.erode(trimap.astype(np.uint8), erosion_kernel, iterations=1)

    # 计算膨胀之后的差异
    diff = dilated_trimap - final_trimap
    # 将膨胀多出来的部分像素值设为128
    final_trimap = final_trimap + np.where(diff > 0, 128, 0)

    return final_trimap

def matting(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    return mask

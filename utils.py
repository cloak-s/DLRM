import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def inpainting(path=None, img=None):
    if path is not None:
        image = cv2.imread(path)
    else:
        image = img
    h_start = 200
    w_start = 400
    image_roi = image[h_start:h_start + 550, w_start:w_start + 650]
    thresh = cv2.inRange(image_roi, np.array([0, 90, 90]), np.array([10, 256, 256]))
    image[h_start:h_start + 550, w_start:w_start + 650] = cv2.inpaint(image_roi, thresh, 3, cv2.INPAINT_TELEA)

    svae_path = os.path.join('static/images', 'inpainted.jpg')
    cv2.imwrite(svae_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def get_advice():
    advice = {}
    advice['ckd'] = '患者需进一步检查病情分期和并发症情况，期间必须持续追踪，并采取规律的生活及运动，' \
                    '不能熬夜、不抽烟饮酒等，且必须控制血糖与血压，饮食采取低盐、低油及低糖策略，' \
                    '必要时先根据医生建议适当使用一些药物来延缓慢性肾病的进程'
    advice['normal'] = '继续保持健康生活习惯，合理饮食，适量运动'
    return advice


if __name__ == '__main__':
    inpainting("test2.jpg")
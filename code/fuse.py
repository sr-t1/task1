import cv2
import numpy as np
def fuse(self, img, face_imgs, face_bboxs):
    '''
    Fuse the repaired face image into the whole image
    :param img: the repaired whole image
    :param face_img: a list that includes all repaired face images
    :return: a final repaired image
    '''

    imgs_bboxs = zip(face_imgs, face_bboxs)
    for imgs_bbox in imgs_bboxs:
        mask = 255 * np.ones(imgs_bbox[0].shape, imgs_bbox[0].dtype)
        center = (imgs_bbox[1][0] + (imgs_bbox[1][3] // 2), imgs_bbox[1][1] + (imgs_bbox[1][2] // 2))
        result = cv2.seamlessClone(imgs_bbox[0], img, mask, center, cv2.MIXED_CLONE)
    return result
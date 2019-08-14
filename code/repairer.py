import my_face_detect
import my_face_repair
import cv2
import numpy as np

class Repair:
    def __init__(self):
        pass

    def repair(self, img):
        '''
        Repair the low-resolution image to high-resolution image
        :param img: low-resolution image, size: H x W x 3
        :return: repair_img: high-resolution image,
        '''

        face_imgs, face_bboxs = self.face_detect(img)

        img_repaired = self.repair_background(img)

        face_img_repaireds = []
        idex = 0
        for face_img in face_imgs:
            idex+=1
            face_img_repaired = self.repair_face(face_img)
            cv2.imwrite('%s.png'%idex, face_img_repaired)
            face_img_repaireds.append(face_img_repaired)

        img_repaired_final = self.fuse(img_repaired, face_img_repaireds, face_bboxs * 4)

        return img_repaired_final

    def face_detect(self, img):
        '''
        Extract the face image from the original image
        :param img: an original low-resolution image
        :return:
            face_imgs: a list that includes the face images (maybe greater than 2)
            face_bbox: a list that includes the the positions of all face images (eg. [ [x1, y1, h1, w1], [x2, y2, h2, w2] ])
        '''
        return my_face_detect.my_face_detect(img)

    def repair_face(self, face_img):
        '''
        Only repair the face image
        :param face_img: an image only include face
        :return: a repaired face image
        '''
        return my_face_repair.repair_face(face_img)

    def repair_background(self, img):
        '''
        Repair the whole low-resolution image
        :param img: the whole low-resolution image
        :return: a repaired whole image
        '''
        pass

    def fuse(self, img, face_imgs, face_bboxs):
        '''
        Fuse the repaired face image into the whole image
        :param img: the repaired whole image
        :param face_img: a list that includes all repaired face images
        :return: a final repaired image
        '''
        pass

def main():
    img = cv2.imread("testset/01.png")
    r = Repair()
    r.repair(img)

if __name__ == '__main__':
    main()

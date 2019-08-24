import cv2
import numpy as np
import torch
import face_repair.architecture as arch

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
            cv2.imwrite('results/face_%s.png'%idex, face_img_repaired)
            face_img_repaireds.append(face_img_repaired)

        img_repaired_final = self.fuse(img_repaired, face_img_repaireds, face_bboxs)

        cv2.imwrite('results/final.png', img_repaired_final)

        return img_repaired_final

    def bb_overlab(self, x1, y1, h1, w1, x2, y2, h2, w2):
        '''
        :return: 两个矩形框如果有交集则返回重合比例, 如果没有交集则返回 0
        '''
        if x1 > x2 + w2:
            return 0
        if y1 > y2 + h2:
            return 0
        if x1 + w1 < x2:
            return 0
        if y1 + h1 < y2:
            return 0
        colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
        rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = colInt * rowInt
        area1 = w1 * h1
        area2 = w2 * h2
        return overlap_area / (area1 + area2 - overlap_area)

    def face_detect(self, img):
        '''
        Extract the face image from the original image
        :param img: an original low-resolution image
        :return:
            face_imgs: a list that includes the face images (maybe greater than 2)
            face_bbox: a list that includes the the positions of all face images (eg. [ [x1, y1, h1, w1], [x2, y2, h2, w2] ])
        '''
        face_imgs = list()
        face_bbox = list()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 正脸：
        face_cascade = cv2.CascadeClassifier('face_detect/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                              minSize=(5, 5))
        for (x, y, w, h) in faces:
            # 扩展bbox一定的scale，使包括头部：高变为原来的1.5倍，宽变为原来的1.4倍
            x = np.max([0, int(x - 0.2 * w)])
            y = np.max([0, int(y - 0.3 * h)])
            w = np.min([int(1.4 * w), img.shape[1] - x])
            h = np.min([int(1.5 * h), img.shape[0] - y])
            face_bbox.append([x, y, h, w])
            face_imgs.append(img[y: y + h, x: x + w])

        # 侧脸：
        face_cascade = cv2.CascadeClassifier('face_detect/haarcascade_profileface.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4,
                                              minSize=(4, 4))
        for (x, y, w, h) in faces:
            # 扩展bbox一定的scale，使包括头部：高变为原来的1.5倍，宽变为原来的1.4倍
            x = np.max([0, int(x - 0.2 * w)])
            y = np.max([0, int(y - 0.3 * h)])
            w = np.min([int(1.4 * w), img.shape[1] - x])
            h = np.min([int(1.5 * h), img.shape[0] - y])
            #计算与已检测到的脸的重合比例
            d = 1
            if face_imgs:
                for face in face_bbox:
                    if self.bb_overlab(x, y, h, w, face[0], face[1], face[2], face[3])>0.4:
                        d = 0
                        break
            #有重合度大于50%的则不保存此人脸
            if d:
                face_bbox.append([x, y, h, w])
                face_imgs.append(img[y: y + h, x: x + w])

        return face_imgs, face_bbox

    def repair_face(self, face_img):
        '''
        Only repair the face image
        :param face_img: an image only include face
        :return: a repaired face image
        '''
        # 去噪
        # img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)
        #读取模型mixed_43600
        model_path = 'face_repair/model.pth'
        device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
        model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                              mode='CNA', res_scale=1, upsample_mode='upconv')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        #生成LR
        face_img = face_img * 1.0 / 255
        face_img = torch.from_numpy(np.transpose(face_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = face_img.unsqueeze(0)
        img_LR = img_LR.to(device)
        #生成SR
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        return output

    def repair_background(self, img):
        '''
        Repair the whole low-resolution image
        :param img: the whole low-resolution image
        :return: a repaired whole image
        '''
        #bicubic上采样
        #height, width = img.shape[:2]
        #return cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)

        # 读取模型nearest_lfw_24000
        model_path = 'face_repair/model_bg.pth'
        device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
        model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                              mode='CNA', res_scale=1, upsample_mode='upconv')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        # 生成LR
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        # 生成SR
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        return output

    def fuse(self, img, face_imgs, face_bboxs):
        '''
        Fuse the repaired face image into the whole image
        :param img: the repaired whole image
        :param face_img: a list that includes all repaired face images
        :return: a final repaired image
        '''
        cv2.imwrite('results/bg.png', img)
        img = cv2.imread('results/bg.png')
        idx = 0
        for face_img,face_bbox in zip(face_imgs, face_bboxs):
            idx+=1
            face_img = cv2.imread('results/face_%s.png'%idx)
            mask = 255 * np.ones(face_img.shape, face_img.dtype)
            center =  (4*face_bbox[0] + 2*face_bbox[3], 4*face_bbox[1] + 2*face_bbox[2])
            #人脸缩小2个像素边缘后直接替换原图
            edge = 2
            img[4 * face_bbox[1] + edge:4 * (face_bbox[1] + face_bbox[2]) - edge, 4 * face_bbox[0] + edge:4 * (face_bbox[0] + face_bbox[3]) - edge] = \
                face_img[edge:face_img.shape[0] - edge, edge:face_img.shape[1] - edge]
            # 人脸边缘的一个像素和原图融合
            img = cv2.seamlessClone(face_img, img, mask, center, cv2.NORMAL_CLONE)
        return img

def main():
    pic = '00'
    img = cv2.imread('testset/%s.png' % pic)
    r = Repair()
    r.repair(img)

if __name__ == '__main__':
    main()

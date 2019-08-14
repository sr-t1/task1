import cv2
import numpy as np

def my_face_detect(img):
	'''
	320*180->1280*720  4x upscale
	test find each face is about 100*100, except the last img with 47 * 47
	
	the last img has two faces while one is not frontal, thus try several ways to detect but only recognize 周星驰
	好了现在大问题，第四张只有侧脸才能检测到那个女的，正脸才能检测到星爷，而且无法同时，而且其它图片能同时被正脸和侧脸检测到。。。重复就重复吧，不是什么大问题
	进一步改进可能两个方向，一个是自己训练个能同时检测正脸和侧脸的模型，还有一个是根据一定阈值判断重叠的bbox，去掉重复的
	
	
	Extract the face image from the original image
	:param img: an original low-resolution image
	:return:
	face_imgs: a list that includes the face images (maybe greater than 2)
	face_bbox: a list that includes the the positions of all face images (eg. [ [x1, y1, h1, w1], [x2, y2, h2, w2] ])
	'''

	face_imgs = list()
	face_bbox = list()
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	#正脸：
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 5, minSize = (5,5))#, flags = cv2.HAAR_SCALE_IMAGE )
	for(x,y,w,h) in faces:
		#扩展bbox一定的scale，使包括头部：长宽各变为原来的1.4倍
		x = np.max([0, int(x - 0.2*w)])
		y = np.max([0, int(y - 0.2*h)])
		w = np.min([int(1.4 * w), img.shape[1]-x])
		h = np.min([int(1.4 * h), img.shape[0]-y])

		face_bbox.append([x, y, h, w])
		face_imgs.append(img[y : y+h, x : x+w])
		
	#侧脸：
	face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 4, minSize = (4,4))#, flags = cv2.HAAR_SCALE_IMAGE )
	for(x,y,w,h) in faces:
		#扩展bbox一定的scale，使包括头部：长变为原来的2倍，宽变为原来的3倍
		x = np.max([0, int(x - 0.3*w)])
		y = np.max([0, int(y - 0.3*h)])
		w = np.min([int(1.6 * w), img.shape[1]-x])
		h = np.min([int(1.6 * h), img.shape[0]-y])

		face_bbox.append([x, y, h, w])
		face_imgs.append(img[y : y+h, x : x+w])

	return face_imgs, face_bbox
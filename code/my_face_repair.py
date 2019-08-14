import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import face_repair.architecture as arch

def repair_face(face_img):
    model_path = 'face_repair/model.pth'
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                          mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    print('Processing...')
    # prepare image
    face_img = face_img * 1.0 / 255
    face_img = torch.from_numpy(np.transpose(face_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = face_img.unsqueeze(0)
    img_LR = img_LR.to(device)
    #generate HR
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output

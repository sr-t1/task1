# task1
sr an img by processing human and background respectively   
# instruction
1.Put the fine-tune face_sr model or any ESRGAN model in code/face_repair/ and rename it to model.pth   
face_sr model download:https://pan.baidu.com/s/1Sl2TvM2dyEp9HPcImHnrjQ   
Put the fine-tune background_sr model or any ESRGAN model in code/face_repair/ and rename it to model_bg.pth   
background_sr model download:https://pan.baidu.com/s/1Sl2TvM2dyEp9HPcImHnrjQ   
   
2.Move the testset inside code/   
   
3.Run repairer.py,   
You can change this code to test different pictures:   
def main():   
    pic = '00'   
    img = cv2.imread('testset/%s.png' % pic)   
    r = Repair()   
    r.repair(img)   
   
The face result picture will be saved in code/results named as face_1.png,face_2.png,face_3.png by the order of faces detected   
The background result will be saved as code/results/bg.png   
The full result will be saved as code/results/final.png   
   
4.Run AI_repairer.py   
You can test pictures with UI   

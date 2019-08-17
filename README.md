# task1
sr an img by processing human and background respectively   
# instruction
1.Put the fine-tune model or any ESRGAN model in code/face_repair/ and rename it to model.pth   
Fine-tune model download:https://pan.baidu.com/s/1Sl2TvM2dyEp9HPcImHnrjQ   
2.Move the testset inside code/   
3.Run repairer.py,   
You can change this code to test different functions or pictures:   
def main():   
    img = cv2.imread("testset/02.png")   
    r = Repair()   
    cv2.imwrite('final.png', r.repair(img))   
4.The face result picture will be saved in code/ named as 1.png,2.png,3.png by the order of faces detected   
The full result will be saved as code/final.png   

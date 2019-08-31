# task1
Sr an img by processing faces and background respectively   
# instruction
1.Put the fine-tune face_sr model or any ESRGAN model in code/face_repair/ and rename it to model.pth   
face_sr model download:https://pan.baidu.com/s/1wSFSs_kDlaosqM5n5VVHhQ#list/path=%2F   
   
2.Move the testset folder inside code/ and create a result folder inside code/   
   
3.Run repairer.py,   
You can change this code to test different pictures:   
def main():   
r = Repair()
    test_img_folder = 'testset/*'   
    for path in glob.glob(test_img_folder):   
        base = os.path.splitext(os.path.basename(path))[0]   
        print(base)   
        img = cv2.imread(path, cv2.IMREAD_COLOR)   
        time_start = time.clock()   
        r.repair(img)   
        print('Generating SR time:%ss' % (time.clock() - time_start))   
   
The face result picture will be saved in code/results named as face_1.png,face_2.png,face_3.png.etc   
The background result will be saved as code/results/bg.png   
The full result will be saved as code/results/final.png   
   
4.Run AI_repairer.py   
You can test pictures with GUI   

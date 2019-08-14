# task1
sr an img by processing human and background respectively
# instruction
1.put your ESRGAN model in code/face_repair/ and rename it to model.pth
2.move the testset inside code/
3.run repairer.py,
you can change this code to test different functions or pictures:
def main():
    img = cv2.imread("testset/01.png")
    r = Repair()
    r.repair(img)
4.the result picture will be saved in code/ named 1.png,2.png,3.png by the order of faces detected

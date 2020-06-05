import sys
import dlib
import cv2
from PIL import Image 
import os 

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

source_dir = "data/cropped_lfw/lfw"
source_dir = "demo"


for r, d, f in os.walk(source_dir):
    for fi in f:
        pic_path = r+"/"+fi
        print(pic_path)
        img = dlib.load_rgb_image(pic_path)

        dets = detector.run(img, 1, 1)
        #print(dets)

        """
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets[0][0])
        dlib.hit_enter_to_continue()
        """

        im_data = cv2.cvtColor(img[dets[0][0].top():dets[0][0].bottom(), dets[0][0].left():dets[0][0].right()],cv2.COLOR_BGR2RGB)


        if type(im_data) != type(None): #if th
            cv2.imwrite(pic_path, im_data)
            

        else:
            im = Image.open(pic_path) 
            left = 83
            top = 92
            right = 166
            bottom = 175
            im1 = im.crop((left, top, right, bottom)) 
            im1.save(pic_path, "JPEG")


        #print(im_data)

        #cv2.imwrite(pic_path, im_data)
    
        #cv2.imwrite(pic_path,  cv2.cvtColor(img[d.top():d.bottom(), d.left():d.right()], cv2.COLOR_BGR2RGB))
        

import cv2
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    original_folder = Path("../images_original/")
    undistorted_folder = Path("../images_undistorted")

    for id in range(1, 856):
        file_name = "{:06d}.png".format(id)
        original_img_name = str(original_folder / file_name)
        
        original_img = cv2.imread(original_img_name)

        ###################################################################################################################
    
        intrinsics = {'f': 1.75468979e+03, 'cx': 2.76974005e+00, 'cy': -4.51774316e+00, \
                    'k1':0.08825813346363495, 'k2': -0.030144145083343458, 'k3': 0.004599722123729709,
                    'width': 3840, 'height': 2160}
        width, height = int(intrinsics['width']), int(intrinsics['height'])
    
    
        K = np.array([[float(intrinsics['f']), 0.0, width/2-float(intrinsics['cx'])],
                [0.0, float(intrinsics['f']), height/2-float(intrinsics['cy'])],
                [0.0, 0.0, 1.0]])
        K_new = K.copy()

        D = np.array([float(intrinsics['k1']), float(intrinsics['k2']), float(intrinsics['k3']), 0.0])

        pad = 0
        K_new[:2, -1] = K_new[:2, -1] + pad
        new_size = (width+2*pad, height+2*pad)

        img_undistort = cv2.fisheye.undistortImage(original_img, K, D=D, Knew=K_new, new_size=new_size)

        undistorted_img_name = str(undistorted_folder / file_name)

        cv2.imwrite(undistorted_img_name, img_undistort)
        print("Saved image: " + undistorted_img_name)


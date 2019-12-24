import matplotlib.pyplot as plt
import numpy as np
import cv2

def Resize(image, width=None, height=None, inter=cv2.INTER_AREA, show=False, image_name='image'):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    if show==True:
        cv2.imshow(image_name, cv2.resize(image, dim, interpolation=inter))
        cv2.waitKey()
        return

    return cv2.resize(image, dim, interpolation=inter)

def resize_multiple(image_list, image_name_list,width=None, height=None, inter=cv2.INTER_AREA, show=False, image_name='image'):
    if len(image_list) != len(image_name_list):
        raise ValueError('Image_list and image_name_list length are not the same!')

    resize_list = []
    for i in image_list:
        resize_list.append(Resize(i, width=width,height=height))
    
    for i in range(len(resize_list)):
        cv2.imshow(str(image_name_list[i]),resize_list[i])
    cv2.waitKey()

    return
        

img = cv2.imread('img.jpg')

img = cv2.blur(img,(10,10))

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

mask = cv2.inRange(h,60,70)

new_h = h+(255-70)
new_h = new_h+((new_h>255)*-255)
new_h = new_h.astype('uint8')

img_list=[img,mask,new_h]
img_name_list = ['image','mask','new_h']
# resize_multiple(img_list,img_name_list,height=600)


class Particle():
     def __init(self,x_dim, y_dim):
          self.position = np.array([int(np.random.rand()*x_dim), int(np.random.rand()*y_dim)])
          self.pbest_position = self.position
          self.pbest_value = 0
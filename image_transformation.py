import cv2
from PIL import Image , ImageFilter,ImageOps
import glob
import numpy as np
from scipy.interpolate import UnivariateSpline
import pixellib
from pixellib.tune_bg import alter_bg
import dlib
from scipy.interpolate import UnivariateSpline
import random

two_Dim_L = [] 
def custom_resize(image,value):
    image = cv2.imread(image, 1)
    (h, w) = image.shape[:2]
    if(w>h): coeff = w/h
    else: coeff = h/w
    new_w = int(value*coeff)
    new_h = int(new_w*coeff)
    cv2.resize(image, (new_w, new_h))
  
def get_coeff(image):
    image = cv2.imread(image, 1)
    (h, w) = image.shape[:2]
    if(w>h): coeff = w/h
    else: coeff = h/w
    return int(coeff)
  
def face_detection(image):
    detector = dlib.get_frontal_face_detector()
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    coeff = get_coeff(image)
    (h,w) = frame.shape[:2]
    fit_x = int(coeff*w/5)
    fit_y = int(coeff*h/5)
    for counter,face in enumerate(faces):
        xx,yx = face.left(),face.top()
        xy,yy = face.right(),face.bottom()
        x1, y1 = xx-fit_x, yx-fit_y
        x2, y2 = xy+fit_x, yy+fit_y
        #t("x1{}-x2{},y1{}-y2{}".format(x1,x2,y1,y2))
        res = save(frame,(x1,y1,x2,y2)) 
        return res
         
def save(img,bbox, width=146,height=250):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    return imgCrop

def white_background(image):
    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("files\deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    change_bg.color_bg(image, colors = (255, 255, 255), output_image_name=image)
    return image

def fragmentation(src_img,rows=3,cols=3):
    im = Image.open(src_img)
    im_w, im_h = im.size
    w = int(im_w/cols)
    h = int(im_h/rows)
    w_num, h_num = int(im_w/w), int(im_h/h)
    
    for wi in range(0, w_num):
        L=[]
        for hi in range(0, h_num):
            box = (wi*w, hi*h, (wi+1)*w, (hi+1)*h)
            piece = im.crop(box)
            tmp_img = Image.new('RGB', (w, h), 255)
            tmp_img.paste(piece)
            img_path = "folder\\{}{}.jpg".format(hi+1,wi+1)
            tmp_img.save(img_path)
            L.append(img_path)
        two_Dim_L.append(L)
def add_border(image):
    img = cv2.imread(image)
    color = [255, 255, 255]
    top, bottom, left, right = [10]*4
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_with_border

def zoom_in_out(image,value):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    scale_percent = 100+value
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)   
    cv2.imwrite(image,resized)
    
def add_shadow(image,shadow):
    bg_img = Image.open(image)
    fg_img = Image.open(shadow)
    resized_fg = fg_img.resize(bg_img.size)
 
    fg_img_trans = Image.new("RGBA",resized_fg.size)
    fg_img_trans = Image.blend(fg_img_trans,resized_fg,.7)

    bg_img.paste(fg_img_trans,(0,0),fg_img_trans)
    bg_img.save(image)
    
def add_effects():
    img22 = cv2.imread("folder\\22.jpg")
    img33 = cv2.imread("folder\\33.jpg")
    img12 = "folder\\12.jpg"
    img23 = "folder\\23.jpg"
    img21 = "folder\\21.jpg"
    img32 = "folder\\32.jpg"
    img34 = "folder\\34.jpg"
    img43 = "folder\\43.jpg"
    img14 = "folder\\14.jpg"
    img13 = "folder\\13.jpg"
    img44 = "folder\\44.jpg"
    img41 = "folder\\44.jpg"
    L = [img12,img21,img23,img32,img34,img43]
    if img22 is not None:
        add_shadow("folder\\22.jpg","files\\cracks.png")

    if img41 is not None:
        add_shadow("folder\\41.jpg","files\\cracked_glass.png")
        add_shadow("folder\\41.jpg","files\\cracked_glass.png")
        add_shadow("folder\\41.jpg","files\\cracked_glass.png")

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def shuffeling(L2D):
    result = []
    mid = L2D[1:-1]
    for m in mid:
        random.shuffle(m)
    random.shuffle(mid)
    result.append(L2D[:1])
    result.append(mid)
    result.append(L2D[-1:])
    return np.array(L2D)

def concat_fragments(im_list_2d):
    L_2D = []
    for row in im_list_2d:
        L=[]
        for elem in row:
            L.append(cv2.imread(elem))
        L_2D.append(L)
    new_L2D = shuffeling(L_2D)
  
    im_list_h = [vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC) for im_list_v in new_L2D]
    return hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC)

def renderV(img_rgb):
    img_rgb = img_rgb
    img_color = img_rgb	
    newImage = img_color.copy()
    i, j, k = img_color.shape
    for x in range(i):
	    for y in range(j):
		    R = img_color[x,y,2] * 0.393 + img_color[x,y,1] * 0.769 + img_color[x,y,0] * 0.189
		    G = img_color[x,y,2] * 0.349 + img_color[x,y,1] * 0.686 + img_color[x,y,0] * 0.168
		    B = img_color[x,y,2] * 0.272 + img_color[x,y,1] * 0.534 + img_color[x,y,0] * 0.131
		    if R > 255:
			    newImage[x,y,2] = 255
		    else:
			    newImage[x,y,2] = R
		    if G > 255:
			    newImage[x,y,1] = 255
		    else:
			    newImage[x,y,1] = G
		    if B > 255:
			    newImage[x,y,0] = 255
		    else:
			    newImage[x,y,0] = B
    return newImage

def VintageFilter(img_path):
    res = renderV(img_path)
    cv2.imwrite("folder\\Vintage_version.jpg", res)
    return res

def add_frame(image):
    img = Image.open(image)
    color = "black"
    border = (320, 320, 320, 320)
    new_img = ImageOps.expand(img, border=border, fill=color)
    new_img.save("folder\\framed.jpg")
    add_shadow("folder\\framed.jpg","frame2.png")
    add_shadow("folder\\framed.jpg","frame2.png")
    add_shadow("folder\\framed.jpg","frame2.png")
    add_shadow("folder\\framed.jpg","frame2.png")

def image_transformation(image,name="person"):
    face = face_detection(image)
    cv2.imwrite("folder\\1_face.jpg",face)
    face_without_bg = white_background("folder\\1_face.jpg")
    fragmentation(face_without_bg,5,4)
    for i in range(1,11):
        print("generating image : {} ({})".format(i,name))
        add_effects()
        concat = concat_fragments(two_Dim_L)
        vintage =cv2.cvtColor(concat,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("generated_images\\{}_image{}.jpg".format(name,i),vintage)
        add_shadow("generated_images\\{}_image{}.jpg".format(name,i),"files\\last_bg2.png")
      
print("Image Transformation Started ...")
image_transformation("adel.jpg",name="adel")
#image_transformation("tom_cruise.jpg",name="tom_cruise")

print("Image Transformation Done ...")




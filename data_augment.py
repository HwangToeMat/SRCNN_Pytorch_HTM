import os
import glob
import h5py
import cv2
from PIL import Image
import numpy as np

def mod_crop(image, scale = 3):
    if len(image.shape) ==3:
        h = image.shape[0]
        w = image.shape[1]
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        return image[0:h,0:w,:]
    else:
        h = image.shape[0]
        w = image.shape[1]
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        return image[0:h,0:w]

def sub_img(input, label, i_size = 33, l_size = 21, stride = 14):
    sub_ipt = []
    sub_lab = []
    pad = abs(i_size-l_size)//2
    for h in range(0, input.shape[0] - i_size + 1, stride):
        for w in range(0, input.shape[1] - i_size + 1, stride):
            sub_i = input[h:h+i_size,w:w+i_size]
            sub_l = label[h + pad :h + pad + l_size,w + pad :w + pad + l_size]
            sub_i = sub_i.reshape(1, i_size,i_size)
            sub_l = sub_l.reshape(1, l_size,l_size)
            sub_ipt.append(sub_i)
            sub_lab.append(sub_l)
    return sub_ipt, sub_lab

def load_img(file_path):
    dir_path = os.path.join(os.getcwd(), file_path)
    img_path = glob.glob(os.path.join(dir_path, '*.bmp'))
    return img_path

def read_img(img_path):
    # read image
    image = cv2.imread(img_path)
    # rgb > ycbcr
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image = image[:, :, 0]
    return image

def img_downsize(img, ds):
    dst = cv2.resize(img, dsize=(0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
    return dst

def zoom_img(img, scale):
    label = img.astype('float') / 255
    temp_input = cv2.resize(label, dsize=(0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
    input = cv2.resize(temp_input, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return input, label

def img_rotate(img, degree):
    height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 90*degree, 1)
    if degree == 1 or degree == 3:
        dst = cv2.warpAffine(img, matrix, (height, width))
    else:
        dst = cv2.warpAffine(img, matrix, (width, height))
    return dst

def save_h5(sub_ip, sub_la, savepath = 'data/train.h5'):
    path = os.path.join(os.getcwd(), savepath)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('input', data=sub_ip)
        hf.create_dataset('label', data=sub_la)

def data_aug(file_path = 'data/Train', savepath = 'data/train.h5', i_size = 33, l_size = 21, stride = 14):
    sub_ip = []
    sub_la = []
    num = 1
    img_path = load_img(file_path)
    for _ in img_path:
        image = read_img(_)
        for degree in [0.,1.,2.,3.]:
            image_r = img_rotate(image, degree)
            for scale in [2,3,4]:
                md_image = mod_crop(image_r, scale)
                input, label = zoom_img(md_image, scale)
                sub_ipt, sub_lab = sub_img(input, label, i_size, l_size, stride)
                sub_ip += sub_ipt
                sub_la += sub_lab
        print('data no.',num)
        num += 1
    sub_ip = np.asarray(sub_ip)
    sub_la = np.asarray(sub_la)
    print('input shape : ',sub_ip.shape)
    print('label shape : ',sub_la.shape)
    save_h5(sub_ip, sub_la, savepath)
    print('---------save---------')

if __name__ == '__main__':
    print('starting data augmentation...')
    data_aug()

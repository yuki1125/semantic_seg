import glob
import os
import random

import numpy as np
from PIL import Image
import six
from chainer.dataset import dataset_mixin
from keras.utils.np_utils import to_categorical

from transforms import random_color_distort

BASE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_ROOT, 'data\\seg_test_images\\seg_test_images')
TRAIN_ANNO_PATH = os.path.join(BASE_ROOT, 'data\\seg_train_anno_without_background\\seg_train_anno_without_background')
TRAIN_PATH = os.path.join(BASE_ROOT, 'data\\seg_train_images\\seg_train_images')

num_labels = [0, # Background
             76, # Person
             29, # Car
             64, # Lane
             225]# Signals

train_file_name = os.listdir(TRAIN_PATH)
test_file_name = os.listdir(TEST_PATH)

val_samples = int(len(train_file_name) * 0.2)

shuffle_train = random.sample(train_file_name, len(train_file_name))
train_out = shuffle_train[val_samples:]
val_out = shuffle_train[:val_samples]

str_train = '\n'.join(train_out)
str_val = '\n'.join(val_out)
str_test = '\n'.join(test_file_name)

with open('train.txt', 'wt') as f:
    f.write(str_train)

with open('val.txt', 'wt') as f:
    f.write(str_val)

with open('test.txt', 'wt') as f:
    f.write(str_test)

def _read_image_as_array(path, dtype, g_scale=False):
    
    opened_image = Image.open(path)

    if g_scale:
        # ラベル画像をグレースケールで読み込む
        opened_image = opened_image.convert('L')

    try:
        image = np.asarray(opened_image)
    finally:
        if hasattr(opened_image, 'close'):
            # hasattr(object, name)
            # nameがobjectの属性を保つ場合True,そうでない場合Falseを返す
            opened_image.close()
    return image

class LabeledImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataset, dtype=np.float32,
                 label_dtype=np.int32, mean=0, crop_size=256, test=True, distort=False, val=False):
        pairs = []
        
        if not val:
            for _, line in enumerate(train_file_name):
                line = line.strip(',')
                if 'jpg' in line:
                    image_filename = line
                    line_label = line.replace('jpg', 'png')
                    label_filename = line_label
                    pairs.append((image_filename, label_filename))
        elif val:
            for _, line in enumerate(val_out):
                line = line.strip(',')
                if 'jpg' in line:
                    image_filename = line
                    line_label = line.replace('jpg', 'png')
                    label_filename = line_label
                    pairs.append((image_filename, label_filename))

        self._pairs = pairs
        self._dtype = dtype
        self._mean = mean
        self._label_dtype = label_dtype
        self._crop_size = crop_size
        self._test = test
        self._distort = distort
    
    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        image_filename, label_filename = self._pairs[i]

        image = _read_image_as_array(TRAIN_PATH + '\\' + image_filename, self._dtype)
        
        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)
    
        image = image / 255.0
        
        label_image = _read_image_as_array(TRAIN_ANNO_PATH + '\\' + label_filename, self._label_dtype, g_scale=True)
        
        for idx, pix_val in enumerate(num_labels):
            label_image = np.where(label_image == pix_val, idx, label_image)

        h, w, _ = image.shape
        
       # label_image = to_categorical(label_image)
        label = label_image
     
        # Randomly flip and crop the image/label for train-set to reduce training time
        if not self._test:

            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]
                label = label[:, ::-1, :]
            
            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
                label = label[::-1, :, :]            

            # Random crop
            top = random.randint(0, h - self._crop_size)
            left = random.randint(0, w - self._crop_size)
            
        else:
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2

        if self._crop_size is 0:
            top = 0
            left = 0
            bottom = h
            right = w
        else:
            bottom = top + self._crop_size
            right = left + self._crop_size

        image = image[top:bottom, left:right]
        label = label[top:bottom, left:right]
        image = np.array(image, dtype='float32')
        label = np.array(label, dtype='int32')
        
        return image.transpose(2, 0, 1), label
    

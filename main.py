#%%
import chainer
from chainer import cuda
from chainercv import evaluations
from chainer import cuda
import cupy
import numpy as np
import PIL


from model_2 import FCNN
import chainer.links as L
cls_label_Red = [0, # Background
                255, # Person
                0, # Car
                69, # Lane
                255]# Signals
cls_label_Green = [0, # Background
                0, # Person
                0, # Car
                47, # Lane
                255]# Signals
cls_label_Blue = [0, # Background
                0, # Person
                255, # Car
                142, # Lane
                0]# Signals              

def evaluate():
    model = FCNN(256, 256, 5)
    model = L.Classifier(model)

    chainer.serializers.load_npz("./result/snapshot_epoch-48", model, path='updater/model:main/')
    
   # img = PIL.Image.open("./data/seg_train_images/seg_train_images/train_0005.jpg")
    img = PIL.Image.open('./data/seg_test_images/seg_test_images/test_191.jpg')
    
    img = img.resize((256, 256))
    img = np.array(img, dtype="float32")
    
    img = img / 255.0
    img = np.reshape(img, (1, 256, 256, 3))
    img = img.transpose(0, 3, 1, 2)
    
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        pred = model.predictor(img)
        pred_trans = chainer.functions.argmax(pred, axis=1)

    return pred_trans

def segmentation(array, cls_labels):
    
    for idx, pix_val in enumerate(cls_labels):
        array = np.where(array == idx, int(pix_val), array) 

    array = np.reshape(array, (256, 256))
    return array

#%%
tmp = evaluate()
print(tmp.shape)
output = tmp.array
#output = cuda.to_cpu(output)
output = np.array(output)
img_r = segmentation(output, cls_label_Red)
img_g = segmentation(output, cls_label_Green)
img_b = segmentation(output, cls_label_Blue)

img_rgb = np.dstack((img_r, img_g))
img_rgb = np.dstack((img_rgb, img_b))

img = PIL.Image.fromarray(np.uint8(img_rgb))
img.save('./img_seg_test191.png')

#%%

import argparse
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions

from model import FCN
from model_2 import FCNN
from dataset import LabeledImageDataset
from mIOU import PixelwiseSigmoidClassifier
from unet import UNet

BASE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_ROOT, 'data\\seg_test_images\\seg_test_images')
TRAIN_ANNO_PATH = os.path.join(BASE_ROOT, 'data\\seg_train_annotations\\seg_train_annotations')
TRAIN_PATH = os.path.join(BASE_ROOT, 'data\\seg_train_images\\seg_train_images')

def create_trainer():
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default=BASE_ROOT,
                         help='Path to directory containing train.txt, val.txt')
    parser.add_argument('--images', default=TRAIN_PATH,
                         help='Root directory of input images')
    parser.add_argument('--labels', default=TRAIN_ANNO_PATH, 
                         help='Root directory of label images')
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--test-batchsize', '-B', type=int, default=4,
                        help='Number of images in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps oever the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='Frequency of taking a snapshot')                        
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result under models directory')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extention')
    parser.add_argument('--tcrop', type=int, default=256,
                        help='Crop size for train-set images')
    parser.add_argument('--vcrop', type=int, default=256,
                        help='Crop size for validation-set images')
    
    args = parser.parse_args()
    
    assert (args.tcrop % 16 == 0) and (args.vcrop % 16 == 0), "tcrop and vcrop must be divisible by 16."

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# Crop-size: {}'.format(args.tcrop))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    # model.pyで定義したモデルを使用
    model = FCNN(out_h=256, out_w=256)
    
    # ピクセルごとに多値分類なので、ロス関数にsoftmax cross entroypを
    # 精度を測る関数としてmean_squared_errorを使用する
    train_model = PixelwiseSigmoidClassifier(model)
    
    # 最適化
    optimizer = optimizers.Adam()
    optimizer.setup(train_model)

    # Load data
    train = LabeledImageDataset(os.path.join(args.dataset, "train.txt"), args.images, args.labels,
                                mean=0, crop_size=args.tcrop, test=True, distort=False)
    val = LabeledImageDataset(os.path.join(args.dataset, "val.txt"), args.images, args.labels,
                                mean=0, crop_size=args.tcrop, test=True, distort=False)
    
    # イテレータ
    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    
    # イテレータからのデータ引き出し、モデルへの受け渡し、損失計算、パラーメタ更新を
    # updaterによって行う
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # Extensionとupdaterをtaraienrに入れる
    trainer = training.trainer.Trainer(updater, (args.epoch, 'epoch'))
    
    logging_attributes = [
        'epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy']
    trainer.extend(extensions.LogReport(logging_attributes))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.PrintReport(logging_attributes))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss']))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.Evaluator(val_iter, optimizer.target, device=args.gpu), name='val')

    return trainer

trainer = create_trainer()
trainer.run()
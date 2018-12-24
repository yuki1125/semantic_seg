import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer import reporter
from chainer import cuda
import cupy
from chainercv import evaluations
from keras.utils.np_utils import to_categorical

num_labels = [0, # Background
             76, # Person
             29, # Car
             64, # Lane
             225]# Signals

class PixelwiseSigmoidClassifier(chainer.Chain):

    def __init__(self, predictor):
        super().__init__()
        with self.init_scope():
            # 学習対象のモデルをpredictorとして保持しておく
            self.predictor = predictor

    def __call__(self, x, t):
        # 学習対象のモデルでまず推論を行う
        y = self.predictor(x)
        
        t = cuda.to_cpu(t)
        
        for idx, pix_val in enumerate(num_labels):
            t = np.where(t == pix_val, int(idx), t)
        
        #t = to_categorical(t)
        
        # 5クラス分類の誤差を計算
        # t = t.transpose(2, 0, 1)
        t = np.array(t, dtype='int32')
        #print('t:', t.shape)
        t = cuda.to_gpu(t)
        #print("y:", y.shape)

        # chainerではsoftmax_cross_entorpyのtとして正解ラベルのint型インデックス番号を与えている
        loss = F.softmax_cross_entropy(y, t)

        # 予測結果（0~1の連続値を持つグレースケール画像）を二値化し，
        # ChainerCVのeval_semantic_segmentation関数に正解ラベルと
        # 共に渡して各種スコアを計算
        #y, t = cuda.to_cpu(F.sigmoid(y).data), cuda.to_cpu(t)
        #y = np.asarray(y > 0.5, dtype=np.int32)
        #y, t = y[:, :, ...], t[:, :, ...]
        #evals = evaluations.eval_semantic_segmentation(y, t)

        # 学習中のログに出力
        reporter.report({'loss': loss},
         #                'miou': evals['miou'],
         #                'pa': evals['pixel_accuracy']}, 
                          self)
        return loss
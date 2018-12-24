import chainer
import chainer.functions as F
import chainer.links as L


class FCN(chainer.Chain):

    def __init__(self, out_h, out_w, n_class=5):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=5, stride=2, pad=2)
            self.conv2 = L.Convolution2D(None, 128, ksize=5, stride=2, pad=2)
            self.conv3 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(None, 128, ksize=1, stride=1, pad=0)
            self.deconv = L.Convolution2D(None, n_class, ksize=32, stride=16, pad=8)
            # 出力サイズは stride*(input_size - 1) + ksize - 2*pad
            # 16*(128-1) + 32 - 2*8 = 2046
        self.out_h = out_h
        self.out_w = out_w


    def forward(self, x):
        h = F.relu(self.conv1(x)) # h: 16, 64, 256, 256 x:16, 3, 256, 256
        h = F.max_pooling_2d(h, 2, 2) #h: 16, 64, 64, 64

        h = F.relu(self.conv2(h)) 
        h = F.max_pooling_2d(h, 2, 2) # h: 16, 128, 16, 16
        
        h = F.relu(self.conv3(h))# h: 16, 128, 16, 16
        h = F.relu(self.conv4(h))
        h = self.conv5(h) # h: 16, 128, 16, 16
        h = self.deconv(h)
        print("h:", h.shape)
        return h.reshape(x.shape[0], 5, h.shape[2], h.shape[3])
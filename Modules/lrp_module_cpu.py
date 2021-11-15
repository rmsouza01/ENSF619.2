import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["CUDA_VISIBLE_DEVICES"] = " "

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model

physical_devices = tf.config.list_physical_devices('GPU')
print('--->Physical devices:'+str(len(physical_devices)))

class LRP:

    def __init__(self, model, param_layers, gamma=0.2, epsilon=0.3, beta=0.0):

        self.model = Model(inputs=[model.input], outputs = [model.layers[-2].output])#drop last layer
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_layers = len(self.model.layers)
        self.beta = beta
        self.alpha = 1.0 + self.beta

        self.pooling_pad = param_layers["pooling_pad"]
        self.conv_pad = param_layers["conv_pad"]
        self.pooling_stride = param_layers["pooling_stride"]
        self.conv_stride = param_layers["conv_stride"]
        self.pooling_ksize = param_layers["pooling_ksize"]
        self.rule =  param_layers["rule"]
        self.type_composite = param_layers["type_composite"]

        for ilayer, layer in enumerate(self.model.layers):
            print("{:3.0f} {:10}".format(ilayer, layer.name))


    def get_model_params(self, x):
        names, activations, weights = [], [], []
        for i in range(self.num_layers):
            names.append(self.model.layers[i].name)
            output_layer = self.model.layers[i].output
            output_model = Model(inputs=[self.model.inputs],
                              outputs=[output_layer])

            output = output_model(x)
            activations.append(output)
            weights.append(self.model.layers[i].get_weights())

        return names, activations, weights

    def LRP_rule_epislon(self, layer, w, b, a, r, epsilon):
        # print(a)
        # print(w)
        #Eq(58) in DOI: 10.1371/journal.pone.0130140
        if layer == 'dense':
            z = tf.tensordot(a,w, axes=1)+b+epsilon
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.tensordot(a,w, axes=1)+b+epsilon
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'conv2d':
            #for vgg padding is same
            z = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad)+b+epsilon #need to recover the strides!
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad)+b+epsilon
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'flatten':
            shape = a.get_shape().as_list()
            shape[0] = -1
            new_r = tf.reshape(r, shape)

        elif layer == 'maxpool2d':

            z  = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+epsilon#ksize and strides from vgg
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+epsilon
                fz = fz*s
            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'input':
            pass

        return new_r

    def LRP_rule_zero(self, layer, w, b, a, r):
        # print(a)
        # print(w)
        #Eq(58) in DOI: 10.1371/journal.pone.0130140
        if layer == 'dense':
            z = tf.tensordot(a,w, axes=1)+b+1e-20
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.tensordot(a,w, axes=1)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'conv2d':
            #for vgg padding is same
            z = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20 #need to recover the strides!
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'flatten':
            shape = a.get_shape().as_list()
            shape[0] = -1
            new_r = tf.reshape(r, shape)

        elif layer == 'maxpool2d':

            z  = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20#ksize and strides from vgg
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20
                fz = fz*s
            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'input':
            pass

        return new_r

    def LRP_gamma_rule(self, layer, w, b, a, r, gamma=1e-20):

        if layer == 'dense':
            #print('w:', w)
            #print('w max:', tf.math.maximum(w, 0.0))
            ww = w+gamma*tf.math.maximum(w, 0.0)
            z = tf.tensordot(a,ww, axes=1)+b+1e-20
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.tensordot(a,ww, axes=1)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'conv2d':
            #for vgg padding is same
            ww = w+gamma*tf.math.maximum(w, 0.0)
            z = tf.nn.conv2d(a, ww, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20 #need to recover the strides!
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.conv2d(a, ww, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'flatten':
            shape = a.get_shape().as_list()
            shape[0] = -1
            new_r = tf.reshape(r, shape)

        elif layer == 'maxpool2d':
            z  = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20#ksize and strides from vgg
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20
                fz = fz*s
            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'input':
            pass

        return new_r

    def LRP_alphabeta_rule(self, layer, w, b, a, r):
        #from paper: Methods for interpreting and understanding deep neural networks
        if layer == 'dense':

            wplus = tf.math.maximum(w, 0.0)
            wminus = tf.math.minimum(w, 0.0)

            z = tf.tensordot(a, wplus, axes=1)+b+1e-20
            s = (self.alpha*r)/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.tensordot(a, wplus, axes=1)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            plus_part =  a * grads

            z = tf.tensordot(a, wminus, axes=1)+b+1e-20
            s = (-self.beta*r)/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.tensordot(a, wminus, axes=1)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            minus_part =  a * grads

            new_r = plus_part + minus_part

        elif layer == 'conv2d':
            #for vgg padding is same
            wplus = tf.math.maximum(w, 0.0)
            wminus = tf.math.minimum(w, 0.0)

            z = tf.nn.conv2d(a, wplus, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
            s = (self.alpha*r)/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.conv2d(a, wplus, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            plus_part =  a * grads

            z = tf.nn.conv2d(a, wminus, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
            s = (-self.beta*r)/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.conv2d(a, wminus, strides=self.conv_stride, padding=self.conv_pad)+b+1e-20
                fz = fz*s

            grads = g.gradient(fz, a)
            minus_part = a * grads

            new_r = plus_part + minus_part

        elif layer == 'flatten':
            shape = a.get_shape().as_list()
            shape[0] = -1
            new_r = tf.reshape(r, shape)

        elif layer == 'maxpool2d':
            z  = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20#ksize and strides from vgg
            s = r/z
            with tf.GradientTape() as g:
                g.watch(a)
                fz = tf.nn.avg_pool(a, ksize = self.pooling_ksize, strides=self.pooling_stride, padding=self.pooling_pad)+1e-20
                fz = fz*s
            grads = g.gradient(fz, a)
            new_r = a * grads

        elif layer == 'input':
            pass

        return new_r

    def find_type_layer(self, layer):
        #print(layer)
        if isinstance(layer, tf.keras.layers.Dense):
            #print('Dense layer')
            return 'dense'
        elif isinstance(layer, tf.keras.layers.Flatten):
            #print('Flatten layer')
            return 'flatten'
        elif isinstance(layer, tf.keras.layers.Conv2D):
            #print('Conv2d layer')
            return 'conv2d'
        elif isinstance(layer, tf.keras.layers.InputLayer):
            #print('input')
            return 'input'
        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.AvgPooling2D):
            return 'maxpool2d'
        else:
            return 'not found'

    def plot_heatmap(self, R, x = None, name_class='', save_path=''):
        R = R.numpy().sum(axis=3)
        R = R / np.max(np.abs(R)) # normalize to [-1,1]
        #coolwarm
        #bwr
        #jet
        cmap = cm.bwr
        #cmap = 'seismic'
        img = R[0]
        norm = Normalize(vmin=-1.0, vmax=1.0)
        mScalar = cm.ScalarMappable(norm=norm, cmap=cmap)
        map_img = mScalar.to_rgba(img)#applying colormap in the image

        if not x is None:
            x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
            if x.shape[3] == 1:
                rgb_img = np.empty((x.shape[1], x.shape[2], 3), dtype=float)
                rgb_img[:,:,0] = x_norm[0,:,:,0]
                rgb_img[:,:,1] = x_norm[0,:,:,0]
                rgb_img[:,:,2] = x_norm[0,:,:,0]
            else:
                rgb_img = x_norm[0]

            #print(np.min(map_img), np.max(map_img))
            plt.figure(figsize=(10,5), dpi=350)
            plt.subplot(1,3,1)
            plt.title('Original Image')
            plt.imshow(rgb_img[...,::-1])#for vgg16
            plt.text(10, map_img.shape[0]-10, name_class, color='red', bbox=dict(fill=False, edgecolor='red', linewidth=2))
            plt.axis('off')

            ax = plt.subplot(1,3,2)
            plt.title('Heatmap')
            plt.imshow(map_img)
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(mScalar, cax=cax)

            plt.subplot(1,3,3)
            plt.title('Image + Heatmap')
            plt.imshow(0.3*rgb_img[...,::-1]+0.7*map_img[:,:,0:3])
            plt.axis('off')
            plt.savefig(save_path)

            plt.figure(figsize=(3,3), dpi=350)
            mask = np.zeros(rgb_img.shape)
            mask[:,:,0] = R>0.05
            mask[:,:,1] = R>0.05
            mask[:,:,2] = R>0.05
            plt.imshow((rgb_img*mask)[...,::-1])
            plt.axis('off')
            plt.savefig('D:\\ML\\LRP\\r0_interp_positive.png')

        else:
            plt.figure(figsize=(3,3), dpi=350)
            plt.imshow(map_img, interpolation='none', cmap=cmap)
            plt.axis('off')
            plt.colorbar(mScalar)
            plt.savefig(save_path)


    def compute_score(self, x):

        print('Getting weights')
        names, activations, weights = self.get_model_params(x)
        print('Computing Score')

        #------one-enconding-------
        ypred = self.model(x)
        r = ypred.numpy()
        index = np.argmax(r)
        new_r = np.zeros(r.shape, dtype=np.float32)
        new_r[0,index] = 1.0
        new_r = new_r*r
        r = tf.convert_to_tensor(new_r)
        #print(r)
        #-------------------------

        R = [r]
        last_layer = []
        for i in range(self.num_layers-1, 1, -1):# last layer is not used
            #print('i:', i)
            last_layer = self.model.layers[i-1]
            type_layer = self.find_type_layer(self.model.layers[i])
            print('Layer:' + names[i])
            if self.rule == 'simple_rule':
                print('Apply basic rule.')
                if type_layer == 'dense' or type_layer == 'conv2d':
                    r = self.LRP_rule_zero(type_layer, weights[i][0], weights[i][1], activations[i-1], r)
                elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                    r = self.LRP_rule_zero(type_layer, None, None, activations[i-1], r)

            elif self.rule == 'epsilon_rule':
                if type_layer == 'dense' or type_layer == 'conv2d':
                    r = self.LRP_rule_epislon(type_layer, weights[i][0], weights[i][1], activations[i-1], r, self.epsilon)
                elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                    r = self.LRP_rule_epislon(type_layer, None, None, activations[i-1], r, self.epsilon)

            elif self.rule == 'gamma_rule':
                if type_layer == 'dense' or type_layer == 'conv2d':
                    r = self.LRP_gamma_rule(type_layer, weights[i][0], weights[i][1], activations[i-1], r, self.gamma)
                elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                    r = self.LRP_gamma_rule(type_layer, None, None, activations[i-1], r, self.gamma)

            elif self.rule == 'composite':
                if i >= self.num_layers-int(self.num_layers*0.2): #apply the simple rule for 20% of the layers
                    print('Apply basic rule.')
                    if type_layer == 'dense' or type_layer == 'conv2d':
                        r = self.LRP_rule_zero(type_layer, weights[i][0], weights[i][1], activations[i-1], r)
                    elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                        r = self.LRP_rule_zero(type_layer, None, None, activations[i-1], r)
                elif i >= self.num_layers-(int(self.num_layers*0.2)+int(self.num_layers*0.4)):
                    print('Apply epsilon rule.')
                    if type_layer == 'dense' or type_layer == 'conv2d':
                        r = self.LRP_rule_epislon(type_layer, weights[i][0], weights[i][1], activations[i-1], r, self.epsilon)
                    elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                        r = self.LRP_rule_epislon(type_layer, None, None, activations[i-1], r, self.epsilon)
                else: #apply gamma rule
                    if self.type_composite == "gamma":
                        print('Apply gamma rule.')
                        if type_layer == 'dense' or type_layer == 'conv2d':
                            r = self.LRP_gamma_rule(type_layer, weights[i][0], weights[i][1], activations[i-1], r, self.gamma)
                        elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                            r = self.LRP_gamma_rule(type_layer, None, None, activations[i-1], r, self.gamma)
                    elif self.type_composite == "alpha_beta":
                        print('Apply alpha-beta rule.')
                        if type_layer == 'dense' or type_layer == 'conv2d':
                            r = self.LRP_alphabeta_rule(type_layer, weights[i][0], weights[i][1], activations[i-1], r)
                        elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                            r = self.LRP_alphabeta_rule(type_layer, None, None, activations[i-1], r)

            elif self.rule == 'alpha_beta':
                print('Apply alpha-beta rule.')
                if type_layer == 'dense' or type_layer == 'conv2d':
                    r = self.LRP_alphabeta_rule(type_layer, weights[i][0], weights[i][1], activations[i-1], r)
                elif type_layer == 'flatten' or type_layer == 'maxpool2d':
                    r = self.LRP_alphabeta_rule(type_layer, None, None, activations[i-1], r)
            else:
                print('Rule not found!')

            R.append(r)

        #input layer
        #the activation nodes are in fact the input
        #z-beta-rule: Layer-Wise Relevance Propagation: An Overview
        print('Input layer rule')
        type_layer = self.find_type_layer(last_layer)
        print('type:', type_layer)
        w = weights[1][0]
        b = weights[1][1]
        a = activations[0] #it is the input x
        r = R[-1]

        wplus = tf.math.maximum(w, 0.0)
        wminus = tf.math.minimum(w, 0.0)

        l = a*0.0+np.min(x)
        h = a*0.0+np.max(x)

        # print(l.shape, h.shape)

        z = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad) \
           -tf.nn.conv2d(l, wplus, strides=self.conv_stride, padding=self.conv_pad) \
           -tf.nn.conv2d(h, wminus, strides=self.conv_stride, padding=self.conv_pad) + 1e-20

        # print('a:', a.shape)
        # print('w', w.shape)
        # print('r', r.shape)
        # print('z', z.shape)

        s =  r/z
        if type_layer == 'conv2d':
            # print('w:',w.shape)
            # #weights[i][0], weights[i][1], activations[i-1]
            # print('wplus', (wplus[0,1,0,0]))
            # print('wplus', wplus)

            with tf.GradientTape(persistent=True) as g:
                g.watch(a)
                g.watch(l)
                g.watch(h)
                fz = tf.nn.conv2d(a, w, strides=self.conv_stride, padding=self.conv_pad) \
                   -tf.nn.conv2d(l, wplus, strides=self.conv_stride, padding=self.conv_pad) \
                   -tf.nn.conv2d(h, wminus, strides=self.conv_stride, padding=self.conv_pad) + 1e-20

                fz = fz*s

            new_r = a * g.gradient(fz, a) + l * g.gradient(fz, l) + h * g.gradient(fz, h)
            #print('new', new_r)


        else:
            pass

        R.append(new_r)


        #self.plot_heatmap(z_conv, None, 'name_class', 'D:\\ML\\LRP\\evolution\\conv.png')


        return R

    def run(self, x, y, name_class = ''):

        #ypred = self.model(x)
        # print('True Class:     ', y)
        # print('Predicted Class:', ypred,'\n')
        print(name_class)
        R = self.compute_score(x)

        print('Checking Conservation Property')
        for i in range(len(R)):
           print('R('+str(len(R)-i-1)+'):' + str(np.sum(R[i].numpy())))
        print('---------------------------------------------------------')
        # for i in range(len(R)-4):
        #     print(i+4)
        #     self.plot_heatmap(R[i+4], None, name_class, 'D:\\ML\\LRP\\evolution\\'+str(i+4)+'.png')
        # return R[-1]
        # print((R[-19].numpy()).shape)
        # print((R[-18].numpy()).shape)
        # print((R[-17].numpy()).shape)
        # print((R[-16].numpy()).shape)
        # print((R[-15].numpy()).shape)
        # print((R[-14].numpy()).shape)
        # print((R[-13].numpy()).shape)
        # print((R[-12].numpy()).shape)
        # print((R[-11].numpy()).shape)
        # print((R[-10].numpy()).shape)
        # self.plot_heatmap(R[-19], None, name_class, 'D:\\ML\\LRP\\evolution\\minus19.png')

        self.plot_heatmap(R[-18], None, name_class, 'D:\\ML\\LRP\\evolution\\minus18.png')

        # self.plot_heatmap(R[-17], None, name_class, 'D:\\ML\\LRP\\evolution\\minus17.png')
        # self.plot_heatmap(R[-16], None, name_class, 'D:\\ML\\LRP\\evolution\\minus16.png')
        # self.plot_heatmap(R[-15], None, name_class, 'D:\\ML\\LRP\\evolution\\minus15.png')

        self.plot_heatmap(R[-14], None, name_class, 'D:\\ML\\LRP\\evolution\\minus14.png')

        # self.plot_heatmap(R[-13], None, name_class, 'D:\\ML\\LRP\\evolution\\minus13.png')
        # self.plot_heatmap(R[-12], None, name_class, 'D:\\ML\\LRP\\evolution\\minus12.png')
        # self.plot_heatmap(R[-11], None, name_class, 'D:\\ML\\LRP\\evolution\\minus11.png')

        self.plot_heatmap(R[-10], None, name_class, 'D:\\ML\\LRP\\evolution\\minus10.png')

        # self.plot_heatmap(R[-9], None, name_class, 'D:\\ML\\LRP\\evolution\\minus9.png')
        # self.plot_heatmap(R[-8], None, name_class, 'D:\\ML\\LRP\\evolution\\minus8.png')
        # self.plot_heatmap(R[-7], None, name_class, 'D:\\ML\\LRP\\evolution\\minus7.png')

        self.plot_heatmap(R[-6], None, name_class, 'D:\\ML\\LRP\\evolution\\minus6.png')

        # self.plot_heatmap(R[-5], None, name_class, 'D:\\ML\\LRP\\evolution\\minus5.png')
        # self.plot_heatmap(R[-4], None, name_class, 'D:\\ML\\LRP\\evolution\\minus4.png')
        # self.plot_heatmap(R[-3], None, name_class, 'D:\\ML\\LRP\\evolution\\minus3.png')

        self.plot_heatmap(R[-2], None, name_class, 'D:\\ML\\LRP\\evolution\\minus2.png')
        self.plot_heatmap(R[-1], None, name_class, 'D:\\ML\\LRP\\evolution\\minus1.png')

        self.plot_heatmap(R[-1], x, name_class, 'D:\\ML\\LRP\\r0_interp.png')

        return R[-1]

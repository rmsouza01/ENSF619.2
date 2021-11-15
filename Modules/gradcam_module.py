import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["CUDA_VISIBLE_DEVICES"] = " "

import tensorflow as tf
from tensorflow.keras.models import Model

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GradCam:

    def __init__(self, model):

        self.model = model

    def plot_heatmap(self, R, x = None, name_class='', save_path=''):
        R = R.numpy()
        #coolwarm
        #bwr
        #jet
        cmap = cm.jet
        #cmap = 'seismic'
        img = R[0]
        if not x is None:
            scale = x.shape[1]/img.shape[-1]
            #print(scale)
            img = zoom(img, scale)
            #print(img.shape)
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        norm = Normalize(vmin=0.0, vmax=1.0)
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
            plt.imshow(0.6*rgb_img[...,::-1]+0.4*map_img[:,:,0:3])
            plt.axis('off')
            plt.savefig(save_path)

            plt.figure(figsize=(3,3), dpi=350)
            mask = np.zeros(rgb_img.shape)
            mask[:,:,0] = img > 0.3
            mask[:,:,1] = img > 0.3
            mask[:,:,2] = img > 0.3
            plt.imshow((rgb_img*mask)[...,::-1])
            plt.axis('off')
            plt.savefig('C:\\Users\\rober\\OneDrive\\Desktop\\mapas\\gradcam_overlap.png')

        else:
            plt.figure(figsize=(3,3), dpi=350)
            plt.imshow(map_img, interpolation='none', cmap=cmap)
            plt.axis('off')
            plt.colorbar(mScalar)
            plt.savefig(save_path)

    def run(self, x, classIdx, lastConvId, name_class = ''):

        self.model_conv = Model(inputs=[self.model.input], \
                                outputs = [self.model.layers[lastConvId].output, self.model.output])


        with tf.GradientTape() as g:
            lastconv, ypred = self.model_conv(x)
            loss = ypred[:, classIdx]

        grads = g.gradient(loss, lastconv)
        alpha  = tf.reduce_mean(grads[0], axis=(0, 1))
        Lcam = alpha*lastconv
        Lcam = tf.reduce_sum(Lcam, axis=-1)
        Lcam = tf.nn.relu(Lcam)

        self.plot_heatmap(Lcam, x, name_class, save_path='C:\\Users\\rober\\OneDrive\\Desktop\\mapas\\grad_map.png')

        return grads

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def remove_blank_slices(img,mask):
    """
    Function to remove blank slices
    """
    img_cropped = img
    mask_cropped = mask
    return img_cropped,mask_cropped

def min_max_normalization(img):
    img_norm = (img - img.min())/(img.max()- img.min())
    return img_norm
    
    
# TensorFLow 
def dice_coef(y_true, y_pred):
    ''' Metric used for CNN training'''
    smooth = 1.0 #CNN dice coefficient smooth
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    ''' Loss function'''
    return -(dice_coef(y_true, y_pred)) # try negative log of the DICE* (-tf.math.log())


def dice_coefficient(seg, ref):
    """
    seg,ref -> boolean array
    """
    dice = 2*((seg*ref).sum())/(seg.sum() + ref.sum())
    return dice

# Jaccard, Haussdorf Distance, TPR, FPR ....

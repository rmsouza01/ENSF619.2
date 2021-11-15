import tensorflow as tf
import numpy as np
# Data generator for domain adversarial neural network
class DataGeneratorDANN(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, source_images, source_labels,  target_images, source_train = True, batch_size = 32, shuffle = True):

        self.source_images = source_images 
        self.source_labels = source_labels
        self.target_images = target_images
        self.batch_size = batch_size
        self.nsamples = source_images.shape[0]
        self.shuffle = shuffle
        self.source_train = source_train
        self.on_epoch_end()
        
    def set_source_train(self,flag):
        self.source_train = flag
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return np.ceil(self.nsamples/self.batch_size).astype(int)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        return self.__data_generation(batch_indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'
    
        Xsource = self.source_images[batch_indexes]
        Ysource = self.source_labels[batch_indexes]
        Xtarget = self.target_images[batch_indexes]
        if self.source_train:
            return Xsource, Ysource
        else:
            return Xsource, Ysource, Xtarget
            
# Model with no domain adaptation
def model_NDA(ishape = (32,32,3)):
    input_layer = tf.keras.layers.Input(ishape)
    x1 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu')(input_layer)
    x2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu')(x1)
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x4 = tf.keras.layers.BatchNormalization()(x3)
    
    x5 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu')(x4)
    x6 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu')(x5)
    x7 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x6)
    x8 = tf.keras.layers.BatchNormalization()(x7)
    # Feature vector
    x9 = tf.keras.layers.Flatten()(x8)
    
    # Label classifier
    out = tf.keras.layers.Dense(10, activation = "softmax")(x9)
    
    model = tf.keras.models.Model(inputs = [input_layer], outputs = [out])
    return model    

#Gradient Reversal Layer
@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return lamda * -dy, None
    
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)

# Domain adversarial neural network implementation
class DANN(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

        #Feature Extractor
        self.feature_extractor_layer0 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu')
        self.feature_extractor_layer1 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu')
        self.feature_extractor_layer2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.feature_extractor_layer3 = tf.keras.layers.BatchNormalization()
        
        
        self.feature_extractor_layer4 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu')
        self.feature_extractor_layer5 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu')
        self.feature_extractor_layer6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.feature_extractor_layer7 = tf.keras.layers.BatchNormalization()
        
        #Label Predictor
        self.label_predictor_layer0 = tf.keras.layers.Dense(10, activation= 'softmax')
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x, train=False, source_train=True, lamda=1.0):
        
        #Feature Extractor
        x = self.feature_extractor_layer0(x)
        x = self.feature_extractor_layer1(x)
        x = self.feature_extractor_layer2(x)
        x = self.feature_extractor_layer3(x , training=train)
        
        x = self.feature_extractor_layer4(x)
        x = self.feature_extractor_layer5(x)
        x = self.feature_extractor_layer6(x)
        x = self.feature_extractor_layer7(x, training=train)
        
        features = tf.keras.layers.Flatten()(x)
        
        
        #Label Predictor
        if source_train is True:
            feature_slice = features
        else:
            feature_slice = tf.slice(features, [0, 0], [features.shape[0] // 2, -1])
        
        #Label Predictor
        l_logits = self.label_predictor_layer0(feature_slice)
        
        #Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dp_x = self.domain_predictor_layer0(features, lamda)    #GradientReversalLayer
            d_logits = self.domain_predictor_layer1(dp_x)
            
            return l_logits, d_logits

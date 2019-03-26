import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from keras.models import load_model
import pandas as pd
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
import matplotlib.pyplot as plt


def Grad_Cam(input_model, x, x1,layer_name):
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X /255.0


    # 予測クラスの算出

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    #  勾配を取得

    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)


    # 画像化してヒートマップにして合成

    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    print(cam)
    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  #モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)   #色をRGBに変換
    print(jetcam.shape)
    print(x1)
    x1=cv2.addWeighted(x1,0.4,jetcam,0.6,0)            #画像を合成

    return x1



input_tensor = Input(shape=(224,224,3))
vgg16 = VGG16(include_top=False, weights=None, input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(6, activation='softmax'))
model = Model(input=vgg16.input, output=top_model(vgg16.output))
model.summary()
model.load_weights("./dataset/sample_weight_ripple.h5")



x = img_to_array(load_img('./dataset/input_motor.bmp', target_size=(224,224)))
x1=cv2.imread('./dataset/input_motor.bmp')
array_to_img(x)

image = Grad_Cam(model,x,x1,'block5_conv3')
array_to_img(image)
plt.imsave('gradcam_input_motor.jpg', image)

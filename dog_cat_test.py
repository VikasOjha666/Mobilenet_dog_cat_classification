


import numpy as np
from keras.preprocessing import image




file_dir='E:/CATS_DOGS/CATS_DOGS/test/DOG/9416.jpg'
dog_img=image.load_img(file_dir,target_size=(150,150))
dog_img=image.img_to_array(dog_img)
dog_img=np.expand_dims(dog_img,axis=0)
dog_img=dog_img/255





#Building the model.
from keras.models import Sequential,Model
from keras.layers import ZeroPadding2D,Conv2D,BatchNormalization,DepthwiseConv2D,Input,Activation,GlobalAveragePooling2D,Reshape,Dropout
from keras import backend as K
from keras import layers
from keras.activations import relu
relu_advanced = lambda x: relu(x, max_value=6.)

def normal_conv_block(inputs,filters,alpha,kernel=(3,3),strides=(1,1)):
    filters=int(filters*alpha)
    x=ZeroPadding2D(padding=((0,1),(0,1)),name='conv1_pad')(inputs)
    x=Conv2D(filters,kernel,padding='valid',use_bias=False,strides=strides,name='conv1')(x)
    x=BatchNormalization(axis=1 if K.image_data_format() == 'channels_first' else -1,name='conv1_bn')(x)
    return Activation(relu_advanced)(x)
def depth_wise_conv_block(inputs, pointwise_conv_filters, alpha,depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x =ZeroPadding2D(((0, 1), (0, 1)),name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=1 if K.image_data_format() == 'channels_first' else -1, name='conv_dw_%d_bn' % block_id)(x)

    x = Activation(relu_advanced)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=1 if K.image_data_format() == 'channels_first' else -1,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu_advanced)(x)


#Main model
def get_model(input_shape=(150,150,3),
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              input_tensor=None,
              pooling=None,
              classes=2,
              ):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    x = normal_conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depth_wise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = depth_wise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), block_id=2)
    x = depth_wise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = depth_wise_conv_block(x, 256, alpha, depth_multiplier,strides=(2, 2), block_id=4)
    x = depth_wise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depth_wise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = depth_wise_conv_block(x, 1024, alpha, depth_multiplier,strides=(2, 2), block_id=12)
    x = depth_wise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = Reshape((classes,), name='reshape_2')(x)
        x = Activation('softmax', name='act_softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return Model(img_input,x)

model=get_model()

def predict_class(model,img):
    arr=model.predict(img)
    return np.argmax(arr)




model.load_weights('Dog_cat_mobilenet_weights.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


print(predict_class(model,dog_img))

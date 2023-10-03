import numpy as np
import keras.utils as image
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras import applications
from keras.models import Model
from keras.layers import Input,concatenate
from packages.spatial_transformer import SpatialTransformer
import matplotlib.pyplot as plt 
import cv2

def locnet():
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((64, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()

    locnet.add(Conv2D(16, (7, 7), padding='valid', input_shape=(64, 64, 3)))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (5, 5), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(64, (3, 3), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))

    locnet.add(Flatten())
    locnet.add(Dense(128, activation='elu'))
    locnet.add(Dense(64, activation='elu'))
    locnet.add(Dense(6, weights=weights))

    return locnet

base_model = applications.VGG16(weights='imagenet', 
                                include_top=False, 
                                input_shape=(64, 64, 3))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

added0_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

stn_model =Sequential()
stn_model.add(Lambda(
    lambda x: 2*x - 1.,
    input_shape=(64, 64, 3),
    output_shape=(64, 64, 3)))
stn_model.add(BatchNormalization())
stn_model.add(SpatialTransformer(localization_net=locnet(),
                                 output_size=(64, 64)))

added_model = Model(inputs=stn_model.input, outputs=added0_model(stn_model.output))

inp = Input(batch_shape=(None, 5))
# out = Dense(8)(inp)
extra_model = Model(inp, inp)

x = concatenate([added_model.output,
           extra_model.output])

# x = Dropout(0.5)(x)
# x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model =  Model([added_model.input,
                extra_model.input],
                x)

model.load_weights('packages\models\Optimized CNN (Extra).hdf5' , by_name = True, skip_mismatch = True)

def path_to_tensor(img_path, shape=(64, 64)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=shape)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)/255
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.vstack([np.expand_dims(x, axis=0)])

def features_preprocessing(age, gender, view):
    agetype = age[-1]
    age = int(age[:-1])
    age = {
              'Y': lambda x: x,
              'M': lambda x: x / 12.,
              'D': lambda x: x / 365.
            }[agetype](age)
    if (gender == 'M'):
        m = 1
        f = 0
    else:
        m = 0
        f = 1
        
    if (view == 'AP'):
        ap = 1
        pa = 0
    else:
        ap = 0
        pa = 1
    return np.array([[age, f, m, ap, pa]])

def predict(img, age, gender, view):
    return model.predict([path_to_tensor(img), features_preprocessing(age, gender, view)])

def show_image(img_path):
    image = cv2.imread(img_path)
    plt.imshow(image)
    plt.show()
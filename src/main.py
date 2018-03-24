import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras import backend as K
import os
K.set_image_data_format("channels_first")
#K.device("/gpu:1")
#K.set_image_dim_ordering('th')
###############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_dir = "./input/"
def load_data(data_dir):
    train = pd.read_json(data_dir+"train.json")
    test = pd.read_json(data_dir+"test.json")
    # Fill 'na' angles with zero
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    test.inc_angle = test.inc_angle.replace('na', 0)
    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)

    return train, test

train, test = load_data(data_dir)

train_pixels_df = pd.read_csv(data_dir + "band1_pixels.csv")
test_pixels_df  = pd.read_csv(data_dir + "band1_pixels_test.csv")

extrastats_train = pd.read_json(data_dir + "train_plus.json")
extrastats_test = pd.read_json(data_dir + "test_plus.json")
num_obj_train = np.concatenate((extrastats_train.band_1_numobj[:,np.newaxis], extrastats_train.band_2_numobj[:,np.newaxis]),axis=1)
num_obj_test = np.concatenate((extrastats_test.band_1_numobj[:,np.newaxis], extrastats_test.band_2_numobj[:,np.newaxis]),axis=1)
#print(train_pixels_df[['numPixel']])
# Put images in an array = [imgNr, pixels, pixels, band], where band is polarization
def process_images(df):
    X_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    X_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    # X_band1_imfs1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_imf_1"]])
    # X_band1_imfs2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_imf_2"]])
    # X_band1_imfs3 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_imf_3"]])
    # X_band1_imfs4 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_imf_4"]])
    # X_band2_imfs1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_imf_1"]])
    # X_band2_imfs2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_imf_2"]])
    # X_band2_imfs3 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_imf_3"]])
    # X_band2_imfs4 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_imf_4"]])
    # Merge bands and add another band as the mean of Band 1 and Band 2 (useful for the ImageDataGenerator later)
    imgs = np.concatenate((X_band1[:, np.newaxis, :, :],
                            # X_band1_imfs1[:, :, :, np.newaxis],
                            # X_band1_imfs2[:, :, :, np.newaxis],
                            X_band2[:, np.newaxis, :, :],
                            # X_band2_imfs1[:, :, :, np.newaxis],
                            # X_band2_imfs2[:, :, :, np.newaxis],
                            ((X_band1+X_band2)/2)[:, np.newaxis, :, :]) ,
                            axis=1)
                            # X_band1_imfs1[:, :, :, np.newaxis]) ,axis=-1)
                            # X_band1_imfs2[:, :, :, np.newaxis],
                            # X_band1_imfs3[:, :, :, np.newaxis],
                            # X_band1_imfs4[:, :, :, np.newaxis],
                            # X_band2_imfs1[:, :, :, np.newaxis]),
                            # X_band2_imfs2[:, :, :, np.newaxis],
                            # X_band2_imfs3[:, :, :, np.newaxis],
                            # X_band2_imfs4[:, :, :, np.newaxis], axis=-1)

    for i in range(len(X_band1)):
        i
    return imgs

a = 2;

X_train = process_images(train)
X_test = process_images(test)

def get_stats(img):
    mean = np.mean(img)
    std = np.std(img)
    max_v = np.max(img)
    min_v = np.min(img)
    median = np.median(img)
    return [mean, std, max_v, median, min_v \
            #(max_v - median), (max_v - min_v), (median - min_v), \
            #((max_v - median) / std), ((max_v - min_v) / std), ((median - min_v) / std) \
            ]

stats_train = [get_stats(X_train[i,0,:,:]) for i in range(len(X_train))]
stats_test = [get_stats(X_test[i,0,:,:]) for i in range(len(X_test))]

X_value_test_temp = np.concatenate((np.array(test.inc_angle)[:,np.newaxis], np.array(test_pixels_df.numPixel)[:,np.newaxis]), axis=1)
X_value_test = np.array([np.concatenate((np.array(stats_test[i]), np.array(X_value_test_temp[i]), num_obj_test[i,:])) for i in range(len(stats_test))])

y_train = np.array(train["is_iceberg"])
X_value_train_temp = np.concatenate((np.array(train.inc_angle)[:,np.newaxis], np.array(train_pixels_df.numPixel)[:,np.newaxis]), axis=1)
X_value_train = np.array([np.concatenate((np.array(stats_train[i]), np.array(X_value_train_temp[i]), num_obj_train[i,:])) for i in range(len(stats_train))])
print(X_value_test.shape)
print(X_value_train.shape)
#X_value_train = X_value_train.reshape(len(y_train),2)

# Create a train and validation split, 75% of data used in training
from sklearn.model_selection import train_test_split
print(X_train.shape)
X_train, X_valid, X_value_train, X_value_valid, y_train, y_valid = train_test_split(X_train,
                                    X_value_train, y_train, random_state=69, train_size=0.75)
print(X_train.shape)





from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

def get_cnn_model_old():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(2, 75, 75)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_siize=(2,2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate, Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, GlobalMaxPooling2D

def cnn_model():
    pic_input = Input(shape=(3, 75, 75))
    ang_input = Input(shape=(9,))
    activation = 'elu'
    cnn = BatchNormalization()(pic_input)
    for i in range(4):
        cnn = Conv2D(8*2**i, kernel_size = (3,3), activation=activation, data_format="channels_first")(cnn)
        cnn = MaxPooling2D((2,2))(cnn) # , dim_ordering='th'


#     cnn = MaxPooling2D(pool_size=(2,2))(cnn)
#     cnn = Flatten()(cnn) #
    cnn = GlobalMaxPooling2D()(cnn)
    cnn = concatenate([cnn,ang_input])
    cnn = Dense(32,activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = Dense(1, activation = 'sigmoid')(cnn)

    cnn_model = Model(inputs=[pic_input,ang_input],outputs=cnn)

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_model
batch_size=8
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         rotation_range=5)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

# Finally create generator
gen_flow = gen_flow_for_two_inputs(X_train, X_value_train, y_train)

print(X_train.shape)
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
model = cnn_model()

weight_path = "weights"
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=50, min_lr=0.001)

early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, min_delta=1e-4, mode='min')
model_chk = ModelCheckpoint(monitor='val_loss', filepath=weight_path, save_best_only=True, save_weights_only=True, mode='min')
callbacks = [early_stop, model_chk, reduce_lr]

model.fit_generator(gen_flow, validation_data=([X_valid, X_value_valid], y_valid),
                    steps_per_epoch=len(X_train) / batch_size, epochs=400, callbacks=callbacks)
# serialize weights to HDF5
model.save_weights("model.h5")
print("evaluate after run")
scores = model.evaluate([X_valid, X_value_valid], y_valid, batch_size=8)
for elem in scores:
    print(elem)
# print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))
# Predict on test data

#weight_path="WithNumPixels17_47"
#weight_path="WithNumPixels1779valid_1792train"
#weight_path="weights_now"
model.load_weights(weight_path)
scores = model.evaluate([X_valid, X_value_valid], y_valid, batch_size=8)
for elem in scores:
    print(elem)
# print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))

test_predictions = model.predict([X_test,X_value_test])
print("hello")
# Create .csv
pred_df = test[['id']].copy()
pred_df['is_iceberg'] = test_predictions
pred_df.to_csv('predictions3.csv', index = False)
pred_df.sample(3)

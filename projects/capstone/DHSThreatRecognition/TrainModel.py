import HelperFuncs as hfuncs
import numpy as np

BATCH_SIZE = 20
FINAL_WIDTH = 400
FINAL_HEIGHT = 600
CHANNELS = 1
ZONES = 17

#Define a generator function
def myGenerator():
    #AWS and Directory information 
    bucketName = 'miscdatastorage'
    dataDir = 'DHSData/'
    temp_dir = 'temp'
    labels_dir = r'stage1_labels.csv'
    #Connect to AWS
    key_id, secret_key = hfuncs.GetAWSCredentials()
    client = hfuncs.GetAWSClient(key_id,secret_key)
    bucket = client.Bucket(bucketName)
    #Initialize required parameters
    key_ary = hfuncs.GetShuffledKeys(bucket)
    labels_dict = hfuncs.GetLabelsDict(labels_dir)
    extension = '.a3daps'
    
    #Initialize AWS Batch Requester
    batchrequester = hfuncs.BatchRequester(bucket,key_ary,labels_dict,dataDir,temp_dir,extension)
    
    #Preprocessing parameters
    n_samples = 10 #Distinct samples (x64 images each) to retrieve iteratively
    angles = 64
    
    
    #While there is data left, yield batch
    while batchrequester.DoItemsRemain():
        X,y = batchrequester.NextBatch(n_samples)
        
      #  if X.shape[0] < n_samples:
      #     return
        #Set counter to 0, channel to 1, and initialize output arrays
        i = 0
        chan = 0 #No need to iterate here
        X_train = np.zeros((n_samples*angles,FINAL_WIDTH,FINAL_HEIGHT,CHANNELS))
        y_train = np.zeros((n_samples*angles,ZONES))
        
        #Clean each image and store it in output array
        for s in range(X.shape[0]):
            for a in range(X.shape[3]):
                X_train[i,:,:,chan] = hfuncs.CropCleanResize(X[s,:,:,a],FINAL_WIDTH,FINAL_HEIGHT)
                y_train[i,:] =  y[s,:]
        i = 0
        while i < n_samples * angles:
            yield X_train[i:i+BATCH_SIZE,:,:,:],y_train[i:i+BATCH_SIZE]
            i += BATCH_SIZE

from keras.layers import Input, Dense, Conv2D, MaxPooling2D , AveragePooling2D,Flatten
from keras.models import Model
from keras.layers.core import Dropout
import keras

#Build Basic model

input_img = Input(shape=(FINAL_WIDTH,FINAL_HEIGHT,CHANNELS))

pooling_1 = MaxPooling2D((2,2),padding='same')(input_img)

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(pooling_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(pooling_1)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(pooling_1)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output_inception = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

pooling_2 = MaxPooling2D((3,2),padding='same')(output_inception)
pooling_2 = Dropout(0.10)(pooling_2)

tower_1_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(pooling_2)
tower_1_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(tower_1_2)

tower_2_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(pooling_2)
tower_2_2 = Conv2D(128, (5, 5), padding='same', activation='relu')(tower_2_2)

tower_3_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(pooling_2)
tower_3_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(tower_3_2)

output_inception_2 = keras.layers.concatenate([tower_1_2, tower_2_2, tower_3_2], axis=1)

output_inception_2 = Dropout(0.10)(output_inception_2)
output_inception_2 = MaxPooling2D((2, 1),strides=(2,1), padding='same')(output_inception_2)

conv_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(output_inception_2)
last = Flatten()(conv_3)

#List of independent guesses for each zone
output_nodes = []
for i in range(ZONES):
    output_nodes.append(Dense(1,activation='sigmoid')(last))

out = keras.layers.concatenate(output_nodes)

multi_label_model = Model(input_img, out)
   
from datetime import datetime
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import metrics

x = datetime.today()
stamp = "{}-{}-{}_{}:{}:{}".format(x.year,x.month,x.day,x.hour,x.minute,x.second)
tensorboard = TensorBoard(log_dir="logs/{}".format(stamp))


multi_label_model.compile(optimizer='SGD',
                          metrics=[metrics.binary_accuracy,metrics.binary_crossentropy],
                         loss= 'binary_crossentropy')
gen = myGenerator()
multi_label_model.fit_generator(gen,steps_per_epoch=10,epochs=5,callbacks=[tensorboard])
     
        



    
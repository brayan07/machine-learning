import HelperFuncs as hfuncs
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.data_utils import Sequence
import h5py
import os
import pickle
import logging

logging.basicConfig(filename='UploadScans.log',level=logging.INFO)

BATCH_SIZE = 20
FINAL_WIDTH = 400
FINAL_HEIGHT = 600
CHANNELS = 1
ZONES = 17
BUCKET_NAME = 'miscdatastorage'
DATA_DIR = 'DHSData/'
TEMP_DIR = 'temp'
LABELS_DIR = r'stage1_labels.csv'
EXTENSION = '.a3daps'
np.random.seed(0)

#Define a generator function
class myScanGenerator:
    #AWS and Directory information 
    bucketName = BUCKET_NAME
    dataDir = DATA_DIR
    temp_dir = TEMP_DIR
    labels_dir = LABELS_DIR
    #Connect to AWS
    key_id, secret_key = hfuncs.GetAWSCredentials()
    client = hfuncs.GetAWSClient(key_id,secret_key)
    bucket = client.Bucket(bucketName)
    extension = EXTENSION
    #labels and keys
    labels_dict = hfuncs.GetLabelsDict(labels_dir)
    key_ary = None
    #Batch information
    n_samples = 0
    batch_size = 0
    #Requester
    batch_requester = None
    #Initialize required parameters
    def __init__(self,keys,n_samples):
        #Keys of samples to process
        self.key_ary = keys
        #Samples to load at a time
        self.n_samples = n_samples
        #Initialize AWS Batch Requester
        self.batchrequester = hfuncs.BatchRequester(self.bucket,self.key_ary,self.labels_dict,self.dataDir,self.temp_dir,self.extension)
    def GenerateSamples(self):
        '''Returns generator that retireves n_sample scans at a time,
        mixes each scan-slice image into a meta-batch, and returns mini-batches of 
        BATCH_SIZE'''
        #While there is data left, yield batch
        while self.batchrequester.DoItemsRemain():
            #Request data
            print("Retrieving data..")
            pointer = self.batchrequester.key_pointer
            logging.info("Last succesful key {} at pointer {}".format(self.batchrequester.keys[pointer],pointer))
            X,y = self.batchrequester.NextBatch(self.n_samples)
            n_angles = X.shape[3] #num angles (64)

            print("Data retrieved")

            #Initialize output arrays
            print("Initializing arrays...")
            X_train = np.zeros((X.shape[0],n_angles,FINAL_WIDTH,FINAL_HEIGHT,CHANNELS))
            y_train = np.zeros((X.shape[0],ZONES))
            print("Arrays initialized")

            #Set counter to 0, channel to 1
            chan = 0 #No need to iterate here
            i = 0
            #Clean each image and store it in output
            for i in range(X.shape[0]):
                for j in range(n_angles):
                    X_train[i,j,:,:,chan] = hfuncs.CropCleanResize(X[i,:,:,j],FINAL_WIDTH,FINAL_HEIGHT)
                    y_train[i,:] = y[i,:]
                    
                yield X_train[i,:,:,:,:],y_train[i]
                
def CleanKeyAry(key_ary,labels_dict,dataDir,extension):
    '''Taken from the BatchRequester class'''
    key_ary_new=[]
    for key in key_ary:
        img_id = key.strip().replace(dataDir,'').replace(extension,'')
        if img_id in labels_dict.keys():
            key_ary_new.append(key)
        else:
            continue
    return key_ary_new

def getTrainTestValData(labels_dir=LABELS_DIR,extension=EXTENSION,dataDir=DATA_DIR,bucketName=BUCKET_NAME):
    '''Retrieves all samples that have corresponding labels 
    and splits data into a train, test, val set. '''
    #Labels        
    labels_dict = hfuncs.GetLabelsDict(labels_dir)
    
    #AWS Bucket
    key_id, secret_key = hfuncs.GetAWSCredentials()
    client = hfuncs.GetAWSClient(key_id,secret_key)
    bucket = client.Bucket(bucketName)
    
    #Get shuffled keys and separate into train,test,and validation
    key_ary = hfuncs.GetShuffledKeys(bucket)
    key_ary = CleanKeyAry(key_ary,labels_dict,dataDir,extension)
    K_train,K_test = train_test_split(key_ary,test_size=0.20,random_state=0)
    K_train,K_val = train_test_split(K_train,test_size=0.25,random_state=0) #0.80*0.25 = 0.20 validation 
    
    return K_train,K_test,K_val                 
#Load train, test, and val sets

#Load data from pickled file
with open("data_separated.pickle","rb") as f:
    save = pickle.load(f)
    K_train = save['K_train']
    K_val = save['K_val']
    K_test = save['K_test']

#Last time accidentatlly put K_train into key K_test.
#Now we must manually create K_test    
#key_id, secret_key = hfuncs.GetAWSCredentials()
#client = hfuncs.GetAWSClient(key_id,secret_key)
#bucket = client.Bucket(BUCKET_NAME) 

#Get all keys and create set of those neither in K_train nor K_val
#labels_dir = LABELS_DIR
#labels_dict = hfuncs.GetLabelsDict(labels_dir)
#dataDir = DATA_DIR
#extension = EXTENSION
#key_ary = hfuncs.GetShuffledKeys(bucket) 
#print("Got all keys")
#key_ary = CleanKeyAry(key_ary,labels_dict,dataDir,extension)
#print("Cleaned keys")
#print("Creating test set again..")
#K_test = [k for k in key_ary if ((k not in K_train) and (k not in K_val))]

#Save data split again
#save = {'K_train':K_train,'K_test':K_test,'K_val':K_val}

#with open("data_separated.pickle","wb") as f:
#    pickle.dump(save,f)

#Connect to aws s3
UPLOAD_BUCKET = 'cleandhsdata'
key_id, secret_key = hfuncs.GetAWSCredentials()
client = hfuncs.GetAWSClient(key_id,secret_key)
bucket = client.Bucket(UPLOAD_BUCKET)

#Clean and upload
logging.info("Starting train upload")
key_root = "train_scan"
trainGen = myScanGenerator(K_train,5)
i = 0
for X, y in trainGen.GenerateSamples():
    filename = os.path.join(TEMP_DIR,"batch_{}.hdf5".format(i))
    key = "{}/{}".format(key_root,"batch_{}.hdf5".format(i))
    with h5py.File(filename,"w") as f:
        dset = f.create_dataset('image',data=X)
        dset2 = f.create_dataset('labels',data=y)
    bucket.upload_file(Filename=filename,Key=key)
    os.remove(filename)
    i += 1
    logging.info("Completed batch {}".format(i))
    print("Completed batch {}".format(i))

#Clean and upload
logging.info("Starting val upload")
key_root = "val_scan"
trainGen = myScanGenerator(K_val,5)
i = 0
for X, y in trainGen.GenerateSamples():
    filename = os.path.join(TEMP_DIR,"batch_{}.hdf5".format(i))
    key = "{}/{}".format(key_root,"batch_{}.hdf5".format(i))
    with h5py.File(filename,"w") as f:
        dset = f.create_dataset('image',data=X)
        dset2 = f.create_dataset('labels',data=y)
    bucket.upload_file(Filename=filename,Key=key)
    os.remove(filename)
    i += 1
    logging.info("Completed batch {}".format(i))
    print("Completed batch {}".format(i))
    
#Clean and upload
logging.info("Starting test upload")
key_root = "test_scan"
trainGen = myScanGenerator(K_test,5)
i = 0
for X, y in trainGen.GenerateSamples():
    filename = os.path.join(TEMP_DIR,"batch_{}.hdf5".format(i))
    key = "{}/{}".format(key_root,"batch_{}.hdf5".format(i))
    with h5py.File(filename,"w") as f:
        dset = f.create_dataset('image',data=X)
        dset2 = f.create_dataset('labels',data=y)
    bucket.upload_file(Filename=filename,Key=key)
    os.remove(filename)
    i += 1
    logging.info("Completed batch {}".format(i))
    print("Completed batch {}".format(i))
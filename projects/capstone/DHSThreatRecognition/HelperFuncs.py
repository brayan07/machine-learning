import IOFunctions as iof
import pickle
import boto3
import configparser
import numpy as np
import pandas as pd
import os
import cv2
import time


# Check to make sure setup_file has been run
DEMO_CONF_FILE = "isDemo.conf"
assert os.path.exists(DEMO_CONF_FILE)

# Check to see whther we should run in demo mode
config = configparser.RawConfigParser()
config.read(DEMO_CONF_FILE)
IS_DEMO = config.getboolean('DEFAULT','isDemo')

# Define important environment variables
LABELS_FILE = 'stage1_labels.csv'
AWS_CONFIG_FILE = 'S3.conf'
RAW_DATA_BUCKET = 'miscdatastorage'
RAW_DATA_DIRECTORY = 'DHSData/'
CLEAN_DATA_BUCKET = 'cleandhsdata'
CLEAN_DATA_DIR = 'fullfeatureextraction'
TEMP_DIR = 'temp/'
EXTENSION = '.a3daps'
NOISE_THRESHOLD = 7000
FINAL_WIDTH = 400
FINAL_HEIGHT = 600
CHANNELS = 1
ZONES = 17
ANGLES = 16

# Set variables depending on value of isDemo
if IS_DEMO:
    print("Setting demo environment variables...")
    CLEAN_DATA_BUCKET = "democleandhsdata"
    AWS_CONFIG_FILE = "DemoS3.conf"
    
# Create temporary directory if it doesn't exist
if not os.path.isdir(TEMP_DIR):
    print("Creating temp directory")
    os.mkdir(TEMP_DIR)


def ReduceNoise(x, thresh=NOISE_THRESHOLD):
    if x < thresh:
        x = thresh
    return x

def FindCropDimensions(x,thresh=NOISE_THRESHOLD):
    ''' Expects a 2d image ndarray. Returns the minimum row and column along each dimension
    with a value greater than the noise threshold.'''
    min_i = x.shape[0]-1
    max_i = 0 
    max_j = 0
    RELEVANT_THRESHOLD = .50
    CONSEC_THRESHOLD = 35
    #Loop through columns
    for j in range(x.shape[1]):
        num_relevant = 0
        max_consec = 0
        num_consec = 0
        for i in range(x.shape[0]):
            if x[i,j] > thresh:
                num_relevant += 1
                num_consec += 1
            else:
                if num_consec > max_consec:
                    max_consec = num_consec
                num_consec = 0
        if j > max_j and (num_relevant/x.shape[0] > RELEVANT_THRESHOLD or max_consec > CONSEC_THRESHOLD):
            max_j = j
    #Loop through rows        
    for i in range(x.shape[0]):
        num_relevant = 0
        max_consec = 0
        num_consec = 0
        for j in range(x.shape[1]):
            if x[i,j] > thresh:
                num_relevant += 1
                num_consec += 1
            else:
                if num_consec > max_consec:
                    max_consec = num_consec
                num_consec = 0
        if i > max_i and (num_relevant/x.shape[1] > RELEVANT_THRESHOLD or max_consec > CONSEC_THRESHOLD):
            max_i = i
        if i < min_i and (num_relevant/x.shape[1] > RELEVANT_THRESHOLD or max_consec > CONSEC_THRESHOLD):
            min_i = i
    
    if min_i > max_i:#Image with no signal
        temp = min_i
        min_i = max_i
        max_i = temp
    
    return min_i,max_i,max_j
    
# Cropping and resizing functions
def CropImage(x,min_i,max_i,max_j,new_i,new_j):
    '''Returns regions of interest in the image'''
    resize_i = False
    resize_j = False
    
    #Width
    old_i = max_i-min_i+1
    #If crop is of smaller width than new_i
    if old_i <= new_i:
        #Calculate total difference and buffer assuming it is evenly split
        diff = new_i-old_i
        buffer,remainder = divmod(diff,2)
        #Check which side has a bigger margin
        margin_i_min = min_i
        margin_i_max = x.shape[0]-max_i-1
        if margin_i_min <= margin_i_max:
            if margin_i_min > buffer:
                min_i = min_i - buffer
            else:
                remainder += buffer - margin_i_min
                min_i = 0
            max_i += buffer + remainder
        else:
            if margin_i_max > buffer:
                max_i = max_i + buffer
            else:
                remainder += buffer - margin_i_max
                max_i = x.shape[0]-1
            min_i -= buffer + remainder
    else:
        #Image needs to be resized along this dimension
        resize_i = True
    
    #Height
    old_j = max_j+1
    #If crop is of smaller height than new_j
    if old_j <= new_j:
        diff = new_j - old_j
        #Move max_j
        max_j += diff
    else:
        #Image needs to be resized
        resize_j = True
        
    #Resize along either dimension?
    if resize_i or resize_j:
        return cv2.resize(x[min_i:max_i+1,:max_j+1],dsize=(new_j,new_i))
    else:
        return x[min_i:max_i+1,:max_j+1]
    
def CropCleanResize(x,new_i,new_j):
    '''Crops and returns 2d image with specified uniform dimensions'''
    min_i,max_i,max_j = FindCropDimensions(x)
    x_new = CropImage(x,min_i,max_i,max_j,new_i,new_j)
    ReduceNoise_v = np.vectorize(ReduceNoise)
    x_new = ReduceNoise_v(x_new)
    return x_new
	
# AWS Helper functions
def GetShuffledKeys(bucket):
    contents = [k.key for k in bucket.objects.all()]
    contents = contents[1:]
    np.random.shuffle(contents) #Remove initial empty key in the result set
    return contents

def GetAWSCredentials():
    '''
    Get AWS credentials from config file.
    
    '''
    config = configparser.ConfigParser()
    config.read(AWS_CONFIG_FILE)
    AWS_ACCESS_KEY_ID = config['DEFAULT']['AccessKeyId']
    AWS_SECRET_ACCESS_KEY = config['DEFAULT']['AccessKeySecret']
    return AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def GetAWSClient(key_id,secret_access_key):
    '''
    Connect to S3 client. 
    
    '''
    client = boto3.resource('s3',aws_access_key_id = key_id ,
                      aws_secret_access_key= secret_access_key)
    return client
    
def GetLabelsDict(labels_dir):
    '''
    Reads csv with label data and organizes it into a dict with the image id's as keys
    and arrays of probabilities for the different zones as values.
    
    '''
    labels = pd.read_csv(labels_dir)
    labels = [[i,j] for i,j in zip(labels['Id'],labels['Probability'])] 
    labels_merged = iof.merge_17(labels) # Returns list of ids and corresponding list of zones
    labels_dict = {i:j for i,j in labels_merged}
    return labels_dict

class BatchRequester:
    '''
    Class used to request batches from AWS S3 Bucket.
    
    '''
    keys = None
    num_retrievals = 0
    key_pointer = 0
    batch_size = None
    bucket = None
    temp = None
    labels_dict = None
    extension = None
    zones = 0
    dim = []
    no_label_ary =[]
    failed_ary = []
    
    class CustomException(Exception):
        '''
        Custom Exception class
    
        '''
        def __init__(self, value):
            self.parameter = value
        def __str__(self):
            return repr(self.parameter)
    
    def __init__(self,bucket,key_ary,labels_dict,dataDir,
                 temp_dir,extension,batch_size=10,zones=17,dim=[512,660,64]):
        self.bucket = bucket
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        self.labels_dict = labels_dict.copy()
        self.dataDir = dataDir
        self.extension = extension    
        self.zones = zones
        self.dim = dim
        self.keys = self.CleanKeyAry(key_ary)
        
    def CleanKeyAry(self,key_ary):
        '''Makes certain that given keys all have labels'''
        key_ary_new=[]
        for key in key_ary:
            img_id = key.strip().replace(self.dataDir,'').replace(self.extension,'')
            if img_id in self.labels_dict.keys():
                key_ary_new.append(key)
            else:
                continue
        return key_ary_new
    def DoItemsRemain(self):
        '''
        Checks to see whether any unexamined keys remain.
        
        '''
        if self.key_pointer < len(self.keys):
            return True
        else:
            return False
    
    def AttemptLabelRetrieval(self,key):
        '''
        Attempts to retrieve label data using key from dictinary passed to 
        the BatchRequester
        
        '''
        
        img_id = key.strip().replace(self.dataDir,'').replace(self.extension,'')
        try:
            label = np.array(self.labels_dict[img_id])
        except(KeyError):
            print("{} is not in the labeled data!".format(img_id))
            self.no_label_ary.append(img_id)
            raise self.CustomException("Label not found!")
        return label
        
    def AttemptDataRetrieval(self,key):
        '''
        Attempts to retrieve data from specified bucket using key.
        
        '''        
        img_id = key.strip().replace(self.dataDir,'').replace(self.extension,'')
        filename = "{}.{}".format(img_id,self.extension)
        path = os.path.join(self.temp_dir,filename)
        failure = False
        with open(path,"w+b") as f:
            self.bucket.download_fileobj(key,f)
            try:
                img_array = iof.read_data(path)
            except:
                print("Something went wrong. Skipping {}".format(img_id))
                self.failed_ary.append(img_id)
                failure = True
        os.remove(path)
        if failure:
            raise self.CustomException("Data could not be retrieved.")
        else:
            return img_array
        
    def NextBatch(self,size=None):
        '''
        Gets a batch of data of the specified size.  If no more images remain,
        then it returns a batch of smaller size.
        
        '''
        if not self.DoItemsRemain():
            return None,None
        
        if not size:
            size = self.batch_size     
        
        i = 0
        batch_data = np.zeros((size,self.dim[0],self.dim[1],self.dim[2]))
        batch_labels = np.zeros((size,self.zones))
        
        #Avoid throttling errors
        MAX_RATE = 200 #Requests per second
        delay = 1/MAX_RATE #seconds
        
        #Grab samples
        while i < size and self.DoItemsRemain():
            time.sleep(delay)
            try:
                batch_labels[i,:] = self.AttemptLabelRetrieval(self.keys[self.key_pointer])
            except(self.CustomException):
                self.key_pointer += 1
                continue
            try:
                batch_data[i,:,:,:] = self.AttemptDataRetrieval(self.keys[self.key_pointer])
            except(self.CustomException):
                self.key_pointer += 1
                continue
            i += 1
            self.key_pointer += 1
        
        if i == size:
            return batch_data, batch_labels        
        else:
            return batch_data[:i,:,:,:],batch_labels[:i]

def CleanKeyAry(key_ary,labels_dict,dataDir,extension):
        '''
        Makes certain that given keys all have labels. 
        This function is adapted from an internal method of the batch requester.
        '''
        key_ary_new=[]
        for key in key_ary:
            img_id = key.strip().replace(dataDir,'').replace(extension,'')
            if img_id in labels_dict.keys():
                key_ary_new.append(key)
            else:
                continue 
        return key_ary_new
 
    
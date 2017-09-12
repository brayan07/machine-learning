import IOFunctions as iof
import pickle
import boto3

#Image preprocessing funcitons and global variables
#Load stats
with open('stats.pickle',"rb") as f:
    stats = pickle.load(f)
    AVERAGE = stats['average']
    STD = stats['std']
	
NOISE_THRESHOLD = 7000


def NormalizeImage(x,average,std):
    return (x-average)/std

def ExtractNormParameters(x):
    mask = x > NOISE_THRESHOLD
    average = np.average(x,weights = mask)
    std = x[mask].std()
    return average,std

def ReduceNoise(x):
    if x < NOISE_THRESHOLD:
        x = NOISE_THRESHOLD
    return x

def ReduceNoiseandNormalize(x,average,std):
    if x < NOISE_THRESHOLD:
        x = NOISE_THRESHOLD
    return (x-average)/std

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
	
#Cropping and resizing functions
def CropImage(x,min_i,max_i,max_j,new_i,new_j):
    '''Returns regions of interest in the image'''
    resize_i = False
    resize_j = False
    
    #Width
    old_i = max_i-min_i
    #If crop is of smaller width than new_i
    if old_i <= new_i:
        diff = new_i-old_i
        buffer,remainder = divmod(diff,2)
        #Move min_i
        if min_i > buffer:
            min_i = min_i - buffer
        else:
            remainder += buffer - min_i
            min_i = 0
        max_i += buffer + remainder
    else:
        #Image needs to be resized along this dimension
        resize_i = True
    
    #Height
    old_j = max_j
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
        return cv2.resize(x[min_i:max_i,:max_j],dsize=(new_j,new_i))
    else:
        return x[min_i:max_i,:max_j]

def CropCleanResize(x,new_i,new_j):
    '''Crops and returns 2d image with specified uniform dimensions'''
    min_i,max_i,max_j = FindCropDimensions(x)
    x_new = CropImage(x,min_i,max_i,max_j,new_i,new_j)
    ReduceNoiseAndNormalize_v = np.vectorize(ReduceNoiseandNormalize)
    x_new = ReduceNoise_v(x_new)
    return x_new
	
#Helper functions
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
    config.read('S3.conf')
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
    labels_merged = iof.merge_17(labels) #Returns list of ids and corresponding list of zones
    labels_dict = {i:j for i,j in labels_merged}
    return labels_dict


class BatchRequester:
    '''
    Class used to request batches from AWS server.
    
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
        self.keys = key_ary.copy()
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        self.labels_dict = labels_dict.copy()
        self.dataDir = dataDir
        self.extension = extension    
        self.zones = zones
        self.dim = dim
    
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
        #DataException = self.CustomException("Failed data retrieval!")
        
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
            raise self.CustomException("Data could ot be retrieved.")
        else:
            return img_array
    def NextBatch(self,size=None):
        '''
        Gets a batch of data of the specified size.  If no more images remain,
        then it returns a batch of smaller size.
        
        '''
        #angles = [0,8,16,24,32,40,48,56]
        #new_dim = 500
        #new_dim2 = 600
        
        if not self.DoItemsRemain():
            return None,None
        
        if not size:
            size = self.batch_size     
        
        i = 0
        batch_data = np.zeros((size,self.dim[0],self.dim[1],self.dim[2]))
        #batch_data = np.zeros((size,new_dim,new_dim2,self.dim[2]))
        batch_labels = np.zeros((size,self.zones))
        
        while i < size and self.DoItemsRemain():
            try:
                batch_labels[i,:] = self.AttemptLabelRetrieval(self.keys[self.key_pointer])
            except(self.CustomException):
                self.key_pointer += 1
                continue
            try:
                batch_data[i,:,:,:] = self.AttemptDataRetrieval(self.keys[self.key_pointer])
                #batch_data[i,:,:,:] = cv2.resize(self.AttemptDataRetrieval(self.keys[self.key_pointer]),
                 #                                dsize=(new_dim2,new_dim))
            except(self.CustomException):
                self.key_pointer += 1
                continue
            i += 1
            self.key_pointer += 1
        
        if i == size:
            return batch_data, batch_labels        
        else:
            return batch_data[:i,:,:,:],batch_labels[:i]
            
 
    
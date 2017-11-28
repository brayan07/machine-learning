Dependencies
-------------------------------------------------------------------------
This project was completed in a python 3.5 environment with the following packages:
Package	Version
Boto3	1.4.7
Botocore	1.7.3
H5py	2.7.1
Imbalanced-learn	0.3.1
Keras	2.0.8
Matplotlib	2.0.2
Opencv3	3.2.0
Pillow	4.2.1
Seaborn	0.8.1
tensorflow	1.3.0

Notice the code does not work with versions of matplotlib > 2.0.2

Before beginning, run the following command:

pip install pandas matplotlib==2.0.2 seaborn boto3 h5py imbalanced-learn tensorflow keras opencv-python pillow

Notebooks
--------------------------------------------------------------------------
The notebooks are interdependent and should be viewed in the order outlined below.  
Some of the processing of the data for this project took several hours to be completed. 
To make it easier for the grader, I created a “demo” version of the code wherein only a few samples are processed.  
Wherever the notebooks can be run using the non-demo version, I have noted it below.
 
The first step is to run the setup file. Use the -d flag to indicate the demo version. 
In the place of {id}, write the id provided in the Udacity message without quotation marks or braces.  
In the place of {secret}, do the same with the secret key provided.
  
python setup_file.py -d {id} {secret}

Now, you may verify that the notebooks are running as intended.

1.	“Exploratory Data Analysis.ipnyb”
2.	“CreateCleanDataSet.ipnyb”
3.	“MultiLayerFeatureExtraction.ipnyb”

4. Important: You may view the last notebook, “ModelingExtractedFeatures.ipnyb”, 
	      either in “demo” mode or in its full version. 
   Please be advised that running any of the cross-validation trials in the full version will 
   require an inordinate amount of time (>30 min).
	To view the notebook in the non-demo mode:
		1.	Run setup_file.py without the -d flag. 
			Reenter the authentication credentials each time.   
		2.	View the notebook, making sure to restart the kernel 
			if you had been viewing the notebook before. 	


# Medical Image Classifier
Full write up can be found here: ![Medical Image Classifer](./MedicalImageClassifer.pdf)  
## Initial Setup
Dataset comes from kaggle (around 1.25gb in size): [here](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data)  
Inspired by Hardik Desmukh's tutorial: [here](https://www.markdownguide.orghttps://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)  
  
Steps to use kaggle datasets:
1. Within your kaggle account, create a new API token. Kaggle.json file will automatically download.  
2. In terminal within your new project:  
    i. `pip install -q kaggle`    
    ii. `mkdir ~/.kaggle`  
    iii. Copy newly downloaded kaggle.json file into .kaggle folder: `cp kaggle.json ~/.kaggle/`  
    iv. Modify permissions: `chmod 600 ~/.kaggle/kaggle.json`  
    v. Download dataset file: `kaggle datasets download -d pcbreviglieri/pneumonia-xray-images`  
3. Within your project run the following:  
    ```
    import zipfile
    zf = "pneumonia-xray-images.zip"
    target_dir = "pneumonia-dataset"
    zfile = zipfile.ZipFile(zf)
    zfile.extractall(target_dir)
    ```
You should now see the extracted folder pneumonia-dataset inside your project directory.

I used the Pycharm IDE to build and run this project. Each .py file in the repo is designed to be a standalone main file that can be run from your IDE of choice once the required packages are installed and the dataset is downloaded. The path-name variables may need to be adjusted in each .py file based on the structure of your project directory:

## Models
The various models used are in the following files:  
1. **simpleCNN.py** is the simplest convolutional neural net model architecture.  
2. **intermediateCNN.py** adds an additional convolutional, max pooling, and dense layer  
3. **deepestCNN.py** is the most complex of the binary classifier CNNs  
4. **multiClassModel.py** file is for the multi-class classifier to distinguish between healthy patients, patients with viral pneumonia, and patients with bacterial pneumonia. This required hand annotating the above dataset to separate out bacterial vs viral cases.  
5. **spatialTransformer.py** adds a spatial transformer network to the simple CNN.  

NOTE: Due to the size of the dataset and it being image data, the first four models took anywhere from 10-30 minutes each to train on my laptop. The spatial transformer file took approximately 3.5 hours to train, however.

(*side note:* tensorflow did not natively play well with Mac M1. A work around was required utilizing Miniforge and a new conda environment inside pycharm)

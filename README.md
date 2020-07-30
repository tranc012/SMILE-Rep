# SMILE-Rep
HiPerGator - A100 

This is a very minimal repo. The primary code is sourced from Momentum Contrast for Unsupervised Visual Representation Learning ( https://arxiv.org/abs/1911.05722 ) with minor changes for compatiability and hyperparameter adapations. Due to the large size, I cannot provide the saved model weights nor the data directly here. \\

Code Summary \\ 
 
Data: The data used was from a medical CT image Kaggle dataset (RSNA Intracranial Hemorrhage Detection https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview )  We split this data by percentages, and end up with the corresponding number of images: unsupervised data (564601), training data (150560), and testing data (37640). To show the effectiveness of the method (and what was done on HiPerGator), 20% of the training data was used giving. (30112) images. \\

Model Architecture: The MoCo paper utilizes a ResNet-50 style encoder. A fully connected layer is used to train a linear classifier after the unsupervised training stage. 

Batch Size: 128 \\  
Learning Rate: 0.03 followed by a lr decay rate. \\
Framework: PyTorch (on A100: version 1.6.0) \\ 
Optimizer: Adam \\ 


1) (Train_MoCo) An unsupervised stage is used with the momentum contrast model. We extract the weights from the best saved model (110 epochs). Note: This was not done on the HiPerGator implementation and is not relevant to the computational speed problem. \\

2) (MoCo_Downstream) These saved weights are used for a supervised downstream task (classification). As mentioned before, in the HiPerGator implementation, 20% of the training data was used along with the test set, evaluated at every epoch. For the benchmark task, the speed was examined on 100 epochs. 





# <font color="green">Pre-trained-LLM-in-Neural-RSA-models</font>
This is an repository(file) containing all the code(and possiblly data) used for the Thesis Project.

# Repository structure
The repository consist of *data* folder which include the data and related scripts used for training of the model, and *experiments* folders which includes codes for experiments we did. The **Experiment01** include the codes for experiments about literal listener models, **Experiment02** include the codes for experiments about literal speaker models, and **Expeirments03** include the codes for experiments about pre-trained large language models and training data size. Note that the other experiments done and stated in the report (for exapmple code for CS-CNN, etc) are included in the folder **Other_Experiments**. In addition, the other experiments codes which is not stated in the report(for example codes tried to construct the amortized models stated in the appendix) are in the **Other**.

<font color="orange">**Important!!**</font>  
Since this repository is collection of codes used for experiments, and folder structure is refactored(reconstructed) for easier view and understanding, some code will have some error which is due to different folder structure from original environment. In such a case, please modify the file path in formation on the code (if you want to run the code). In addition, due to file size, some data and model parameter information is not included. If you need it for running code, please first run the code which will generate the required data.

# Data
The Color folder include the raw data and data processing codes, while the ShapeWorld folder contains the codes for processing the data.
### Color
- colors.csv : The raw data of color data in csv format. It is take from https://www.kaggle.com/datasets/nickatomlin/colors?select=colors.csv
- corpus.py : The code defining data loader for the color data. The implementations are based on the code in https://www.kaggle.com/code/nickatomlin/cs288-hw6-public 
- split_data.py : The code to split the color data. Remember the Literal Listener and Literal Spaeker need to be trained on different data when evelutated with communication accuracy between L0 and S0.
### ShapeWorld
- The raw ShapeWorld data i snot included since it is too big to include. The data can be generated by script provided from original work's code(https://github.com/juliaiwhite/amortized-rsa/blob/master/shapeworld.py) Please refer thier Readme file for detailed explanation for generating data. The script will produce many chunk of numpy file which include 1000 data each.
- shapeworld_data.py : The code for loading the generated data by scripts. The implemantation is based on the original work's https://github.com/juliaiwhite/amortized-rsa/blob/master/data.py. The code is modified to provide data for BERT-based models.  


# Experiment01
This folder includes the codes for experiments about literal listner L0. There are 3 different folder for 3 different tasks, relative probability modeling for Color data, direct probability modeling for Color data, and relative probability modeling for ShapeWorld data.
### Color_direct_literal_listener
This folder includes the codes for training 5 models for direct probability modeling literal listener.
- L0_simple.ipynb : The code for model **Simple L0** in the report.
- L0_original.ipynb : The code for model **Original L0** in the report.
- L0_bert_rnn.ipynb : The code for model **BERT-RNN L0** in the report.
- L0_bert_sum.ipynb : The code for model **BERT-SUM L0** in the report.
- L0_bert_cls.ipynb : The code for model **BERT-CLS L0** in the report.
- tmp/vocab.pkl: The vocabulary file for training the model(for avoiding vocabulary dictionaly become different for each excecution).  

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp* file. From second time the code is designed to load the embedding data from *tmp* for saving time. Data loader code *corpus.py* is moved to *data* directory. For literal listener for validation(communication accuracy between L0 and S0), please split the data first and change the data path to appropriate one.  

### Color_relative_literal_listener
This folder includes the codes for training 5 models for relative probability modeling literal listener.
- Simple_L0.ipynb : The code for model **Simple L0** in the report.
- Emb-RNN_L0.ipynb : The code for model **Original L0** in the report.
- BERT-RNN_L0.ipynb : The code for model **BERT-RNN L0** in the report.
- BERT-SUM_L0.ipynb : The code for model **BERT-SUM L0** in the report.
- BERT-CLS_L0.ipynb : The code for model **BERT-CLS L0** in the report.
- tmp/vocab.pkl: The vocabulary file for training the model(for avoiding vocabulary dictionaly become different for each excecution). 

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp* file. From second time the code is designed to load the embedding data from *tmp* for saving time. Data loader code *corpus.py* is moved to *data* directory. For literal listener for validation(communication accuracy between L0 and S0), please split the data first and change the data path to appropriate one.

### ShapeWorld_relative_literal_listener
This folder includes the codes for training 5 models for relative probability modeling literal listener.
- L0_Simple.ipynb : The code for model **Simple L0** in the report.
- L0_emb_rnn.ipynb : The code for model **Original L0** in the report.
- L0_bert_rnn.ipynb : The code for model **BERT-RNN L0** in the report.
- L0_bert_sum.ipynb : The code for model **BERT-SUM L0** in the report.
- L0_bert_sent.ipynb : The code for model **BERT-CLS L0** in the report.  

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. In addition, the training process logs are saved to *metrics* folder. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp_embs* file. From second time the code is designed to load the embedding data from *tmp_embs* for saving time. Data loader code *shapeworld_data.py* is moved to *data* directory. For CS-CNN encoder, the code is moved to *Other_Experiments/CS-CNN/cs-cnn.py* and the actual model parameters for CS-CNN encoder need to be produced by code in *Other_Experiments/CS-CNN* and move to *model_params*. For literal listener for validation(communication accuracy between L0 and S0), please change the data path to appropriate one.


# Experiment02

# Experiment03

# Other Experiment
- CS-CNN
- effect of DeepSet
- emb vector sparsity


# Other Codes
- include All lang shapeworld and cite it

# <font color="green">Pre-trained-LLM-in-Neural-RSA-models</font>
This is an repository containing all the code(and possiblly data) used for the Thesis Project for Advanced Computer Science Course.

# Repository structure
The repository consist of *data* folder which include the data and related scripts used for training of the model, and *experiments* folders which includes codes for experiments we did. The **Experiment01** include the codes for experiments about literal listener models, **Experiment02** include the codes for experiments about literal speaker models, and **Expeirments03** include the codes for experiments about pre-trained large language models and training data size. Note that the other experiments done and stated in the report (for exapmple code for CS-CNN, etc) are included in the folder **Other_Experiments**. In addition, the other experiments codes which is not stated in the report(for example codes tried to construct the amortized models stated in the appendix) are in the **Other**.

<font color="orange">**Important!!**</font>  
Since this repository is collection of codes used for experiments, and folder structure is refactored(reconstructed) for easier view and understanding, some code will have some error which is due to different folder structure from original environment. In such a case, please modify the file path in formation on the code (if you want to run the code). In addition, due to file size, some data and model parameter information is not included. If you need it for running code, please first run the code which will generate the required data.

# Data
The Color folder include the raw data and data processing codes, while the ShapeWorld folder contains the codes for processing the data.
### Color
- colors.csv : The raw data of color data in csv format. It is take from https://www.kaggle.com/datasets/nickatomlin/colors?select=colors.csv
- <font color="blue">corpus.py</font> : The code defining data loader for the color data. The implementations are based on the code in https://www.kaggle.com/code/nickatomlin/cs288-hw6-public 
- <font color="blue">split_data.py</font> : The code to split the color data. Remember the Literal Listener and Literal Spaeker need to be trained on different data when evelutated with communication accuracy between L0 and S0.
### ShapeWorld
- The raw ShapeWorld data i snot included since it is too big to include. The data can be generated by script provided from original work's code(https://github.com/juliaiwhite/amortized-rsa/blob/master/shapeworld.py) Please refer thier Readme file for detailed explanation for generating data. The script will produce many chunk of numpy file which include 1000 data each.
- <font color="blue">shapeworld_data.py</font> : The code for loading the generated data by scripts. The implemantation is based on the original work's https://github.com/juliaiwhite/amortized-rsa/blob/master/data.py. The code is modified to provide data for BERT-based models.  


# Experiment01
This folder includes the codes for experiments about literal listner L0. There are 3 different folder for 3 different tasks, relative probability modeling for Color data, direct probability modeling for Color data, and relative probability modeling for ShapeWorld data.
### Color_direct_literal_listener
This folder includes the codes for training 5 models for direct probability modeling literal listener.
- <font color="green">L0_simple.ipynb</font> : The code for model **Simple L0** in the report.
- <font color="green">L0_original.ipynb</font> : The code for model **Original L0** in the report.
- <font color="green">L0_bert_rnn.ipynb</font> : The code for model **BERT-RNN L0** in the report.
- <font color="green">L0_bert_sum.ipynb</font> : The code for model **BERT-SUM L0** in the report.
- <font color="green">L0_bert_cls.ipynb</font> : The code for model **BERT-CLS L0** in the report.
- tmp/vocab.pkl: The vocabulary file for training the model(for avoiding vocabulary dictionaly become different for each excecution).  

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp* file. From second time the code is designed to load the embedding data from *tmp* for saving time. Data loader code *corpus.py* is moved to *data* directory. For literal listener for validation(communication accuracy between L0 and S0), please split the data first and change the data path to appropriate one.  

### Color_relative_literal_listener
This folder includes the codes for training 5 models for relative probability modeling literal listener.
- <font color="green">Simple_L0.ipynb</font> : The code for model **Simple L0** in the report.
- <font color="green">Emb-RNN_L0.ipynb</font> : The code for model **Original L0** in the report.
- <font color="green">BERT-RNN_L0.ipynb</font> : The code for model **BERT-RNN L0** in the report.
- <font color="green">BERT-SUM_L0.ipynb</font> : The code for model **BERT-SUM L0** in the report.
- <font color="green">BERT-CLS_L0.ipynb</font> : The code for model **BERT-CLS L0** in the report.
- tmp/vocab.pkl: The vocabulary file for training the model(for avoiding vocabulary dictionaly become different for each excecution). 

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp* file. From second time the code is designed to load the embedding data from *tmp* for saving time. Data loader code *corpus.py* is moved to *data* directory. For literal listener for validation(communication accuracy between L0 and S0), please split the data first and change the data path to appropriate one.

### ShapeWorld_relative_literal_listener
This folder includes the codes for training 5 models for relative probability modeling literal listener.
- <font color="">L0_emb_rnn_original.ipynb</font> : The code which reconstructed the original work implementation. Use to compare with CS-CNN-based model to observe the effect of CS-CNN.
- <font color="green">L0_Simple.ipynb</font> : The code for model **Simple L0** in the report.
- <font color="green">L0_emb_rnn.ipynb</font> : The code for model **Original L0** in the report.
- <font color="green">L0_bert_rnn.ipynb</font> : The code for model **BERT-RNN L0** in the report.
- <font color="green">L0_bert_sum.ipynb</font> : The code for model **BERT-SUM L0** in the report.
- <font color="green">L0_bert_sent.ipynb</font> : The code for model **BERT-CLS L0** in the report.  

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. In addition, the training process logs are saved to *metrics* folder. Note that when you run the BERT-related code first, it will cache the embedding representation of the target data in *tmp_embs* file. From second time the code is designed to load the embedding data from *tmp_embs* for saving time. Data loader code *shapeworld_data.py* is moved to *data* directory. For CS-CNN encoder, the code is moved to *Other_Experiments/CS-CNN/cs-cnn.py* and *Other_Experiments/CS-CNN/vision.py*, and the actual model parameters for CS-CNN encoder need to be produced by code in *Other_Experiments/CS-CNN* and move to *model_params*. For literal listener for validation(communication accuracy between L0 and S0), please change the data path to appropriate one.


# Experiment02
This folder includes the codes for experiments about literal speaker S0. There are 2 different folder for 2 different tasks, literal speaker modeling for Color data, and literal speaker modeling for ShapeWorld data.
### Color_literal_speaker
This folder includes the codes for training 3 models for modeling literal speaker.
- S0_Baseline_Lang-loss.ipynb : The code for model **Original-RNN S0** with out pad and packing the utterance. Note that this code was used for testing model architecture and not used in the experiment reported.
- <font color="green">S0_Baseline-pad-pack_Lang-loss.ipynb</font> : The code for model **Original-RNN S0** in the report.
- <font color="green">S0_GPT-based.ipynb</font> : The code for model **All-fine-tuned GPT-2 S0** in the report.
- <font color="green">S0_GPT-based_freeze_weights.ipynb</font> : The code for model **Part-fine-tuned GPT-2 S0** in the report.
- <font color="blue">literal_listener_color.py</font> : The code including the model for literal listener used in evaluation of speaker models.
- tmp/vocab.pkl: The vocabulary file for training the model(for avoiding vocabulary dictionaly become different for each excecution). 

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Data loader code *corpus.py* is moved to *data* directory. Also the model parameter for validation literal listener need to be placed in *model_params* too. For literal speaker for validation(communication accuracy between L0 and S0), please change the data path to appropriate one.

### Shapeworld_literal_speaker
This folder includes the codes for training 4 models for modeling literal speaker.
- <font color="green">S0_classifier_shapeworld.ipynb</font> : The code for model **Classifier S0** in the report.
- <font color="green">S0_RNN_shapeworld.ipynb</font> : The code for model **Original-RNN S0** in the report.
- <font color="green">S0_GPT-based_shapeworld.ipynb</font> : The code for model **All-fine-tuned GPT-2 S0** in the report.
- <font color="green">S0_GPT-based_shapeworld_freeze_weights.ipynb</font> : The code for model **Part-fine-tuned GPT-2 S0** in the report.
- <font color="blue">literal_listener_shapeworld.py</font> : The code including the model for literal listener used in evaluation of speaker models. 

Once the code is executed, the code should save the PyTorch model parameter file in the directory *model_params*. Data loader code *shapeworld_data.py* is moved to *data* directory. For CS-CNN encoder, the code is moved to *Other_Experiments/CS-CNN/cs-cnn.py* and *Other_Experiments/CS-CNN/vision.py*, and the actual model parameters for CS-CNN encoder need to be produced by code in *Other_Experiments/CS-CNN* and move to *model_params*. Also the model parameter for validation literal listener need to be placed in *model_params* too. For literal speaker for validation(communication accuracy between L0 and S0), please change the data path to appropriate one.

# Experiment03
This folder includes the codes for the experiments about relationship between LLM and train data size.
### Color_L0
- <font color="green">Color_L0_BERT-CLS_VS_Emb-sum_relative.ipynb</font> : This is the full code for running training of LLM-based model and Simple model with different data size. Note this might require very large memory.
- <font color="yellow">Emb_trainSize.py, BERT_trainSize.py and run_experimets.bat</font> : This is the experiments scripts which the code above separetd for several part so that it will not require large memory. Please use this scripts for training.
- <font color="blue">color_literal_listener.py</font> : The code including the model for literal listener used in evaluation of speaker models.
### ShapeWorld_L0
- <font color="green">ShapeWorld_L0_BERT-CLS_VS_Emb-sum.ipynb</font> : This is the full code for running training of LLM-based model and Simple model with different data size. Note this might require very large memory.
- <font color="yellow">Emb_trainSize.py, BERT_trainSize.py and run_experimets.bat</font> : This is the experiments scripts which the code above separetd for several part so that it will not require large memory. Please use this scripts for training.
- <font color="blue">literal_listener_shapeworld.py</font> : The code including the model for literal listener used in evaluation of speaker models.
### Color_S0
- <font color="green">Color_S0_GPT2_VS_RNN.ipynb</font> : This is the full code for running training of LLM-based model and RNN-based model with different data size. Note this might require very large memory.
- <font color="yellow">RNN_trainSize.py, GPT2_trainSize.py and run_experimets.bat</font> : This is the experiments scripts which the code above separetd for several part so that it will not require large memory. Please use this scripts for training.
- <font color="blue">color_literal_listener.py</font> : The code including the model for literal listener used in evaluation of speaker models.
- <font color="blue">color_literal_speaker.py</font> : The code including the model for literal speakers.
### ShapeWorld_S0
- <font color="green">ShapeWorld_S0_GPT2_VS_RNN.ipynb</font> : This is the full code for running training of LLM-based model and RNN-based model with different data size. Note this might require very large memory.
- <font color="yellow">RNN_trainSize.py, GPT2_trainSize.py and run_experimets.bat</font> : This is the experiments scripts which the code above separetd for several part so that it will not require large memory. Please use this scripts for training.
- <font color="blue">literal_listener_shapeworld.py</font> : The code including the model for literal listener used in evaluation of speaker models.
- <font color="blue">literal_speaker_shapeworld.py</font> : The code including the model for literal speakers.  

Again make sure all the model parameters will be saved in *model_params*. The data loader code and CS-CNN encoder related codes are in corresponding directory. The actual CS-CNN encoder and literal listener(for S0) model file need to be placed in *model_params*. All the training process log will be saved in *metrics*. If required, put appropriate embedding vector caches to *tmp* or *tmp_embs* folder.

- <font color="blue">plot.ipynb</font> : The code for analyzing the log data produced by each experiment. Used to produce the results in the report.

# Other Experiment
This folder contains the codes for other experiments which is stated in the report, but not directly related to above 3 experiments.

### CS-CNN
- <font color="yellow">Effect_of_CS-CNN</font> : The collection of codes to study the effect of CS-CNN encoder in the trainig process like epoch number.
- <font color="green">reconstruct_with_CNN.ipynb and reconstruct_with_original_CNN.ipynb</font> : These codes are tried to construct the pre-trained CS-CNN encoder stated in the report. By comparing two different CNN structures, we found out the 4 layer CNN provided from *vision.py* is better for performance. If you need to get CS-CNN encoder, please run the <font color="green">*reconstruct_with_original_CNN.ipynb*</font>.
- <font color="blue">auto_encoder.ipynb</font> : This code is also experiments to find out best CNN architecture for CS-CNN encoder.
- <font color="green">cs-cnn.py</font> : The file to define the CS-CNN encoder.
- <font color="green">vision.py</font> : The file used to define the CS-CNN encoder. The implementation is from original work(https://github.com/juliaiwhite/amortized-rsa/blob/master/vision.py).

### DeepSet
We compared the performance result of DeepSet-based encoder speaker(<font color="green">S0_classifier_shapeworld.ipynb</font>) and Non-DeepSet-based encoder speaker(<font color="green">S0_NoDeepSet_classifier_shapeworld.ipynb</font>) to found out the effect of using DeepSet architecture for speaker encoder.

### Embedding_sparsity
Make sure the required data like BERT-embedding vector file and literal listener model parameter file are placed in the *model_params*. We examined the sparsity of the embedding vector for BERT and nn.Embedding by using the scripts <font color="green">dense_check.ipynb</font>. <font color="blue">literal_listener_shapeworld.py</font> is the code including the model for literal listeners.


# Other Codes
This folder includes codes which is not used in the experiments stated in the report. Note that since these are not related to the main purpose of our report, they are not well documented and prepared. If you want to run them, there will be many directory structure problems(but these problem should be easy to solve if you observed the code).
### Amortized Speaker
The *amortized_speaker/Color* and *amortized_speaker/ShapeWorld* includes the trial to implement the amortized speaker for Color data and ShapeWorld data respectively.
### Same context more utterance ShapeWorld data
The folder *context_data* include the scripts and code for training model with more utterance data. More utterance data means with one context three utterance for each target image is generated (usually it is one utterance for one context). The details are explained in the report, but the main purpose is avoid overfitting of the model to specific context. *S1_Baseline_CEL_3-langs_shapeworld.ipynb* is model used this more utterace data, and the *shapeworld.py* is the scripts to produce such a data(it is based on the original work code, https://github.com/juliaiwhite/amortized-rsa/blob/master/shapeworld.py with modifications). If you want to generate the data, please download the original repository(https://github.com/juliaiwhite/amortized-rsa) and replace the *shapeworld.py* file with ours.

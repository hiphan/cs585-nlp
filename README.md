# Deep Learning for Text Classification by Reading Difficulty

## Files
- `data_utils:` download the [OneStopEnglishCorpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) dataset 
from Github and create json file. 
- `newsela_data_utils`: read and split the Newsela dataset into train and test sets as Pandas' Dataframe. IMPORTANT: this 
script requires a local folder of the Newsela dataset, which is available [here](https://newsela.com/data/).  
<u>NOTE</u>: The two scripts above need to run before running the models below. 
- `RNN.ipynb`: RNN model using GloVe Embeddings.
- `RNN_Random_Embeddings.ipynb`: Initial RNN code (uses randomly initialized embeddings).
- `bayes_newsela.py`: Naive Bayesian Classifier for the Newsela dataset.
- `bayes_onestop.py`: Naive Bayesian Classifier for the OneStopEnglish Corpus dataset.
- `utils.py`: tokenizer for the OneStopEnglishCorpus dataset using nltk library. 
- `svm_onestop.py`: Support Vector Machine (SVM) model for the OneStopEnglishCorpus dataset.
- `svm_newsela.py`: SVM model for the Newsela dataset. 
- `logreg.py`: Logistic regression for OneStopEnglishCorpus dataset.
- `dataload.py`: Contains methods for Newsela dataset analysis.
<u>NOTE</u>: For the two notebooks below, Pandas' Dataframe version of the Newsela dataset needs to be on Google Drive 
to mount.
- `BERTRank.ipynb`: Fine-tuned BERT model. Link to Google Colab is 
[here](https://colab.research.google.com/drive/1PLyNxB430viZId2-pEFFNWUkYfYsv2s9), which is also included in the notebook.
- `USERank.ipynb`: Universal Sentence Encoder model. Link to Google Colab is 
[here](https://colab.research.google.com/drive/1KIAszDpVugPFyjrWiIdVLHOCSTMH8s88), which is also included in the notebook.

## Requirements
 - As mentioned above, a local copy of the Newsela dataset.
 - All required packages are included in `requirements.txt`. Note that this list also contains several redundant packages
 that we tried but did not need for our final models. The environment can be created using `conda create --name <env> --file <this file>
`.
 - For our BERT and Universal Sentence Encoder models, you might need to install Tensorflow Hub to run the notebook on 
 Google Colab.
 - For our BERT model, the package bert-tensorflow is required, but already is included as part of the notebook. 
 
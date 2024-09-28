import os
os.environ['KAGGLE_USERNAME'] = ''
os.environ['KAGGLE_KEY'] = 'masked'

!kaggle datasets download -d fmendes/fmendesdat263xdemos

import zipfile
with zipfile.ZipFile('fmendesdat263xdemos.zip', 'r') as zip_ref:
    zip_ref.extractall('fmendesdat263xdemos')

import pandas as pd
file_path = 'fmendesdat263xdemos/filename.csv'  # Change filename as per the dataset
data = pd.read_csv(file_path)

print(data.head())

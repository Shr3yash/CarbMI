import os
import zipfile
import pandas as pd
import requests

os.environ['KAGGLE_USERNAME'] = ''
os.environ['KAGGLE_KEY'] = ''

!kaggle datasets download -d vatsalmavani/fitness-workout-diet-dataset
!kaggle datasets download -d arashnic/fitbit
!kaggle datasets download -d dhanyajothimani/athlete-vo2-max

def unzip_dataset(zip_file, folder_name):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_name)

unzip_dataset('fitness-workout-diet-dataset.zip', 'fitness_workout_diet')
unzip_dataset('fitbit.zip', 'fitbit_data')
unzip_dataset('athlete-vo2-max.zip', 'vo2max_data')

fitness_diet_df = pd.read_csv('fitness_workout_diet/your_file.csv')
fitbit_df = pd.read_csv('fitbit_data/your_file.csv')
vo2max_df = pd.read_csv('vo2max_data/your_file.csv')

uci_sports_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip'
uci_sports_response = requests.get(uci_sports_url)
with open('uci_sports.zip', 'wb') as f:
    f.write(uci_sports_response.content)
unzip_dataset('uci_sports.zip', 'uci_sports_activities')

uci_smartphones_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
uci_smartphones_response = requests.get(uci_smartphones_url)
with open('uci_smartphones.zip', 'wb') as f:
    f.write(uci_smartphones_response.content)
unzip_dataset('uci_smartphones.zip', 'uci_smartphones')

print(fitness_diet_df.head())

# kaggle datasets download -d dbdmobile/myanimelist-dataset

## Public Dataset from Kaggle
# https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset

# Authenticate kaggle API
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def load_set():
    api = KaggleApi()
    api.authenticate()

    ## Dowload Dataset
    # List and information about the different attributes and characteristics of each anime.
    api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='anime-dataset-2023.csv', path='./data/')
    # WARNING LARGE FILE
    api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='users-score-2023.csv', path='./data/')
    


    # Unzip file
    with zipfile.ZipFile('./data/anime-dataset-2023.csv', 'r') as zipref:
        zipref.extractall('./data/')
    with zipfile.ZipFile('./data/users-score-2023.csv', 'r') as zipref:
        zipref.extractall('./data/')

    # Delete zip file
    if os.path.exists('./data/anime-dataset-2023.csv.zip'):
        os.remove('./data/anime-dataset-2023.csv.zip')
    if os.path.exists('./data/users-score-2023.csv.zip'):
        os.remove('./data/users-score-2023.csv.zip')


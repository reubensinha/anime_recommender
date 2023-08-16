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
    if not os.path.exists('./data/anime-dataset-2023.csv.zip') and not os.path.exists('./data/anime-dataset-2023.csv'):
        api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='anime-dataset-2023.csv', path='./data/')
    # WARNING LARGE FILE
    if not os.path.exists('./data/users-score-2023.csv.zip') and not os.path.exists('./data/users-score-2023.csv'):
        api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='users-score-2023.csv', path='./data/')
    


    # Unzip and delete file
    if os.path.exists('./data/anime-dataset-2023.csv.zip'):
        with zipfile.ZipFile('./data/anime-dataset-2023.csv.zip', 'r') as zipref:
            zipref.extractall('./data/')
        os.remove('./data/anime-dataset-2023.csv.zip')
        
    if os.path.exists('./data/users-score-2023.csv.zip'):
        with zipfile.ZipFile('./data/users-score-2023.csv.zip', 'r') as zipref:
            zipref.extractall('./data/')
        os.remove('./data/users-score-2023.csv.zip')


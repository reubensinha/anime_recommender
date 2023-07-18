# kaggle datasets download -d dbdmobile/myanimelist-dataset

## Public Dataset from Kaggle
# https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset

# Authenticate kaggle API
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

## Dowload Dataset
# List and information about the different attributes and characteristics of each anime.
api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='anime-filtered.csv', path='./data/')

# Unzip file
with zipfile.ZipFile('./data/anime-filtered.csv.zip', 'r') as zipref:
    zipref.extractall('./data/')

# Delete zip file
if os.path.exists('./data/anime-filtered.csv.zip'):
    os.remove('./data/anime-filtered.csv.zip')
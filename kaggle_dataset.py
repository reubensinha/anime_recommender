# kaggle datasets download -d dbdmobile/myanimelist-dataset

## Public Dataset from Kaggle
# https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset

# Authenticate kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

## Dowload Datasets

# List and information about the different attributes and characteristics of each anime.
api.dataset_download_file("dbdmobile/myanimelist-dataset", file_name='anime-filtered.csv', path='./data/')
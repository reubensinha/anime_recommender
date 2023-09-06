## Imports
import MAL_my_data
import kaggle_dataset
import model

import os
import numpy as np
import pandas as pd


DEBUG = 1

numeric_keys = {
    'genres': {},
    'producers': {},
    'licensors': {},
    'studios': {}
    }


def To_Int(str, list):
    # Convert strings to integer keys
    dict = numeric_keys[str]
    new_list = []

    for item in list:
        if item not in dict:
            dict[item] = len(dict) + 1
        new_list.append(dict[item])
    
    numeric_keys[str] = dict
    return new_list


def get_anime_data():
    ## Download and return dataframe with MAL anime database
    # genres = {'Action': 1,
    #           'Adventure': 2,
    #           'Avant Garde': 3,
    #           'Award Winning': 4,
    #           'Boys Love': 5,
    #           'Comedy': 6,
    #           'Drama': 7,
    #           'Fantasy': 8,
    #           'Girls Love': 9,
    #           'Gourmet': 10,
    #           'Horror': 11,
    #           'Mystery': 12,
    #           'Romance': 13,
    #           'Sci-Fi': 14,
    #           'Slice of Life': 15,
    #           'Sports': 16,
    #           'Supernatural': 17,
    #           'Suspense': 18,
    #           'Ecchi': 19,
    #           'Erotica': 20,
    #           'Hentai': 21}
    kaggle_dataset.load_set()
    ani_list = pd.read_csv('./data/anime-dataset-2023.csv', converters={'Genres' : lambda x : To_Int('genres', list(map(str.strip, x.split(',')))),
                                                                    'Producers' : lambda x : To_Int('producers', list(map(str.strip, x.split(',')))),
                                                                    'Licensors' : lambda x : To_Int('licensors', list(map(str.strip, x.split(',')))),
                                                                    'Studios' : lambda x : To_Int('studios', list(map(str.strip, x.split(','))))})
    
    rating_list = pd.read_csv('./data/users-score-2023.csv', usecols=["user_id", "anime_id", "rating"])

    return ani_list, rating_list


def delete_data():
    if os.path.exists('data'):
            os.remove('data')
    return


def select_input(select):
    match select:
        case 0:
            print("Shuting down...")
        case 1:
            ani_list, rating_list = get_anime_data() # Dataframe
            # TODO: Implement Timeout
            my_list = MAL_my_data.get_my_anime_list() # Dataframe
            my_list.to_csv('./cache/my_list.csv') # Cache my data
            # TODO: return files
            model.rec_anime(ani_list, my_list, rating_list)
        case 2: #TODO
            print("Feature not implemented \nChoose again")
            # manga_list = get_manga_data()
            # my_list = MAL_my_data.get_my_manga_list()
            # rec_manga(manga_list, my_list)
        case 3:
            delete_data()
            get_anime_data()
        case _:
            print("Invalid Selection\n")


def list_menu():
    print("\n1: Recommend Anime" + 
        "\n2: Recommend Manga *WIP Placeholder*" + #TODO
        "\n3: Reset Data" +
        "\n0: Exit")
    
    return int(input("\nEnter Selection: "))


def main():
    # Authorize MAL API
    MAL_my_data.OAuth2()
    
    # manga_list = get_manga_data()
    select = -1 # Some Arbitrary number

    while select != 0:
        select = list_menu()
        select_input(select)
    
    # Exit program
    exit()


main()




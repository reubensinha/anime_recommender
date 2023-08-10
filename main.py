## Imports
import MAL_user_data
import kaggle_dataset

import numpy as np
import pandas as pd

def rec_anime(ani_list, user_list):
    #TODO:
    return


def get_anime_data():
    ## Download and return dataframe with MAL anime database
    kaggle_dataset.load_set()
    ani_list = pd.read_csv('./data/anime-filtered.csv', index_col='Anime_id')

    return ani_list


def select_input(select):
    match select:
        case 0:
            print("Shuting down...")
        case 1:
            ani_list = get_anime_data() # Dataframe
            user_list = MAL_user_data.get_user_anime_list() # Array
            rec_anime(ani_list, user_list)
        # case 2: TODO
        #     manga_list = get_manga_data()
        #     user_list = MAL_user_data.get_user_manga_list()
        #     rec_manga(manga_list, user_list)
        case _:
            print("Invalid Selection\n")


def list_menu():
    print("\n1: Recommend Anime" + 
        # "\n2: Recommend Manga" + TODO
        "\n0: Exit")
    
    return int(input("\nEnter Selection: "))


def main():
    login = False
    while not login:
        # Authorize MAL API
        login = MAL_user_data.OAuth2()
    
    # manga_list = get_manga_data()

    int(select)
    while select != 0:
        select = list_menu()
        select_input(select)
    
    # TODO: Exit


main()
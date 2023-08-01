## Imports
import MAL_user_data
import kaggle_dataset

def rec_anime():
    #TODO:
    return


def select_input(select):
    match select:
        case 0:
            print("Shuting down...")
        case 1:
            rec_anime()
        # case 2: TODO
        #     return 
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

    int(select)
    while select != 0:
        select = list_menu()
        select_input(select)


main()
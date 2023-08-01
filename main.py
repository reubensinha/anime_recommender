## Imports
import MAL_user_data
import kaggle_dataset


def select_input():
    # TODO:
    return


def list_menu():
    # TODO:
    return


def main():
    login = False
    while not login:
        # Authorize MAL API
        login = MAL_user_data.OAuth2()

    int(select)
    while select != 0:
        list_menu()
        select = int(input("Enter Selection: "))
        select_input(select)


main()
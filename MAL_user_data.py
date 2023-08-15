## Access data via MAL API

import os
import json
import requests
import secrets
import pandas as pd

import private

# Debug mode
DEBUG = 1

## CLIENT ID is gotten by registering at https://myanimelist.net/apiconfig
CLIENT_ID = private.get_id()
# CLIENT_SECRET = ""


# Generate a new Code Verifier / Code Challenge.
def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)
    return token[:128]


# Generate URL needed to authorise application.
# TODO: Implement state parameter
def print_new_authorisation_url(code_challenge: str):
    global CLIENT_ID

    url = f'https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&code_challenge={code_challenge}'
    print(f'Login by clicking here: {url}\n')


# 3. Once you've authorised your application, you will be redirected to the webpage you've
#    specified in the API panel. The URL will contain a parameter named "code" (the Authorisation
#    Code). You need to feed that code to the application.
def generate_new_token(authorisation_code: str, code_verifier: str) -> dict:
    global CLIENT_ID #, CLIENT_SECRET

    url = 'https://myanimelist.net/v1/oauth2/token'
    data = {
        'client_id': CLIENT_ID,
        # 'client_secret': CLIENT_SECRET,
        'code': authorisation_code,
        'code_verifier': code_verifier,
        'grant_type': 'authorization_code'
    }

    response = requests.post(url, data)
    response.raise_for_status()  # Check whether the request contains errors

    token = response.json()
    response.close()
    if DEBUG:
        print('Token generated successfully!')

    with open('token.json', 'w') as file:
        json.dump(token, file, indent = 4)
        if DEBUG:
            print('Token saved in "token.json"')
    

    return token


# Test the API by requesting your profile information
def print_user_info():
    access_token = str(token['access_token'])
    url = 'https://api.myanimelist.net/v2/users/@me'
    response = requests.get(url, headers = {
        'Authorization': f'Bearer {access_token}'
        })
    
    response.raise_for_status()
    user = response.json()
    response.close()

    print(f"\n>>> Greetings {user['name']}! <<<")


def get_user_anime_list():
    access_token = str(token['access_token'])
    url = 'https://api.myanimelist.net/v2/users/@me/animelist?fields=list_status'
    ani_list = []

    while url != "":
        response = requests.get(url, headers = {
            'Authorization': f'Bearer {access_token}'
            })
        
        response.raise_for_status()
        ani_list_json = response.json()
        response.close()

        with open('ani_list_json.json', 'w') as file:
            json.dump(ani_list_json, file, indent = 4)
            if DEBUG:
                print('ani_list saved in "ani_list_json.json"')

        with open('ani_list_json.json', 'r') as file:
            data = json.load(file)
            if DEBUG:
                print(f"Data in file is saved as {type(data)} type")

        ani_list.extend(data["data"])

        pageing = data["paging"]
        try:
            url = pageing["next"]
        except KeyError:
            url = ""
        except:
            print("Something went wrong!")

        if DEBUG:
            print(f"Next page page is {url}")
        

    if not DEBUG:
        if os.path.exists('ani_list_json.json'):
            os.remove('ani_list_json.json')
    
    ## TODO: Convert ani_list to Dataframe
    ani_df = pd.json_normalize(ani_list)
    return ani_df



def OAuth2():
    code_verifier = code_challenge = get_new_code_verifier()
    print_new_authorisation_url(code_challenge)

    # TODO: Callback
    global token
    authorisation_code = input('Copy-paste the Authorisation Code found in url following http://localhost/oauth?code= \n: ').strip()
    token = generate_new_token(authorisation_code, code_verifier)

    print_user_info()


if DEBUG:
    OAuth2()
    ani_df = get_user_anime_list()

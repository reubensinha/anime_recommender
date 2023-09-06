import MAL_my_data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau

# CONSTANTS
CHECKPOINT_FILEPATH = './weights/weights.h5'
MY_ID = 0
NAME = MAL_my_data.get_my_info()['name']
DEBUG = 1
if DEBUG:
    SEED = 73
else:
    SEED = None


# Global variables for RecommenderNet func.
n_users = 0
n_animes = 0


def combine_user_list(my_list, user_list):
    rating_list = my_list['anime_id', 'rating']
    rating_list['user_id'] = MY_ID

    rating_list = pd.concat([rating_list, user_list])

    return rating_list


def scale_ratings(rating_list):
    # Scale Ratings between 0 and 1
    min_rating = min(rating_list['rating'])
    max_rating = max(rating_list['rating'])
    rating_list['rating'] = rating_list["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values.astype(np.float64)

    AvgRating = np.mean(rating_list['rating'])

    if DEBUG:
        print(f"Average Rating: {AvgRating}")
    
    return rating_list


def remove_duplicates(rating_list):
    duplicates = rating_list.duplicated()
    if duplicates.sum() > 0:
        if DEBUG:
            print(f'> {duplicates.sum()} duplicates')

        rating_list = rating_list[~duplicates]

    if DEBUG:
        print(f'> {rating_list.duplicated().sum()} duplicates')
    
    return rating_list


def encode_categorical(rating_list):
    # Encoding categorical data
    user_ids = rating_list["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}
    rating_list["user"] = rating_list["user_id"].map(user2user_encoded)
    n_users = len(user2user_encoded)

    user_encoded = (user2user_encoded, user_encoded2user)

    anime_ids = rating_list["anime_id"].unique().tolist()
    anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
    anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}
    rating_list["anime"] = rating_list["anime_id"].map(anime2anime_encoded)
    n_animes = len(anime2anime_encoded)

    anime_encoded = (anime2anime_encoded, anime_encoded2anime)


    if DEBUG:
        print(f"Num of users: {n_users}, Num of animes: {n_animes}")
        print(f"Min rating: {min(rating_list['rating'])}, Max rating: {max(rating_list['rating'])}")

    return rating_list, user_encoded, anime_encoded

def RecommenderNet():
    # Embedding Layers
    embedding_size = 128
    
    user = Input(name = 'user', shape = [1])
    user_embedding = Embedding(name = 'user_embedding',
                       input_dim = n_users, 
                       output_dim = embedding_size)(user)
    
    anime = Input(name = 'anime', shape = [1])
    anime_embedding = Embedding(name = 'anime_embedding',
                       input_dim = n_animes, 
                       output_dim = embedding_size)(anime)
    
    #x = Concatenate()([user_embedding, anime_embedding])
    x = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, anime_embedding])
    x = Flatten()(x)
        
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
    
    return model


def lrfn(epoch):
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005

    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8

    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
    
    
def callbacks():
    lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

    model_checkpoints = ModelCheckpoint(filepath=CHECKPOINT_FILEPATH,
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)

    early_stopping = EarlyStopping(patience = 3, monitor='val_loss', 
                                mode='min', restore_best_weights=True)

    my_callbacks = [
        model_checkpoints,
        lr_callback,
        early_stopping,   
    ]

    return my_callbacks


def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))

    return weights


def find_similar_users(item_input, user_encoded, user_weights, n=10,return_dist=False, neg=False):
    user2user_encoded, user_encoded2user = user_encoded

    try:
        index = item_input
        encoded_index = user2user_encoded.get(index)
        weights = user_weights
    
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        if DEBUG:
            print(f'> users similar to #{item_input}')

        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]

            if isinstance(item_input, int):
                decoded_id = user_encoded2user.get(close)
                SimilarityArr.append({"similar_users": decoded_id, 
                                      "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", 
                                                        ascending=False)
        
        return Frame
    
    except:
        print(f'{NAME}!, Not Found in User list')


def get_user_preferences(user_id, rating_list, verbose=0):
    animes_watched_by_user = rating_list[rating_list.user_id==user_id]
    user_rating_percentile = np.percentile(animes_watched_by_user.rating, 75)
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
    top_animes_user = (
        animes_watched_by_user.sort_values(by="rating", ascending=False)
        .anime_id.values
    )
    

    # anime_df_rows = pd.DataFrame(columns= ['anime_id'])
    # for anime_id in top_animes_user:
    #     anime = MAL_my_data.get_anime_info(anime_id)

    #     anime_df_rows = pd.concat([anime_df_rows, anime["anime_id"]])
    anime_df_rows = top_animes_user['anime_id']

    
    if verbose != 0:
        print("> User #{} has rated {} movies (avg. rating = {:.1f})".format(
          user_id, len(animes_watched_by_user),
          animes_watched_by_user['rating'].mean(),
        ))
    
    
        
    return anime_df_rows


def get_recommended_animes(similar_users, rating_list, n=10):
    recommended_animes = pd.DataFrame(columns= ['anime_id', 'name', 'genres', 'synopsis', 'size'])
    anime_list = pd.DataFrame(columns= ['anime_id'])

    user_pref = get_user_preferences(MY_ID, rating_list, verbose=1)
    
    for user_id in similar_users.similar_users.values:  
        pref_list = get_user_preferences(int(user_id), rating_list, verbose=0)
        pref_list = pref_list[~ pref_list.anime_id.isin(user_pref.anime_id.values)]
        anime_list = pd.concat([anime_list, pref_list])
        
    sorted_list = anime_list.groupby(anime_list.columns.tolist(), as_index=False).size()
    sorted_list = sorted_list.sort_values(by= ['size']).head(n)

    sorted_matrix = sorted_list.to_numpy()

    for row in sorted_matrix:
        anime = MAL_my_data.get_anime_info(row[0])
        anime['size'] = [row[1]]

        recommended_animes = pd.concat([recommended_animes, anime])




    # for i, anime_id in enumerate(sorted_list.anime_id):  
    #     anime = sorted_list[sorted_list.anime_id == anime_id]      
    #     n_user_pref = anime.values[0][0]
    #     if isinstance(anime_id, int):
    #         try:
    #             anime['n'] = n_user_pref
    #             recommended_animes = pd.concat([recommended_animes, anime])
    #         except:
    #             pass



    
    return recommended_animes


def rec_anime(ani_list, my_list, user_list):
    ## ani_list, my_list and rating_list are all dataframes
    if DEBUG:
        print("rec_anime func.")
    
    rating_list = combine_user_list(my_list, user_list)
    

    ## Pre-Processing
    # Scale Ratings between 0 and 1
    rating_list = scale_ratings(rating_list)

    # Removing Duplicated Rows
    rating_list = remove_duplicates(rating_list)

    # Encoding categorical data
    rating_list, user_encoded, anime_encoded = encode_categorical(rating_list)

    # Shuffle
    rating_list = rating_list.sample(frac=1, random_state=SEED)

    X = rating_list[['user_id', 'anime_id']].values
    y = rating_list["rating"]

    # Split
    test_set_size = 10000 #10k for test set
    train_indices = rating_list.shape[0] - test_set_size 

    X_train, X_test, y_train, y_test = (
        X[:train_indices],
        X[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )

    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    if DEBUG:
        print(f'> Train set ratings: {len(y_train)}')
        print(f'> Test set ratings: {len(y_test)}')
    
    ## Build Model
    # Embedding Layers
    model = RecommenderNet()
    if DEBUG:
        model.summary()

    # Callbacks
    batch_size = 10000

    my_callbacks = callbacks()

    # Model training
    history = model.fit(
        x=X_train_array,
        y=y_train,
        batch_size=batch_size,
        epochs=20,
        verbose=1,
        validation_data=(X_test_array, y_test),
        callbacks=my_callbacks
    )

    model.load_weights(CHECKPOINT_FILEPATH)

    if DEBUG:
        plt.plot(history.history["loss"][0:-2])
        plt.plot(history.history["val_loss"][0:-2])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()


    # Extracting weights from model
    anime_weights = extract_weights('anime_embedding', model)
    user_weights = extract_weights('user_embedding', model)


    ## Finding Similar Users (User Based Recommendation)
    similar_users = find_similar_users(int(MY_ID), user_encoded, user_weights, n=5, neg=False)

    similar_users = similar_users[similar_users.similarity > 0.4]
    similar_users = similar_users[similar_users.similar_users != MY_ID]

    if DEBUG:
        similar_users.head(5)

    recommended_animes = get_recommended_animes(similar_users, rating_list, n=10)
    # getFavGenre(recommended_animes, plot=True)

    print(f'\n> Top recommendations for user: {NAME}')
    print(recommended_animes)
    

    return
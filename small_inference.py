import pandas as pd 
import numpy as np
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel


from helper import prepare_dataset 


@task(name="Loading movies data")
def load_movies():
    movies = pd.read_csv('ml-100k/u.item', sep='|',
                         usecols=range(2), names=['movie_id', 'title'],
                         encoding='latin-1')
    
    return movies

@task(name="Inferring with ONNX")
def recommendation(movies, num_movies, user_id=619, top_n=10):
    model = GizaModel(model_path="./small-model.onnx")
    all_movie_ids = np.arange(num_movies)  # Generate all item IDs
    user_ids = np.full((num_movies,), user_id, dtype=np.int64) 
    input_data = {"user_ids": user_ids, "movie_ids": all_movie_ids}

    preds = model.predict(input_feed=input_data, verifiable=False)

    sorted_movie_ids = np.argsort(preds)[::-1]
    top_n_movie_ids = sorted_movie_ids[:top_n]
    print(top_n_movie_ids)
    top_n_rec = movies[movies['movie_id'].isin(top_n_movie_ids.flatten())]
    print(top_n_rec)
    return top_n_rec

@action(name="Action: Recommending movies using ONNX model", log_prints=True)
def execution():
    ratings = prepare_dataset()
    movies = load_movies()    
    num_users = ratings['user_idx'].nunique()
    num_movies = ratings['movie_idx'].nunique()
    top_10_rec = recommendation(movies, num_movies, user_id=388)

    return top_10_rec

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="onnx-movie-recommendation")
    action_deploy.serve(name="onnx-movie-recommendation-deployment")


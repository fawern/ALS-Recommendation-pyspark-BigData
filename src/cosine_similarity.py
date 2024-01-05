import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

class ItemSimilarity:
    
    def __init__(self, movie_title):
        self.title_df = movie_title.toPandas()

    def item_similarity(self, model, item_id, item_number):
        item_factors = model.itemFactors
        item_factors.show(5)

        item_df = item_factors.toPandas()

        user_vectors = np.array(item_df[item_df['id'] == item_id]['features'].values[0])

        item_df['cosine_similarity'] = item_df['features'].apply(
            lambda x: cosine_similarity([x], [user_vectors])[0][0]
        )

        sorted_df = item_df.sort_values(by='cosine_similarity', ascending=False)

        sorted_df.reset_index(drop=True, inplace=True)

        sorted_df.columns = ['movie_id', 'vectors', 'cosine_similarity']

        sorted_df = sorted_df.loc[:item_number, :]

        return sorted_df

    def id_to_movie(self, movie_id):
        for index, id_ in enumerate(self.title_df['movie_id'].values):
            if movie_id == id_:
                return self.title_df['title'].values[index]

class UserSimilarity:

    def user_similarity(model, user_id, user_number):
        user_factors = model.userFactors
        user_factors.show(5)

        user_df = user_factors.toPandas()
        
        user_vectors = np.array(user_df[user_df['id'] == user_id]['features'].values[0])

        user_df['cosine_similarity'] = user_df['features'].apply(
            lambda x : cosine_similarity([x], [user_vectors])[0][0]
        )

        sorted_df = user_df.sort_values(by='cosine_similarity', ascending=False)

        sorted_df.reset_index(drop=True, inplace=True)

        sorted_df.columns = ['user_id', 'vectors', 'cosine_similarity']
        sorted_df = sorted_df.loc[:user_number, :]
        return sorted_df


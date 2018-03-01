import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data csv
# collect 4 or higher
data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))

# loss funciton measure difference between model prediction and desired output
# mimize during traing to increase prediction
# loss warp weighted approximate-rank pairwise
# help create recommendation by existing using rating pairs - gradient descent iterativly ways to improve over time
# takes user past (content based) and smiliar users rating (collaborative)
model = LightFM(loss='warp')
# train model
# epochs or runs
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    # number of users and movies
    n_users, n_times = data['train'].shape

    # generate recommendations
    for user_id in user_ids:
        # movies theay already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_times))

        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("\t Known positives:")
        for x in known_positives[:3]:
                print("\t\t%s" % x)
        print ("\tRecommended:")
        for x in top_items[:3]:
                print("\t\t%s" % x)
        
# three random user ids
sample_recommendation(model, data, [3, 25, 450])
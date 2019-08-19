import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import random

#only recomends movies with 4 stars or greater reating
data = fetch_movielens(min_rating=4.0)

#training adnd testting data
print(repr(data['train']))
print(repr(data['test']))

#creating model
model=LightFM(loss='warp')
#training
model.fit(data['train'],epochs=30,num_threads=2)

def sample_recommendations(model, data, user_ids):
    #number of users and movies in training data
    num_users, num_items=data['train'].shape

    #generate recommendation for each user
    for user_id in user_ids:

        #movies they like
        know_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies recommended by the model
        recommendations=model.predict(user_id, np.arange(num_items))

        #sorting from most liked to least
        top_recommendations=data['item_labels'][np.argsort(-recommendations)]

        #printing the results
        print("User %s" % user_id)
        print("     Know positives:")
        for x in know_positives[:3]:
            print("           %s" % x)
        print("      Recommended: ")
        for x in top_recommendations[:3]:
            print("          %s" % x)

random_IDs=[random.randint(1,400),random.randint(1,400),random.randint(1,400)]
sample_recommendations(model,data, random_IDs)
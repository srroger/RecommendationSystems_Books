#pip install numpy
#pip install scipy
#pip install lightfm ( allow to perform any number of popular recommendation libraries )

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from fetch_amazonratingonly import fetch_amazonratingonly

#fetch data and format it
data = fetch_amazonratingonly(min_rating=3.0)

#create model WARP
model = LightFM(loss='warp') #Weighted Approximate-Rank Pairwise
#train model
model.fit(data['matrix'], epochs=30, num_threads=2)

# model2 = LightFM(loss='warp-kos') #A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.
# #train model
# model2.fit(data['matrix'], epochs=30, num_threads=2)

# #('logistic', 'warp', 'bpr', 'warp-kos')

# model3 = LightFM(loss='bpr') #A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.
# #train model
# model3.fit(data['matrix'], epochs=30, num_threads=2)


# model4 = LightFM(loss='logistic') #A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.
# #train model
# model4.fit(data['matrix'], epochs=30, num_threads=2)

def sample_recommentation(model, data, user_ids):

	#number of users and movies in training data
	n_items = data['matrix'].shape[1]
	#n_users, n_items = data['matrix'].shape

	#generate recommendations for each user we input
	for user_id in user_ids:

		#movies they already like
		#known_positives = data['books'][ data['matrix'].tocsr() [user_id].indices ]#tocrs = Compressed Sparse Row Format

		#movies our model predicts they will like
		scores = model.predict(user_id, np.arange(n_items))
		top_scores = np.argsort(-scores)[:3]
		#rank them in order of most liked to least
		#top_items = data['books'][np.argsort(-scores)]

		#print out the results
		keys=list(data['users'].keys())  #in python 3, you'll need `list(i.keys())`
		values=list(data['users'].values())
		userAmazonName = keys[values.index(user_id)]  #'foo'
		print("	User %s, user amazon name %s" % (user_id, userAmazonName) )
		#print("		Known positives:")
		#for x in known_positives[:3]:
		#	print("				%s" % x)


		print ("		Recommended:")
		for x in top_scores[:3]:

			keys=list(data['books'].keys())  #in python 3, you'll need `list(i.keys())`
			values=list(data['books'].values())
			bookAmazonName = keys[values.index(x)]  #'foo'
			print("					%s, book amazon name %s" % (x, bookAmazonName))
		#for x in top_items[:3]:
		#	print("					%s" % x)

print('warp result:')
sample_recommentation(model, data, [0,1])

# print('k-OS WARP result:')
# sample_recommentation(model2, data, [3,25,450])

# print('bpr result:')
# sample_recommentation(model3, data, [3,25,450])

# print('logistic result:')
# sample_recommentation(model4, data, [3,25,450])

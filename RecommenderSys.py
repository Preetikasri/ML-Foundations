#Recommender Systems

graphlab.product_key.set_product_key('EAAC-EC40-83CA-48D4-8094-D9B4-C3AD-E2C2')

import graphlab

song_data = graphlab.SFrame('song_data.gl/')


#Exploring the music data
song_data.head()

#Histogram to show the song distiburtion 
graphlab.canvas.set_target('ipynb')
song_data['song'].show()

#Count the number of users
users = song_data['user_id'].unique()
len(users)

#create a song recommender System

train_data,test_data = song_data.random_split(0.8, seed = 0)

#1. Popularity based model
popularity_mod = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id = 'song')

#Now predict the recommendations

popularity_mod.recommend(users= [users[0]])

popularity_mod.recommend(users= [users[1]])

#everybody gets recommended the same thing. Obsviusly. 

#personalized model
person_mod = graphlab.item_similarity_recommender.create(train_data, user_id= 'user_id', item_id='song')

person_mod.recommend(users = [users[0]])

person_mod.recommend(users = [users[1]])

# Recommending similar items

person_mod.get_similar_items(['With Or Without You - U2'])

#Comparison of two models

 %matplotlib inline
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_mod, person_mod], user_sample=.05)

#Area under the curve fpr person_mod is much more, as expected. 

##Programming Assignment

Kanye_west_fans = song_data[song_data['artist'] == 'Kanye West']
NumKWFans = len(Kanye_west_fans.unique())
#3775

Foo_Fighters_fans = song_data[song_data['artist'] == 'Foo Fighters']
NumFFFans = len(Foo_Fighters_fans.unique())
#3429

Taylor_Swift_fans = song_data[song_data['artist'] == 'Taylor Swift']
NumTSFans = len(Taylor_Swift_fans.unique())
NumTSFans
#6227

Lady_GaGa_fans = song_data[song_data['artist'] == 'Lady GaGa']
NumLGFans = len(Lady_GaGa_fans.unique())
NumLGFans
#4129


#Most popular artists

popular_artist= song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})
popular_artist.sort('total_count', ascending = False)

#Kings Of Leon	43218 <- Most popular
#William Tabbert 14  <- Least popular

#Most recommended songs for first 10,000 users

subset_test_users = test_data['user_id'].unique()[0:10000]
recommended_songs = person_mod.recommend(subset_test_users,k=1)
table_most_recommended_song = recommended_songs.groupby(key_columns= 'song', operations={'count': graphlab.aggregate.COUNT()})
table_most_recommended_song.sort('count', ascending= False)

#Undo-Björk	434
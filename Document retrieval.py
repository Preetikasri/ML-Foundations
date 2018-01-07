elton = people[people['name']== 'Elton John']

elton.head()

elton[['word_count']].stack('word_count', new_column_name = ['word', 'count']).sort('count', ascending = False)
#the, in , and is most common 

elton[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidfCount']).sort('tfidfCount', ascending = False)
#furnish, elton, billboard

victoria = people[people['name']== '‘Victoria Beckham'] 
graphlab.distances.cosine(elton['tfidf'][0], victoria['tfidf'][0])
#0.9567006376655429


McCartney = people[people['name'] == 'Paul McCartney'] 
graphlab.distances.cosine(elton['tfidf'][0], McCartney['tfidf'][0])
#0.8250310029221779


Knn_Model_wc = graphlab.nearest_neighbors.create(people, features= ['word_count'], distance='cosine', label='name')
#elton-> Cliff Richard
#victoria -> Mary Fitzgerald (artist)

Knn_Model_wc.query(elton)
Knn_Model_wc.query(victoria)

Knn_Model_tfidf = graphlab.nearest_neighbors.create(people, features = ['tfidf'], distance='cosine', label='name')
Knn_Model_tfidf.query(elton)
Knn_Model_tfidf.query(victoria)
#elton->Rod Stewart
#victoria->David Beckham

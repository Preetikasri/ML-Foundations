import graphlab
products = graphlab.SFrame('amazon_baby.gl/')
products['wordCount'] = graphlab.text_analytics.count_words(products['review'])
products.head()
#defining function for awesome count
def awesome_count(word_count):
    if 'awesome' in word_count:
        return word_count['awesome']
    return 0

products['Count_of_awesome'] = products['wordCount'].apply(awesome_count)
products.head()
#list of the selected words
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

def great_count(word_count):
    if 'great' in word_count:
        return word_count['great']
    return 0

def fantastic_count(word_count):
    if 'fantastic' in word_count:
        return word_count['fantastic']
    return 0

def amazing_count(word_count):
    if 'amazing' in word_count:
        return word_count['amazing']
    return 0

def love_count(word_count):
    if 'love' in word_count:
        return word_count['love']
    return 0


def horrible_count(word_count):
    if 'horrible' in word_count:
        return word_count['horrible']
    return 0

#selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
def bad_count(word_count):
    if 'bad' in word_count:
        return word_count['bad']
    return 0


def hate_count(word_count):
    if 'hate' in word_count:
        return word_count['hate']
    return 0


def terrible_count(word_count):
    if 'terrible' in word_count:
        return word_count['terrible']
    return 0


def awful_count(word_count):
    if 'awful' in word_count:
        return word_count['awful']
    return 0


def wow_count(word_count):
    if 'wow' in word_count:
        return word_count['wow']
    return 0

#for all the words
products['count_of_wow'] = products['wordCount'].apply(wow_count)

for word in selected_words:
    print '{0}:{1}'.format(word, products['count_' + 'of_' + word].sum())

products = products[products['rating'] != 3]
products['sentiments'] = products['rating'] >= 4

for i in range(0,(len(selected_words)), 1):
        selected_words[i] = 'count_of_' + selected_words[i]

selected_words_model = graphlab.logistic_classifier.create(train_data, target='sentiments', 
                                                  features= selected_words, validation_set=test_data)

selected_words_model['coefficients']


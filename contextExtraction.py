'''
Sequoia hack nlp engine for context extraction from search queries.
'''

import nltk
entities=[]
sentence = "suggest me an ecommerce sites running in Mumbai which has a good customer service"
tokens = nltk.word_tokenize(sentence)
posTags = nltk.pos_tag(tokens)
chunked = nltk.chunk.ne_chunk(posTags)
namedEntities = []
location = None

for tree in chunked:
    # The way NER tagging is done is, it creates a tree off each sentence, each phrase being a 
    # represnetative of the tag
    if hasattr(tree, 'label'):
        label = tree.label()
        if label == 'GPE' or label == 'LOCATION':
            location = tree[0]
        break


def getLocation(chunked):
	'''
    Returns the location if at all it is specified in the chunked tags generated from nltk
	'''
    namedEntities = []
    location = None
    for tree in chunked:
        if hasattr(tree, 'label'):
            label = tree.label()
            if label == 'GPE' or label == 'LOCATION':
                location = tree[0][0]
    return location

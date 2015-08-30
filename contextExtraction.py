#!/usr/bin/env python

'''
Sequoia hack nlp engine for context extraction from search queries.
'''

import nltk
from urlparse import urlparse, parse_qs
import web
from nltk.corpus import wordnet


lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
urls = (
    '/getNounPhrases', 'GetNounPhrases',
    '/getLocation', 'GetLocation',
    '/getNormalizedWord', 'GetNormalizedWord',
    '/getSynonyms', 'GetSynonyms',
    '/getQuery', 'GetQuery'
)

filterTags = {
    "AvgCustRating" : ["rating", "star"],
    "Location" : ['location', 'place', 'city', 'country']
}

productTags = {
    "ecommerce" : ["e-commerce", "online" "retail", "shopping", "ecommerce"],
    "JavaScript Library" : ["JavaScript", "JS"]
}

sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
        '''

application = web.application(urls, globals())
web.config.debug = True

class GetQuery:
	def GET(self):
		params = web.input()
		return json.dumps(analyzeQuery(params.name))
class GetSynonyms:
    def GET(self):
        params = web.input()
        return getSynonyms(params.name)
class GetNounPhrases:
    def GET(self):
        params = web.input()
        return getNounPhrases(params.name)

class GetLocation:
    def GET(self):
        params = web.input()
        return getLocation(params.name)

class GetNormalizedWord:
    def GET(self):
        params = web.input()
        return normalise(params.name)



##### METHOD IMPLEMENTATIONS

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

def getNounPhrases(sentence):
    '''
    I have quickly copied this loosely from: https://gist.github.com/alexbowe/879414
     which cites S. N. Kim, T. Baldwin, and M.-Y. Kan. Evaluating n-gram
      based evaluation metrics for automatic keyphrase extraction. 
      Technical report, University of Melbourne, Melbourne 2010.
    '''
    nounPhrases = []
    # Used when tokenizing words
    
    '''
    #Taken from Su Nam Kim Paper...
    '''
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    npChunker = nltk.RegexpParser(grammar)
    toks = nltk.regexp_tokenize(sentence, sentence_re)
    posTokens = nltk.tag.pos_tag(toks)
    tree = npChunker.parse(posTokens)
    terms = get_terms(tree)
    for term in terms:
        nounPhrases.append(' '.join(term))
    return nounPhrases

def getLocation(sentence):
    ''' 
    Returns the location if at all it is specified in the chunked tags generated from nltk
    '''
    chunked = getChunkTrees(sentence)
    location = None
    for tree in chunked:
        if hasattr(tree, 'label'):
            label = tree.label()
            if label == 'GPE' or label == 'LOCATION':
                location = tree[0][0]
    return location

def getChunkTrees(sentence):
    ''' 
    Returns the parse tree for the sentence
    '''
    nounPhrases = []
    
    toks = nltk.regexp_tokenize(sentence, sentence_re)
    posTokens = nltk.tag.pos_tag(toks)
    chunked = nltk.chunk.ne_chunk(posTokens)
    return chunked


def getSynonyms(word):
    synonyms = set()
    synsets = wordnet.synsets(word)
    if len(synsets) > 2:
        limit = 2
    else:
        limit = len(synsets)
    for synset in synsets[0:limit]:
        synonyms.update(synset.lemma_names())
    return synonyms

def getHyponyms(word):
    hyponyms = set()
    synsets = wordnet.synsets(word)
    if len(synsets) > 2:
        limit = 2
    else:
        limit = len(synsets)
    for synset in synsets[0:limit]:
        hyponymSynsets = synset.hyponyms()
        if len(synsets) > 2:
            limit = 2
        else:
            limit = len(synsets)
        for hyponymSynset in hyponymSynsets[0:limit]:
            hyponyms.update(hyponymSynset.lemma_names())
    return hyponyms

def getHypernyms(word):
    hypernyms = set()
    synsets = wordnet.synsets(word)
    if len(synsets) > 2:
        limit = 2
    else:
        limit = len(synsets)
    for synset in synsets[0:limit]:
        hypernymSynsets = synset.hypernyms()
        if len(synsets) > 2:
            limit = 2
        else:
            limit = len(synsets)
        for hypernymSynset in hypernymSynsets[0:limit]:
            hypernyms.update(hypernymSynset.lemma_names())
    return hypernyms

def analyzeQuery(query):
    answerhash = {}
    answerhash['location'] = getLocation(query)
    answerhash['filter'] = set()
    answerhash['productType'] = set()
    nounPhrases = getNounPhrases(query)
    for phrase in nounPhrases:
        for key in filterTags:
            for term in filterTags[key]:
                if(checkCongruence(term, phrase)):
                    answerhash['filter'].add(key)
        for key in productTags:
            for term in productTags[key]:
                if(checkCongruence(term, phrase)):
                    answerhash['productType'].add(key)
    answerhash['filter'] = list(answerhash['filter'])
    answerhash['productType'] = list(answerhash['productType'])
    return answerhash

def checkCongruence(term, phrase):
    for word in nltk.regexp_tokenize(phrase, sentence_re):
        similarToTerm = getSynonyms(term) | getHypernyms(term) | getHyponyms(term) | set([term])
        similarToWord = getSynonyms(word) | getHypernyms(word) | getHyponyms(word) | set([word])
        similarToTerm = set([normalise(text.replace('_', ' ').lower()) for text in similarToTerm])
        similarToWord = set([normalise(text.replace('_', ' ').lower()) for text in similarToWord])
        if any(st in similarToTerm for st in similarToWord):
            return True
    return False

if __name__ == '__main__':
    application.run()
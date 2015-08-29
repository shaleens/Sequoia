'''
Sequoia hack nlp engine for context extraction from search queries.
'''

import nltk



lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()


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
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    '''
	#Taken from Su Nam Kim Paper...
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
    	nounPhrases.extend(term)
    return nounPhrases

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

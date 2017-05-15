#************************************************************************
#      __   __  _    _  _____   _____
#     /  | /  || |  | ||     \ /  ___|
#    /   |/   || |__| ||    _||  |  _
#   / /|   /| ||  __  || |\ \ |  |_| |
#  /_/ |_ / |_||_|  |_||_| \_\|______|
#    
# 
#   Written by < Daniel L. Marino (marinodl@vcu.edu) > (2016)
#
#   Copyright (2016) Modern Heuristics Research Group (MHRG)
#   Virginia Commonwealth University (VCU), Richmond, VA
#   http://www.people.vcu.edu/~mmanic/
#   Do not redistribute without author's(s') consent
#  
#   Any opinions, findings, and conclusions or recommendations expressed 
#   in this material are those of the author's(s') and do not necessarily 
#   reflect the views of any other entity.
#  
#   ***********************************************************************
#
#   Description: Definition of a vocabulary using stemmed words extracted from a 
#                given string
#
#   ***********************************************************************


from nltk.stem.porter import PorterStemmer
import numpy as np
import pickle
import operator

class Vocabulary(object):     
    def __init__(self, text_list, max_size= None):
        ''' Builds a dictionary with the stemmed words extracted from a given list of words
        inputs:
            text_list: text with the words splitted in a list
        outputs:
            key_dict: dictionary with stemmed words as keys, and an index as values. 
        '''
        self.stemmer = PorterStemmer()
        # extracting the set of words in the string     
        self.key_dict = dict()
        key_hist = dict()
        
        key= 1
        for word in text_list:
            word = self.stemmer.stem(word) # this is the porter stemmer
            if word not in self.key_dict:
                self.key_dict[word] = key
                key_hist[word] = 1
                key += 1
            else:
                key_hist[word] += 1
        
        # Delete words with low appareances
        if max_size is not None:
            sorted_hist = sorted(key_hist.items(), key=operator.itemgetter(1), reverse= True)
            
            self.key_dict= dict((d[0], i) for d, i in zip(sorted_hist[:max_size], range(1,max_size+1)))
            
        self.key_dict['N/A'] = 0        
        
        self.vocabulary_size= len(self.key_dict)
        self.key_list = sorted(self.key_dict.items(), key=operator.itemgetter(1))
        
    def text2keys(self, text_list, ignore_unknown= False):
        text_id= list()

        for word in text_list:
            word = self.stemmer.stem(word)
            if word not in self.key_dict:
                if not ignore_unknown:
                    text_id.append(0)
            else:
                text_id.append(self.key_dict[word])

        return text_id

    def keys2text(self, text_id):
        text= ''
        for i in text_id:
            try:
                text= text + self.key_list[i][0]+' '
            except IndexError:
                text= text + 'N/A '
            
            #else:
            #    raise ValueError('Dictionary should be constructed with increasing values')
        return text
    
    def prob2char(self, probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation.
        It returns the character corresponding to the higest probability """
        return self.keys2text([c for c in np.argmax(probabilities, 1)])
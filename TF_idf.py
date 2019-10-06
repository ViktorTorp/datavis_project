import math as math
from collections import defaultdict

class TFIDF:
    """
    A class used to calc tf-idf scores
    ...

    Attributes
    ----------
    bows : list
        list of dicts with bag of word representationf
    names : list 
        list of names for domains

    Methods
    -------
    calc_tf(bow)
        Calculates term frequency
    calc_idf(bows=None)
        Calculates Inverse document frequency
    calc_tfidf(bow, idf)
        Calculates tf-idf
    _calc_tfidfs
        Calculates selveral tf-idfs
    fit(bows=None, names=None)
        fit tf-idfs module
    transform()
        transforms all bows to tf-idfs
    fit_transform(bows=None, names=None)
        fit and transforms

    """
    def __init__(self, bows=None, names=None):
        """
        Parameters
        ----------
        bows : list
            list of dicts with bag of word representationf
        names : list 
            list of names for domains
        """
        self.bows = bows
        self.names = names
        self.fitted = False

    def calc_tf(self, bow):
        """ Calculates tf

        Parameters
        ----------
        bow : dict
            dict with bag of word representationf

        Returnes
        --------
        tf : dict
            Tf dict
        """
        tf = defaultdict(int)
        num_words = sum(bow.values())
        for word, count in bow.items():
            tf[word] = count/num_words
        return tf

    def calc_idf(self, bows=None):
        """ Calculates idf

        Parameters
        ----------
        bows : list(optional)
            list of dicts with bag of word representation

        Returnes
        --------
        idf : dict
            idf dict
        """
        if not bows:
            bows = self.bows
        idf = defaultdict(int)
        num_docs = len(bows)
        for bow in bows:
            for word, count in bow.items():
                if count > 0:
                    idf[word] += 1
        for word, count in idf.items():
            idf[word] = math.log10(num_docs/count)
        return idf


    def calc_tfidf(self, bow, idf):
        """ Calculates tf-idf

        Parameters
        ----------
        bow : dict
            dict with bag of word representation
        idf : dict
            dict woth words idf

        Returnes
        --------
        tfidf : dict
            tf-idf dict
        """
        tfidf = defaultdict(int)
        for word, count in bow.items():
            tfidf[word] = count * idf[word]
        return tfidf


    def _calc_tfidfs(self):
        """ Calculates tf-idfs

        Returnes
        --------
        tfidf : dict
            dict with a tf-idf dict for each domain
        """
        tfidf = {}
        for i in range(len(self.bows)):
            tfidf[self.names[i]] = self.calc_tfidf(self.bows[i],self.idf)
        return tfidf

    def fit(self, bows=None, names=None):
        """ Fits tf-idf model

        Parameters
        ----------
        bows : list(optional)
            list of dicts with a tf-idf dict for each domain
        names : list(optional)
            list of names for each domain
        """
        if bows:
            self.bows = bows
        else:
            assert self.bows != None
        if names:
            self.names = names
        else:
            assert self.names != None

        self.tfs = [self.calc_tf(bow) for bow in self.bows]
        self.idf = self.calc_idf()
        self.fitted = True

    def transform(self):
        """ Transforms tf-idf model
        Returns
        -------
        tf-idfs : dict
            dict with a tf-idf dict for each domain
        """
        if self.fitted == False:
            print("Model has not been fitted yet")
            return
        self.tfidfs = self._calc_tfidfs()
        return self.tfidfs

    def fit_transform(self, bows, names):
        """ Fits and transforms tf-idf model
        
        Parameters
        ----------
        bows : list(optional)
            list of dicts with a tf-idf dict for each domain
        names : list(optional)
            list of names for each domain

        Returns
        -------
        tf-idfs : dict
            dict with a tf-idf dict for each domain
        """
        if bows:
            self.bows = bows
        else:
            assert self.bows != None
        if names:
            self.names = names
        else:
            assert self.names != None
        self.fit(self.bows, self.names)
        return self.transform()



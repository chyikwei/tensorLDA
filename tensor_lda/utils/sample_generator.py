"""Simulate the generative process of LDA and generate corpus based on it
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import poisson

from sklearn.utils import check_random_state
from six.moves import xrange


class LdaSampleGenerator(object):
    """Generate LDA samples

    Parameters
    ----------
    n_topics : int
        Number of topics
    
    n_words : int
        Number of words in corpus

    min_doc_size : int
        Min word count in a document
    
    mean_doc_size : int
        Mean word count in a document
    
    doc_topic_prior : double
        Uniform Dirichlet prior of a document
    
    topic_word_prior : double
        Uniform Dirichlet prior of a topic

    mean_doc_size: int
        Mean Value if word count in each document


    Attributes
    ----------
    topic_word_distr_ : array, [n_topics, n_words]
       Topic word distribution.

    """

    def __init__(self, n_topics, n_words, min_doc_size,
                 mean_doc_size, doc_topic_prior,
                 topic_word_prior, random_state=None):
        self.n_topics = n_topics
        self.n_words = n_words
        self.min_doc_size = min_doc_size
        self.mean_doc_size = mean_doc_size
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.random_state = random_state

        self.random_state_ = check_random_state(self.random_state)
        # hidden variables
        self.topic_word_prior_ = np.repeat(topic_word_prior, n_words)
        # (n_topics, n_words)
        self.doc_topic_prior_ = np.repeat(self.doc_topic_prior, n_topics)
        self.topic_word_distr_ = self.random_state_.dirichlet(
            self.topic_word_prior_, n_topics)
        
        
    def generate_documents(self, n_docs):
        """Generate Random doc-words Matrix

        Parameters
        ----------
        n_docs : int
            number of documents

        Return
        ------
        doc_word_mtx : sparse matrix, [n_docs, n_words]
            document words matrix

        """
        rs = self.random_state_
        n_topics = self.n_topics
        n_words = self.n_words
        docs_size = poisson.rvs(mu=(self.mean_doc_size - self.min_doc_size),
                                size=n_docs, random_state=rs)
        docs_size += self.min_doc_size
        doc_prior = np.repeat(self.doc_topic_prior, n_topics)
        # (n_docs, n_topics)
        docs_distr = rs.dirichlet(doc_prior, n_docs)

        rows = []
        cols = []
        for i in xrange(n_docs):
            word_dist = np.dot(self.topic_word_distr_.T, docs_distr[i, :])
            word_idx = rs.choice(n_words, docs_size[i], p=word_dist, replace=True)
            rows = np.append(rows, np.repeat(i, docs_size[i]))
            cols = np.append(cols, word_idx)
        data = np.ones(len(rows))
        doc_word_mtx = coo_matrix((data, (rows, cols)),
                                  shape=(n_docs, n_words)).tocsr()
        return docs_distr, doc_word_mtx

import numpy as np

from six.moves import xrange
from numpy.testing import assert_equal
from nose.tools import assert_true
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal

from tensor_lda.utils.sample_generator import LdaSampleGenerator


def test_lda_sample_generator():
    n_topics = 20
    n_words = 500
    mean_words = 10
    min_words = 3
    doc_topic_prior = 1.
    topic_word_prior = 1.
    n_doc = 10000

    gen = LdaSampleGenerator(n_topics=n_topics, n_words=n_words,
                             min_doc_size=min_words,
                             mean_doc_size=mean_words,
                             doc_topic_prior=doc_topic_prior,
                             topic_word_prior=topic_word_prior,
                             random_state=0)

    docs_distr, doc_word_mtx = gen.generate_documents(n_doc)
    topic_word_distr = gen.topic_word_distr_

    # check shape
    assert_equal((n_doc, n_topics), docs_distr.shape)
    assert_equal((n_doc, n_words), doc_word_mtx.shape)
    assert_equal((n_topics, n_words), topic_word_distr.shape)

    # check word count
    word_cnts = doc_word_mtx.sum(axis=1)
    assert_equal(word_cnts.shape[0], n_doc)
    assert_true(np.all(word_cnts >= min_words))
    assert_true(abs(word_cnts.mean() - mean_words) < 0.1)

    # check doc distribution
    assert_array_almost_equal(np.ones(n_doc), docs_distr.sum(axis=1))
    doc_dirichelet_mean = gen.doc_topic_prior_ / float(gen.doc_topic_prior_.sum())
    assert_array_almost_equal(doc_dirichelet_mean, docs_distr.mean(axis=0), decimal=3)

    # check topic-word distr
    assert_array_almost_equal(np.ones(n_topics), topic_word_distr.sum(axis=1))
    word_dirichelet_mean = gen.topic_word_prior_ / float(gen.topic_word_prior_.sum())
    # set decimal to 2 since variance is large
    assert_array_almost_equal(word_dirichelet_mean, topic_word_distr.mean(axis=0), decimal=2)

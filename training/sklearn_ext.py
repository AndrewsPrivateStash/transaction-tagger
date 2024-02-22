import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from operator import itemgetter


class PipeSpec(Pipeline):
    """ wraps Pipeline class to add feature name and vocabulary extraction """

    def get_feature_names(self):
        try:
            return self._final_estimator.get_feature_names()
        except AttributeError:
            raise Exception(f'could not fetch feature names for: {self._final_estimator}')

    @property
    def vocabulary_(self):
        try:
            return self._final_estimator.vocabulary_
        except AttributeError:
            raise Exception(f'could not fetch vocabulary for: {self._final_estimator}')


class ColsExtractor(TransformerMixin):
    """ extracts columns from raw data """

    def __init__(self, cols, encoding):
        """
        extracts column(s) from raw data
        cols : list
            list of column names
        encoding : str
            {scale, bow, one-hot}
            scale, transpose the vectors into column vectors for minmax scaling [a, b] -> [[a], [b]]
            bow, single column
            one-hot, columns returned as dictionary
        """
        self.cols = cols
        self.encoding = encoding

    def fit(self, x, y=None):
        return self

    def transform(self, records):

        # encoding names in config are invariant
        if self.encoding == 'minmax':
            # minmax scalar requires column vectors of floats
            return np.array([[float(rec[c]) for c in self.cols] for rec in records])
        elif self.encoding == 'bow':
            # return list for single collumn used in bag of words transform
            return np.array([rec[self.cols[0]] for rec in records])
        elif self.encoding == 'one-hot':
            # list of dictionary of columns for dictvectorizor
            return [{x:rec[x] for x in self.cols} for rec in records]
        else:
            # if no encoding specified return matrix of columns selected
            return np.array([[rec[c] for c in self.cols] for rec in records])


class DictVectVoc(DictVectorizer):
    """ DictVectorizer subclass which handles a vocabulary of one-hot features """

    def __init__(self, vocab=None):
        """
        controls the output of the categorical fields based on a passed vocabulary
        vocab : dict
            dictionary of model vocabulary (local to categorical fields)
        """
        super().__init__(dtype=np.float64, separator="=", sparse=True, sort=True)
        self.loc_vocab=vocab

    def rebuild_csr(self, pre_matrix):
        # re-build the CSR matrix after construction
        # mat returned may be missing columns or contain columns not in vocab
        # build empty array, populate columns in proper position and encode as csr
        
        # mapping for each column in current mat (in vocab: bool, value: str, column of current: int)
        mask = [(t in self.loc_vocab, t, i) for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))]

        # build empty array having row count of returned mat and column count of vocab
        tmp_arr = np.zeros((pre_matrix.shape[0],len(self.loc_vocab)), dtype=np.float64)
        mat_dense = pre_matrix.toarray()   # let's not worry about memory at the moment

        # copy columns from mat to tmp using mask indecies
        for cp, k, cur_col in mask:
            if cp:
                # copyto(dest, src, opts)
                np.copyto(tmp_arr[:,self.loc_vocab[k]], mat_dense[:,cur_col], casting='no')

        # construct csr matrix from populated tmp_arr
        ret_mat = csr_matrix(tmp_arr)
        self.vocabulary_ = self.loc_vocab
        return ret_mat

    # override base method to handle vocabulary
    def fit_transform(self, X, y=None):
        mat = super().fit_transform(X)
        if not self.loc_vocab:
            return mat
        else:
           return self.rebuild_csr(mat)


class ScaleFilter(TransformerMixin):
    """ trasnformer to handle scaled numeric feature columns and vocabularies """

    def __init__(self, cols, vocab=None):
        """
        controls the output of the minmax amount fields based on a passed vocabulary
        cols : list
            list of column names to supply to get feature names
        vocab : dict
            dictionary of model vocabulary
        """
        self.cols = cols
        self.vocabulary_ = {}
        self.exclude = False
        self.apply_vocab = False

        # null vocab, create dict of values in column order
        if vocab is None:
            for col, key in enumerate(cols):
                self.vocabulary_.update({
                    key : col
                })

        else:
            # passed vocab is empty, then exclude
            if not vocab:
                self.exclude = True

            # passed vocab is not empty
            else:
                self.vocabulary_ = vocab
                self.apply_vocab = True


    def map_vocabulary(self, cur_cols, vocab, recs):
        """ take scaled records and apply vocabulary (only vocab columns returned in vocab order) """
        
        # zeros matrix to populate
        zero_mat = np.zeros((len(recs), len(vocab)), dtype=np.float64)
        vals = np.array(recs)

        for i, col in enumerate(cur_cols):
            if col in vocab:
                np.copyto(zero_mat[:,vocab[col]], vals[:,i])

        return csr_matrix(zero_mat)

    def get_feature_names(self):
        if self.exclude:
            return None
        elif self.apply_vocab:
            return [k for k in self.vocabulary_.keys()]
        else:
            return self.cols

    def fit(self, x, y=None):
        return self

    def transform(self, records):
        # return nothing if excluded in vocabulary
        if self.exclude:
            return None
        # return mapped values if vocab provided
        elif self.apply_vocab:
            return self.map_vocabulary(self.cols, self.vocabulary_, records)
        else:
            return csr_matrix(records)

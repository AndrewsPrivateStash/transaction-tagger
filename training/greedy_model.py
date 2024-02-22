from training.model import TaggingModel
from training.data import ModelData

import logging

"""
    Greedy models do not train NULLs as first-class values but rather
    treat NULLs as proper unknows based on probability thresholds.

    This is accomplished by training a model using only tagged records over a wider timeframe.
    However, regular NULL-trained models can also use the Greedy sub-class as it only adds 
    a more flexible predict method.
    
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GreedyTaggingModel(TaggingModel):
    """ TaggingModel subclass to manage greedy models
        predict_one/many handled by super
        new method to return top_n and all_above a threshold
    """

    def __init__(self):
        super().__init__()

    def _makeDat(self, record):
        """ create encoded data from record """
        dat = ModelData(vocab=self.model_vocabulary, config_obj=self.raw_config)
        dat.load_record(record=record)
        dat.fit_transform()
        return dat

    def predict_n(self, record, top_n=1, thr=None):
        """ predict top N tags, or all tags above a proba threshold
            record : single record dictionary
            top_n : the number of tags to return sorted by proba desc
            thr : the probability threshold above which all tags are returned. Overrides top_n
        """

        if not self.model:
            logger.warning('predict_n called without a model loaded, aborting..')
            return
 
        try:
            if not self.raw_config:
                logger.warning('model loaded has an empty config attribute, aborting..')
                return
        except AttributeError:
            logger.warning('model loaded does not contain a config attribute, aborting..')
            return

        try:
            if not self._version:
                logger.warning('model loaded has an empty _version attribute, aborting..')
                return
        except AttributeError:
            logger.warning('model loaded does not contain a _version attribute, aborting..')

        datMat = self._makeDat(record)  # create data object from passed record
        prob_mat = self.model.predict_proba(datMat.xmat)  # m-samples by n-classes, elements are probas of class matching sample

        if thr:
            best_idxs = prob_mat.argsort(axis=1)[0][::-1]
            # take first row of m,n matrix returned by argsort (given we want a vector), then reverse order for descending proba sort
            best_idxs = [ix for ix in best_idxs if prob_mat[0][ix] > thr]
            # filter the indices such that each is associated with a proba greater than the passed threshold
        else:
            best_idxs = prob_mat.argsort(axis=1)[0][::-1][:top_n]
            # take first row of m,n matrix returned by argsort, reverse, then return top_n elements
        tags = self.model.classes_[best_idxs]  # extract classes in best_idxs order
        probs = prob_mat[0][best_idxs] # extract probabilities in best_idxs order

        out_values = []
        for i, (t, p) in enumerate(zip(tags, probs)):
            out_values.append({
                'rank': i+1, 
                'tag': t,
                'prob': p,
                'version': self._version
            })
    
        return out_values

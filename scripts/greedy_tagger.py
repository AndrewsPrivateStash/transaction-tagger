# pylint: disable=import-error
import sys 
from os import path

projRoot = path.join(path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports (horible module management, ugh)

from training.data import ModelData
from training.model import Train
from training.model import TaggingModel
import csv, pprint as pp, json

"""
this model is intended to tag unseen strings in a simplified manner and always seek to produce a tag
it optionally can return multiple tags (the top n) to choose from or all tags above a proba threshold
it works on the desc string alone and does not train nulls as first class values
"""

CFG = path.join(projRoot,'config/data_config_greedy.json')
BND = path.join(projRoot,'data/brands.json')
DATAPATH = path.join(projRoot,'data/dat_greedy.csv')
MODPATH = path.join(projRoot,'data/model_greedy.gz')


def build_model(config=CFG, data=DATAPATH, mod_path=MODPATH):
    """ train greedy model """
    dat = ModelData(config_path=config)
    dat.load_csv(data)
    dat.fit_transform()

    tr = Train()
    tr.load_data(dat)
    tr.train_model(c_penalty=0.1, weights='balanced', output_path=mod_path)

def getBrands(brand_path=BND):
    with open(brand_path, 'r') as f:
        return json.load(f)

class GreedyMod:
    """ wraps TaggingModel and related methods to perform one-off tagging """

    def __init__(self, model=MODPATH):
        self.greedyModel = TaggingModel()
        self.greedyModel.load_model(model)
        self.brands = getBrands()

    def makeDat(self, record):
        dat = ModelData(vocab=self.greedyModel.model_vocabulary, config_path=CFG)
        dat.load_record(record=record)
        dat.fit_transform()
        return dat

    def testModel(self, data_path):
        dat = ModelData(vocab=self.greedyModel.model_vocabulary, config_path=CFG)
        dat.load_csv(data_path)
        dat.fit_transform()

        test_results = self.greedyModel.test(data_obj=dat, thr=0.49)
        pp.pprint(test_results)

    def pred(self, desc, top_n=1, thr=None):
        """ predict top N brands using transaction description """
        record = {
            'tx_desc': desc
        }
        datMat = self.makeDat(record)

        prob_mat = self.greedyModel.model.predict_proba(datMat.xmat)
        if thr:
            best_idxs = prob_mat.argsort(axis=1)[0][::-1]
            best_idxs = [ix for ix in best_idxs if prob_mat[0][ix] > thr]
        else:
            best_idxs = prob_mat.argsort(axis=1)[0][::-1][:top_n]

        tags = self.greedyModel.model.classes_[best_idxs]
        probs = prob_mat[0][best_idxs]

        out_values = []
        for i, (t, p) in enumerate(zip(tags, probs)):
            out_values.append({
                'rank': i+1, 
                'tag': t,
                'brand' : self.brands.get(t, "NULL"),
                'prob': p
            })
        if not out_values:
            out_values = ['NULL']

        return out_values


def main():
    print("called directly, but inteded to be interactive in interpreter")
    return


if __name__ == '__main__':
    main()
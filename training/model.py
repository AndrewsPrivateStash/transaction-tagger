from training.data import ModelData

import numpy as np
from joblib import dump, load
import datetime, logging
import multiprocessing as mp
import re

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# collection of global regexp objects that override the model proper
# applied to the tx_desc and if match then defines a matched tag bipasing SVM model with prob = 1
PATTERNS = {
    "50f8f8d9-de80-48f2-b4b6-c534adb68170": {
        "regexp": re.compile(r"(hll|hello)\s*fre?(sh)?", re.I),
        "brand_name": "HelloFresh"
    },
    "ecf5eba3-f5e8-4b45-a132-768b3f175e18": {
        "regexp": re.compile(r"willamette\s*val.*(v.*|turner)", re.I),
        "brand_name": "Willamette Valley Vineyards"
    }
}


def get_environ_details():
    from platform import python_version
    import sklearn, numpy, joblib, scipy
    return {
        "python_version": python_version(),
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": numpy.__version__,
        "joblib_version": joblib.__version__
    }    

# global reporting functions
def metric_roll(pred_dict, class_dim=False):
    """
    takes prediction dictionary and returns prediction metrics (optionally by class)
    pred_dict : dict
        prediction results
    class_dim : bool
        toggle to return metrics by class verus global metrics
    """

    def pop_class(res_dict, actual, metric):
        if actual not in res_dict:
            res_dict.update({
                actual : {
                    'cnt' : 0,
                    'TN' : 0,
                    'TP' : 0,
                    'FuzzP' : 0,
                    'NoTag' : 0,
                    'FP' : 0,
                    'rej' : 0,
                    'dis' : 0
                }
            })

        if metric not in res_dict[actual]:
            raise Exception(f'unexpected metric: {metric}')
        
        res_dict[actual][metric] += 1


    # global counts
    TN = 0
    TP = 0
    FuzzP = 0
    NoTag = 0
    FP = 0
    rej = 0
    dis = 0
    cnt = len(pred_dict)
    result = {}

    # container for the class roll
    if class_dim:
        class_res = {}
        classes = {}

    for rec in pred_dict:
        pred = rec['tag']
        act = rec['y_act']

        if class_dim:
            pop_class(class_res, act, 'cnt')  # class count

        if pred == '' and act == '':
            TN += 1
            if class_dim:
                pop_class(class_res, act, 'TN')
        if pred != '' and act != '' and pred == act:
            TP += 1
            if class_dim:
                pop_class(class_res, act, 'TP')
        if pred != '' and act == '':
            FuzzP += 1
            if class_dim:
                pop_class(class_res, act, 'FuzzP')
        if pred == '' and act != '':
            NoTag += 1
            if class_dim:
                pop_class(class_res, act, 'NoTag')
        if pred != '' and act != '' and pred != act:
            FP += 1
            if class_dim:
                pop_class(class_res, act, 'FP')
        if 'rejected' in rec:
            rej += 1
            if class_dim:
                pop_class(class_res, act, 'rej')
        if pred != act:
            dis += 1
            if class_dim:
                pop_class(class_res, act, 'dis')

    glb_res = {  
        'Records' : cnt,
        'True Negative' : {'cnt' : TN, 'rate' : TN / cnt},
        'True Positive' : {'cnt' : TP, 'rate' : TP / cnt},
        'Fuzzy Positive' : {'cnt' : FuzzP, 'rate' : FuzzP / cnt},
        'No Tag' : {'cnt' : NoTag, 'rate' : NoTag / cnt},
        'False Positive' : {'cnt' : FP, 'rate' : FP / cnt},
        'Rejected' : {'cnt' : rej, 'rate' : rej / cnt},
        'Total Disagreement' : {'cnt' : dis, 'rate' : dis / cnt}
    }
   
    if class_dim:
        for cls_name, mets in class_res.items():
            classes.update({
                cls_name : {
                    'Records' : mets['cnt'],
                    'True Negative' : {'cnt' : mets['TN'], 'rate' : mets['TN'] / mets['cnt']},
                    'True Positive' : {'cnt' : mets['TP'], 'rate' : mets['TP'] / mets['cnt']},
                    'Fuzzy Positive' : {'cnt' : mets['FuzzP'], 'rate' : mets['FuzzP'] / mets['cnt']},
                    'No Tag' : {'cnt' : mets['NoTag'], 'rate' : mets['NoTag'] / mets['cnt']},
                    'False Positive' : {'cnt' : mets['FP'], 'rate' : mets['FP'] / mets['cnt']},
                    'Rejected' : {'cnt' : mets['rej'], 'rate' : mets['rej'] / mets['cnt']},
                    'Total Disagreement' : {'cnt' : mets['dis'], 'rate' : mets['dis'] / mets['cnt']}
                }
            })
        
        result.update({
            'GLOBAL' : glb_res,
            'CLASSES' : classes
            })
    else:
        result = glb_res
        
    return result


def rpt_filter(class_roll, n=None, threshold=0, desc=True):
    """
    find the n-worst/best classes
    class_roll : dict
        class prediction results
    n : int
        limit of return results
    threshold : float
        filters results to Total Disagreement >= this value
    desc : bool
        descending sort
    """
    sorted_cls = [{k : v} for (k,v) in sorted(class_roll['CLASSES'].items(), key=lambda s: s[1]['Total Disagreement']['rate'], reverse=desc)]
    sorted_cls = [{k:v for (k,v) in x.items() if v['Total Disagreement']['rate'] >= threshold} for x in sorted_cls]
    sorted_cls = [x for x in sorted_cls if x != {}]
    return sorted_cls[:n]


def rpt_dump_flat(class_roll, output_path=None):
    """ return flat list of lists for test results or save to outpath """
    out_arr = []
    class_vals = class_roll['CLASSES']
    global_vals = class_roll['GLOBAL']

    def flatten(key, metrics, type):
        return [
                key,
                type,
                metrics['Records'],
                metrics['True Negative'][type],
                metrics['True Positive'][type],
                metrics['Fuzzy Positive'][type],
                metrics['No Tag'][type],
                metrics['False Positive'][type],
                metrics['Rejected'][type],
                metrics['Total Disagreement'][type]
        ]

    for k,v in class_vals.items():
        out_arr.append(flatten(k,v,'cnt'))
        out_arr.append(flatten(k,v,'rate'))

    out_arr.append(flatten('GLOBAL', global_vals, 'cnt'))
    out_arr.append(flatten('GLOBAL', global_vals, 'rate'))

    if output_path:
        header = ['Class', 'Type', 'Records', 'True Negative', 'True Positive', 'Fuzzy Positive', 'No Tag', 'False Positive', 'Rejected', 'Total Disagreement']
        out_arr.insert(0,header)
        with open(output_path, 'w') as f:
            for row in ['\t'.join([str(c) for c in r]) for r in out_arr]:
                f.write(row + '\n')
    
    else:
        return out_arr


class Train:
    """ wraps sklearn classes and controls the training, testing, and saving of models """

    def __init__(self):
        self.dat = None
        self.model = None
        self._version = None
        self.meta = {
            'create time': datetime.datetime.now().isoformat(' ','seconds'),
            "env": get_environ_details()
        }

    def load_data(self, data_obj):
        """ load data.ModelData object for training
        dat : ModelData
            data.ModelData object 
        """

        if self.dat:
            logger.warning('data object already exists, aborting load')
            return

        self.dat = data_obj


    def train_model(self, c_penalty=1, weights=None, output_path=None, version=None):
        """
        trains a SVM model and saves it
        Parameters
        ----------
        c_penalty : float
            penalty parameter C of the error term
        weights : dict
            class weights
        output_path : str
            path to save trained model
        """

        if self.model:
            logger.warning('called train when model already exists, aborting train')
            return
        
        if not self.dat:
            logger.warning('train called without data, aborting train')
            return


        samples, features = self.dat.xmat.shape
        is_dual = True
        loss_func = 'hinge'
        if samples > features:
            is_dual = False
            loss_func = 'squared_hinge'

        self.model = LinearSVC(
                        C=c_penalty,
                        dual=is_dual,
                        class_weight=weights,
                        loss=loss_func,
                        verbose=0)

        logger.info(f'training calibrated model with C={c_penalty}..')
        start = datetime.datetime.now()

        if len(self.dat.ys) < 100e3:
            # small sets can calibrate against the full set
            self.model = CalibratedClassifierCV(base_estimator=self.model, method='sigmoid', cv=3)  # must have at least 3 samples of each class
            self.model.fit(self.dat.xmat, self.dat.ys)      
        else:
            # sample data for calibration on larger sets (data must have at least 10 samples of each class with 10% strat split)
            _, x_valid, _, y_valid = train_test_split(self.dat.xmat, self.dat.ys, test_size=0.1, stratify=self.dat.ys)
            if np.setdiff1d(self.dat.ys, y_valid):
                raise Exception('y_valid missing classes, failed to stratify at 10% level: must have at least 10 samples of each class')

            self.model.fit(self.dat.xmat, self.dat.ys)  # train full set to avoid missing lables; violates disjoint sets recomended
            self.uncal_model = self.model  # stored for tuning which doesn't like calibrated models, this can probably be resolved
            self.model = CalibratedClassifierCV(base_estimator=self.model, method='sigmoid', cv='prefit')
            self.model.fit(x_valid, y_valid)

        end = datetime.datetime.now()
        logger.info(f"train time: {str(end - start)}")
        self.meta.update({
            'estimator': 'LinearSVC',
            'calibrated': True,
            'fit params': {
                'X shape': self.dat.xmat.shape
                },
            'train time': str(end - start),
            'data config': self.dat.CONFIG,
            'version':version
            })

        self.meta['fit params'].update(self.model.get_params())
        self.model_vocabulary = self.dat.full_vocab
        self.model_classes = self.model.classes_
        self.raw_config = self.dat.CONFIG
        self._version = version
        
        if output_path:
            self.save_model(output_path)

    def train_model_rf(self, n_est=100, max_depth=None, max_leaf=None, njobs=None, verbose=0, version=None, output_path=None):
        """
        trains a Random Forest model
        Parameters
        ----------
        n_est : int (100)
        max_depth : int (None)
        max_leaf : int (None)
        njobs : int (None)
            number of parallel jobs to spawn for fit, and predict
        verbose : int (0)
            controls the stdout verbosity of fit and predict
        version : string (none)
            version associated with model
        output_path : string (None)
        """

        if self.model:
            logger.warning('called train when model already exists, aborting train')
            return
        
        if not self.dat:
            logger.warning('train called without data, aborting train')
            return 

        self.model = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_depth,
                        max_leaf_nodes=max_leaf,
                        n_jobs=njobs,
                        class_weight="balanced",
                        verbose=verbose)

        logger.info(f"training random forest model with {n_est} estimators")
        start = datetime.datetime.now()
        self.model.fit(self.dat.xmat, self.dat.ys)
        end = datetime.datetime.now()
        logger.info(f"train time: {str(end - start)}")
                    
        self.meta.update({
            'estimator': 'RandomForestClassifier',
            'calibrated': False,
            'fit params': {
                'X shape': self.dat.xmat.shape
                },
            'train time': str(end - start),
            'data config': self.dat.CONFIG,
            'version':version
            })

        self.meta['fit params'].update(self.model.get_params())
        self.model_vocabulary = self.dat.full_vocab
        self.model_classes = self.model.classes_
        self.raw_config = self.dat.CONFIG
        self._version = version

        if output_path:
            self.save_model(output_path)


    def calc_model_score(self, chunk_size=10000):
        # accuracy of in-sample fitted data
        logger.info('calcuating model score..')
        
        # build predicted vector in pieces to conserve memory
        pred_y = np.concatenate([self.model.predict(self.dat.xmat[i:i+chunk_size]) for i in range(0, self.dat.xmat.shape[0], chunk_size)])
        acc = accuracy_score(self.dat.ys, pred_y)
        logger.info(f"score: {acc}")
        self.meta.update({'fit score': acc})


    def save_model(self, output_path):
        logger.info(f'saving model to: {output_path}')
        elems = {k: v for k, v in self.__dict__.items() if k != 'dat'}
        with open(output_path, 'wb') as f:
            dump(elems, f, compress='gzip')


    def load_model(self, model_path, data_obj):
        """ load a trained model to tune or test, requires associated dat object used to train """
        if self.model:
            logger.warning('model already exists, aborting load')
            return

        with open(model_path, 'rb') as f:
            self.__dict__ = load(f)

        self.dat = data_obj


    def cross_val(self, kfolds=10, njobs=-1):
        """
        perform stratified kfold cross validation to produce accuracy estimate
        Parameters
        ----------
        kfolds : int
            number of stratisfied kfolds to generate in cross validation
        njobs : int
            number of jobs to spawn, -1 is all
        """
        if self.model is None:
            logger.warning('cross validation called without estimation model')
            return

        logger.info(f'cross validating model with {kfolds} folds')
        start = datetime.datetime.now()
        cross_val = cross_val_score(self.model, self.dat.xmat, self.dat.ys, cv=kfolds, n_jobs=njobs)
        end = datetime.datetime.now()
        logger.info(f'cross val finished in: {str(end - start)} with niave accuracy: {cross_val.mean()}')
        self.meta.update({
            'cross_validation': {
                'strat kfolds': kfolds,
                'accuracy': "{:0.4f} (+/- {:0.4f})".format(cross_val.mean(), cross_val.std() * 2),
                'cross time': str(end - start)
                }
            })


    def test(self, thr=0.49, ret=False, cls_dim=False, chunk_size=10000):
        """
        test model using self.dat

        Metrics
        TN (True Negative) - predicted NULL, actual is NULL
        TP (True Positive) - predicted value x, actual is value x
        FuzzP (Fuzzy Positive) - predicted value, actual is NULL (possible human error)
        NoTag - predicted NULL, actual is value (tag didn't meet threshold and/or pattern is un-trained--not strictly an error)
        FP (False Positive) - predicted value x, actual is value y (true errors, however also possible the human tagged value is incorrect and this will find them)
        Rejected - tags rejected due to insufficient probability
        TotDis - Total Disagreement, all gross disagreements

        Parameters
        ----------
        thr : float
            necessary probability to return tag, otherwise NULL
        ret : bool
            return a result or not. set to true when called repeatedly to avoid info spam, used for tuning
        cls_dim : bool
            return metrics by class {CLASSES: dict, GLOBAL: dict}
        """
        if not self.model:
            logger.warning('test called without trained model')
            return

        start = datetime.datetime.now()

        # predictions
        if not ret:
            logger.info('generating predictions..')
        prob_mat = np.concatenate([self.model.predict_proba(self.dat.xmat[i:i+chunk_size]) for i in range(0, self.dat.xmat.shape[0], chunk_size)])
        tags = self.model.classes_[prob_mat.argmax(axis=1)]
        probs = prob_mat.max(axis=1)
        pred_dict = [{'tag':tag, 'prob':prob} for (tag, prob) in zip(tags, probs)]

        # to predict a class prob must be >= thr
        if not ret:
            logger.info(f'nulling results with prob < {thr}..')
        for i, rec in enumerate(pred_dict):
            if rec['prob'] < thr:
                rec.update({'rejected': rec['tag']})
                rec['tag'] = ''
            
            rec.update({'y_act': self.dat.ys[i]})

        # roll metrics
        if not ret:
            logger.info('rolling metrics..')
 
        result = metric_roll(pred_dict, cls_dim)

        end = datetime.datetime.now()
        result.update({
                'Test Time' : str(end - start),
                'Threshold' : thr
            })

        if not ret:
            logger.info(f'testing metrics time: {str(end - start)}')
            self.meta.update({'test results': result})
            logger.info(f"test results:\n{result}")
        else:
            return result


    def tune_params(self, grid={'C':[0.01, 0.05, 0.1, 0.5, 1, 10]}, re_fit=False):
        """
        grid search to find best parameters for model
        Parameters
        ----------
        grid : dict
            dictionary of paramters(key) and values to tune `grid`
        refit : Bool
            refit model with best params when done; don't do this currently as it is using uncalibrated model to tune params
        """
        if not self.uncal_model:
            logger.warning('tune called without trained model, aborting tune.')
            return

        logger.info(f'starting grid search with: {grid}..')
        grid_search = GridSearchCV(
            self.uncal_model,
            grid,
            scoring=None,
            n_jobs=-1,
            refit=re_fit,
            cv=10,
            verbose=3)
        
        start = datetime.datetime.now()
        grid_search.fit(self.dat.xmat, self.dat.ys)
        end = datetime.datetime.now()
        logger.info(f'grid search time: {str(end - start)}\t best params: {grid_search.best_params_}')
        self.meta.update({
            'grid search': {
                'grid time' : str(end - start),
                'best' : grid_search.best_params_
            }
        })


    def tune_thr(self, penalty=1, procs=8):
        """
        tune threshold value
        Parameters
        ----------
        penalty : float
            penalty to exponentiate false positives by
        procs : int
            number of processes to spawn
        """    
        def metric(_tp, _tn, _fp, _n):
            return (_tp + _tn) / (_fp ** penalty + _n)

        def calc_mets(chunk, q):

            for thr in chunk:
                res = self.test(thr=thr/100, ret=True)
                tp = res['True Positive']['cnt']
                tn = res['True Negative']['cnt']
                fp = res['False Positive']['cnt'] + res['Fuzzy Positive']['cnt']
                n = res['Records']
                met = metric(tp, tn, fp, n)
    
                # print(f"thr: {thr}, met: {met}")
                q.put({'thr':thr, 'met': met, 'test_res': res})

        logger.info(f'tunning threshold with penalty: {penalty}')
        results = {}
        start = datetime.datetime.now()
        split_range = [range(100)[i::procs] for i in range(procs)]

        processes = []
        q = mp.Queue()
        print('starting procs..')
        for chk in split_range:
            p = mp.Process(target=calc_mets, args=(chk,q))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        pool_ret = []
        while not q.empty():
            pool_ret.append(q.get())
       
        pool_ret.sort(key=lambda s: s['met'], reverse=True)
        results['iters'] = pool_ret
        best_rec = pool_ret[0]

        end = datetime.datetime.now()
        results.update({
            'best': {'thr': best_rec['thr'], 'met': best_rec['met']},
            'tune_time': str(end - start)
            })
        self.meta.update({'tune thr': results['best']})
        logger.info(f"best thr: {results['best']} after: {results['tune_time']}")

        return results



class TaggingModel:
    """ wraps an sklearn model to allow for predictions and/or testing against a holdout sample """

    def __init__(self, model_obj=None):
        self.model = None
        self.model_vocabulary = None
        self.meta = {}
        self.raw_config = None
        self._version = None
        self._exc_model_obj = None  # can load another model object to pre-process txns for excluded items

        if model_obj:
            self.__dict__ = {k: v for k, v in model_obj.__dict__.items() if k != 'dat'}

    @staticmethod
    def _regexp_matches(tx_desc):
        """
            check the passed tx_desc for regexp matches from defined collection
            assumes patterns are disjoint and returns the first match
            returns tag if matched, otherwise None
        """
        for tag, pat in PATTERNS.items():
            if pat["regexp"].search(tx_desc):
                return tag
        
        return None


    def predict_one(self, record, thr=0.49):
        """
        predict a single record
        rec : dict
            feature dict, a single row from ModelData._load_data method
        thr : float
            necessary probability to return tag, otherwise NULL ('')

        record looks like (depending on config):
            {   'tx_desc': 'POS Debit - Visa Check Card 7817 - BP#9492372CIRCL OVIEDO FL',
                'tx_amt': '21.38',
            }
        """
        actual_brand = record.get("brand_id", None)

        # check regexp patterns before using model
        regexp_result = self._regexp_matches(record["tx_desc"])
        if regexp_result:
            pred_dict = {
                'tag': regexp_result,
                'prob': 1
            }
            
        # no regexp matches use model
        else:
            # encode record using model vocabulary
            dat = ModelData(vocab=self.model_vocabulary, config_obj=self.raw_config)  # uses model config to encode data
            dat.load_record(record=record)
            dat.fit_transform()

            prob_mat = self.model.predict_proba(dat.xmat)
            tag = self.model.classes_[prob_mat.argmax(axis=1)]
            prob = prob_mat.max(axis=1)

            pred_dict = {
                'tag':tag[0],
                'prob':prob[0]
            }

            if prob < thr:
                pred_dict.update({'rejected': pred_dict['tag']})
                pred_dict['tag'] = ''
            
        # add in known brand if provided
        if actual_brand:
            pred_dict.update({"y_act": actual_brand})

        # add model version to result
        pred_dict.update({'version': self._version})

        # add in exclusion results if exc_model loaded
        # return exclusion if matched regardless of brand tag to provide flexibility
        if self._exc_model_obj:
            exc_results =  self.exc_predict_one(record)
            if exc_results["tag"]:
                pred_dict.update({
                    "exclusion_tag": exc_results["tag"],
                    "exclusion_prob": exc_results["prob"],
                    "exclusion_version": self._exc_model_obj["_version"],

                })

        return pred_dict


    def predict_many(self, records_path=None, data_obj=None, thr=0.49, chunk_size=10000):
        """
        predict class (tag) using csv data source
        source_path : str
            path string to data csv to be predicted
        thr : float
            necessary probability to return tag, otherwise NULL

        note: does not include the added exclusion model in it's output
        """
        logger.info(f'predicting from file, using thr = {thr}')
        start = datetime.datetime.now()

        if records_path:
            pred_data = ModelData(vocab=self.model_vocabulary, config_obj=self.raw_config)
            pred_data.load_csv(records_path)
            pred_data.fit_transform()
    
        elif data_obj:
            pred_data = data_obj
            if pred_data.full_vocab != self.model_vocabulary:
                logger.warning('recieved data vocabulary does not match model vocabulary, aborting prediction')
                return
        else:
            logger.warning('prediction requires a filepath to records or a data object, aborting prediction')
            return

        prob_mat = np.concatenate([self.model.predict_proba(pred_data.xmat[i:i+chunk_size]) for i in range(0, pred_data.xmat.shape[0], chunk_size)])
        tags = self.model.classes_[prob_mat.argmax(axis=1)]
        probs = prob_mat.max(axis=1)
        pred_dict = [{'tag':tag, 'prob':prob, 'version':self._version} for (tag, prob) in zip(tags, probs)]

        # post process and override using regexp patterns
        if hasattr(pred_data, "tx_desc"):
            for i in range(len(pred_data.tx_desc)):
                regexp_match = self._regexp_matches(pred_data.tx_desc[i]["tx_desc"])
                if regexp_match:
                    pred_dict[i]["tag"] = regexp_match
                    pred_dict[i]["prob"] = 1

        # to predict a class prob must be >= thr
        for i, rec in enumerate(pred_dict):
            if rec['prob'] < thr and rec['tag'] != '':
                rec.update({'rejected': rec['tag']})
                rec['tag'] = ''
            
            if pred_data.ys is not None:
                rec.update({'y_act': pred_data.ys[i]})

        end = datetime.datetime.now()
        logger.info(f'prediction time: {str(end - start)}')

        return pred_dict
    

    def exc_predict_one(self, record, thr=0.85):
        """
            this method uses the loaded exclusion model (if present)
            to determine if the transaction is from an excluded brand
        """
        # encode record using model vocabulary
        dat = ModelData(vocab=self._exc_model_obj["model_vocabulary"], config_obj=self._exc_model_obj["raw_config"])  # uses model config to encode data
        dat.load_record(record=record)
        dat.fit_transform()

        prob_mat = self._exc_model_obj["model"].predict_proba(dat.xmat)
        tag = self._exc_model_obj["model"].classes_[prob_mat.argmax(axis=1)]

        prob = prob_mat.max(axis=1)

        pred_dict = {
            'tag':tag[0],
            'prob':prob[0]
        }

        if prob < thr:
            pred_dict.update({'rejected': pred_dict['tag']})
            pred_dict['tag'] = None

        return pred_dict

        

    def test(self, test_path=None, data_obj=None, thr=0.49, cls_dim=False):
        """
        test model against un-seen data (or seen)
        test_path : str
            path to test records (csv)
        thr : float
            value below which class is set to Null
        cls_dim : bool
            perform class-based roll-up of results
        """
        start = datetime.datetime.now()
        pred_dict = self.predict_many(records_path=test_path, data_obj=data_obj, thr=thr)
        if 'y_act' not in pred_dict[0]:
            logger.warning(f'test called without y_actuals in file: {test_path}')
            return

        # roll metrics
        logger.info('rolling metrics..')
        result = metric_roll(pred_dict, cls_dim)
        end = datetime.datetime.now()
        logger.info(f"elapsed: {str(end - start)}")
        if cls_dim:
            logger.info(f"GLOBAL results: {result['GLOBAL']}")

        return result


    def load_model(self, model_path):
        """ load trained model """
        if self.model:
            logger.warning('model already exists, aborting load.')
            return

        logger.info(f'loading model: {model_path}')
        with open(model_path, 'rb') as f:
            loaded_object = load(f)
            #dict_keys(['model', '_version', 'meta', 'model_vocabulary', 'model_classes', 'raw_config'])

            self.model = loaded_object["model"]
            self._version = loaded_object["_version"]
            self.meta = loaded_object["meta"]
            self.model_vocabulary = loaded_object["model_vocabulary"]
            self.model_classes = loaded_object["model_classes"]
            self.raw_config = loaded_object["raw_config"]
            #self.__dict__ = load(f)
            
    def save_model(self, output_path):
        logger.info(f'saving model to: {output_path}')
        elems = {k: v for k, v in self.__dict__.items() if k != 'dat'}
        with open(output_path, 'wb') as f:
            dump(elems, f, compress='gzip')


    def load_exc_model_obj(self, model_path):
        """ load exclusion model object to filter results with """
        if self._exc_model_obj:
            logger.warning('exclusion model object already exists, aborting load.')
            return

        logger.info(f'loading exclusion model object: {model_path}')
        with open(model_path, 'rb') as f:
            self._exc_model_obj = load(f)


    def get_version(self):
        return self._version

    def get_exc_version(self):
        return self._exc_model_obj["_version"]

    def get_config(self):
        return self.raw_config

    def get_exc_config(self):
        return self._exc_model_obj["raw_config"]

from training import sklearn_ext as ske

import numpy as np
import csv, datetime, logging, json
from joblib import dump, load
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from operator import itemgetter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

class ModelData:
    """
    pre-processing class for subsequent use by a Model class
    expects source data with headers and encoding defined in config json
    y-values are optional (needed for subsequent training/testing, but not predictions)
    method seq: {init, load_csv, fit_transform}
    """
    def __init__(self, vocab=None, config_path='./config/data_config.json', config_obj=None):
        
        if config_obj:
            self.CONFIG = config_obj
        else:
            with open(config_path) as f:
                self.CONFIG = json.load(f)

        self._parse_config(self.CONFIG)
        self.provided_vocab = False
        if vocab:
            self.provided_vocab = True
            self.full_vocab = vocab
            
        self.raw_dat = None
        self.pipe = None
        self.xmat = None
        self.tx_desc = None
        

    def load_record(self, record):
        """
        loads raw data from single record dict and stores list of dictionaries
        record : dict
            record dictionary
        """
        if self.raw_dat:
            logger.warning('raw data is already loaded, aborting load')
            return

        if not self.provided_vocab:
            logger.warning('a single record requires a vocabulary, aborting load')
            return

        logger.info(f"loading single record: {record}")
        self.headers_ = record.keys()
        self._check_headers()

        self.raw_dat = [record]
        self.tx_desc = [{"tx_desc": rec["tx_desc"]} for rec in self.raw_dat]
        if self.config['y_vals'] in record:
            self.ys = record[self.config['y_vals']]
        else:
            self.ys = None


    def load_csv(self, filepath):
        """
        loads raw data from csv and returns ordered list of dictionaries
        filepath : str
            path to csv
        """
        if self.raw_dat:
            logger.warning('raw data is already loaded, canceling load')
            return

        logger.info(f"loading data from csv: {filepath}")
        start = datetime.datetime.now()

        self.headers_ = self._load_headers(filepath)
        self._check_headers()
        self._sort_config()  # this sorts the config order to match the document

        self.raw_dat = list(self._load_data(filepath))
        if self.config['y_vals'] in self.raw_dat[0]:
            self.ys = np.array([rec[self.config['y_vals']] for rec in self.raw_dat])
            if len(set(self.ys)) == 1:
                logger.warning(f'a single class value was given: {self.ys[0]} this will cause all features to be removed during chi2 reduction.')
        else:
            self.ys = None

        self.tx_desc = [{"tx_desc": rec["tx_desc"]} for rec in self.raw_dat]
        logger.info(f"loading data elapsed time: {datetime.datetime.now() - start}")


    def fit_transform(self, thr=0.05, overwrite=False):
        """
        encode loaded data as csr matrix after reducing features using chi2 reduction
        thr : float
            pvalue above which features are dropped
        """
        if not self.raw_dat:
            logger.warning('called fit_transfrom before loading data, aborting fit')
            return

        if self.xmat is not None and not overwrite:
            logger.warning('fit called with existing csr matrix, use: "overwrite=True" to re-build. Aborting fit')
            return

        logger.info('encoding raw data..')
        start = datetime.datetime.now()
        if not self.pipe:
            self._make_pipe()
        tran = self.pipe.fit_transform(self.raw_dat)

        # reduce features using chi2 if no vocabulary
        if not self.provided_vocab:
            logger.info(f"starting shape: {tran.shape}\treducing features..")
            pvals = chi2(tran, self.ys)
            mask = np.array([p <= thr for p in pvals[1]])  # mask for all features
            self.xmat = tran[:,mask]
            logger.info(f"ending shape: {self.xmat.shape}\tremoved: {tran.shape[1] - self.xmat.shape[1]} features")
            
            # store vocabulary
            self._store_vocab(mask)

            # store raw mask mapping
            raw_vocab = []
            for _, stage in enumerate(self.pipe.named_steps['union'].transformer_list):
                raw_vocab.extend( [f"{stage[0]}_{k}" for k, _ in sorted(stage[1].vocabulary_.items(), key=itemgetter(1))] )
            self.mask_map = list(zip(raw_vocab, mask, pvals[1]))

        else:
            self.xmat = tran
            self.vocabulary_ = self._flatten_vocab()

        logger.info(f"encoding elapsed time: {datetime.datetime.now() - start}")


    def save_data_obj(self, output_path):
        """ serialize ModelData object excluding raw data """
        logger.info(f'saving ModelData object to: {output_path}')
        if not self.tx_desc and self.raw_dat:
            self.tx_desc = [{"tx_desc": rec["tx_desc"]} for rec in self.raw_dat]
        elems = {k: v for k, v in self.__dict__.items() if k != 'raw_dat'}
        with open(output_path, 'wb') as f:
            dump(elems, f, compress='gzip')


    def load_data_obj(self, data_path):
        """ de-serialize ModelData object excluding raw data """
        if self.xmat:
            logger.warning('data already exists, aborting load.')
            return

        logger.info(f'loading ModelData object: {data_path}')
        with open(data_path, 'rb') as f:
            self.__dict__ = load(f)


    def _store_vocab(self, mask):
        # store vocabulary by stage
        # indecies are local to the stage of the pipe
        logger.info('storing vocabulary..')

        # lift pipe stage vocabularies and park in dictionary
        stage_vocabularies = {}
        lwr_bnd, upr_bnd = (0,0)
        for i, stage in enumerate(self.pipe.named_steps['union'].transformer_list):
            stg_len = len(stage[1].vocabulary_)
            upr_bnd += stg_len
            stage_vocabularies.update({
                stage[0] : {
                    'ord' : i,
                    'vocab' : stage[1].vocabulary_,
                    'length' : stg_len,
                    'lwr' : lwr_bnd,
                    'upr' : upr_bnd
                }
            })
            lwr_bnd += stg_len
            
        # filter and rebuild each stage vocabulary  {bow:cols, one-hot, minmax}
        for stage, vals in sorted(stage_vocabularies.items(), key= lambda s: s[1]['ord']):
            self.full_vocab[stage] = self._filter_dict(vals['vocab'], mask[vals['lwr']:vals['upr']])

        # store global vocabulary
        self.vocabulary_ = self._flatten_vocab()


    @staticmethod
    def _filter_dict(in_dict, mask):
        """
        return dictionary filtered by passed boolean mask
        used to take a raw vocabulary and filter it by a slice of bools from the chi2 reduction
        this return dictionary is the stored vocabulary for the pipe-stage
        """
        ret = [ (k, v) for k,v in sorted(in_dict.items(), key=itemgetter(1)) if mask[v] ]  # list of ordered tuples filtered by mask
        return { k[0]:v for v, k in enumerate(ret) }  # apply new indices


    def _flatten_vocab(self):
        """ flatten full_vocab """

        # define pipe-order
        pipe_order = [stg[0] for stg in self.pipe.named_steps['union'].transformer_list]

        flat_vocab = []
        for stage in pipe_order:
            flat_vocab.extend(sorted(self.full_vocab[stage].items(), key=itemgetter(1)))

        return { k[0]:v for v, k in enumerate(flat_vocab) }


    def _make_pipe(self):
        """ construct feature unions from config (feature concatination) """
        if not self.raw_dat:
            # config order is modified based on loaded data, should build pipe after data order is known (not tested)
            logger.warning('make_pipe called before data load, please load data first.')
            return

        union_list = []

        # bag of words features
        for ftr in self.config['bow']:
            # grab one column at a time to transform
            union_list.append(
                (ftr, ske.PipeSpec([
                    ('ext', ske.ColsExtractor([ftr], encoding='bow')),
                    ('vect', TfidfVectorizer(
                        analyzer=self.vect_params[ftr]['analyzer'],  # either word or char
                        token_pattern=self.vect_params[ftr]['token_pattern'], # only used if analyzer == 'word'
                        ngram_range=(self.vect_params[ftr]['ngram'][0], self.vect_params[ftr]['ngram'][1]),
                        vocabulary=self.full_vocab[ftr]))
                ]))
            )

        # one-hot features
        if self.config['one_hot']:
            union_list.append(
                ('one-hot', ske.PipeSpec([
                    ('extcat', ske.ColsExtractor(self.config['one_hot'], encoding='one-hot')),
                    ('vectcat', ske.DictVectVoc(self.full_vocab['one-hot']))

                ]))
            )

        # minmax features
        if self.config['minmax']:
            union_list.append(
                ('minmax', ske.PipeSpec([
                    ('ext', ske.ColsExtractor(self.config['minmax'], encoding='minmax')),
                    ('scale', MinMaxScaler(copy=False)),
                    ('filter', ske.ScaleFilter(vocab=self.full_vocab['minmax'], cols=self.config['minmax']))
                ]))
            )
            
        # construct pipe from feature union
        self.pipe = ske.PipeSpec(
            steps=[
                ('union', FeatureUnion(union_list, n_jobs=-1))
            ],
            verbose=1
        )


    def _check_headers(self):
        """ check headers (excluding class as this may or may not exist in data) """
        hdr_set = {hdr for hdr in self.headers_ if hdr != self.config['y_vals'] and hdr not in self.ignore_columns}
        config_set = {hdr for hdr in self.all_columns if hdr != self.config['y_vals'] and hdr not in self.ignore_columns}
        check_columns = hdr_set ^ config_set  # symetric set difference
        if check_columns:
            raise Exception(f"columns don't match config: {check_columns}\nCorrect config: {self.CONFIG}")


    def _sort_config(self):
        """ put config feature columns in document order if more than one """
        for enc, lst in self.config.items():
            if len(lst) > 1 and enc != 'y_vals':

                doc_order = [col for col in self.headers_ if col in lst]
                for i, col in enumerate(lst):
                    doc_value = doc_order[i]
                    if col != doc_value:
                        self.config[enc][i] = doc_value


    def _load_data(self, filepath):
        try:
            with open(filepath) as csvfile:
                reader = csv.DictReader(csvfile)
                try:
                    for row in reader:
                        yield row
                except csv.Error as e:
                    raise Exception('file {}, line {}: {}'.format(filepath, reader.line_num, e))

        except FileNotFoundError as e:
            raise Exception(f"could not load raw data: {e}")


    def _load_headers(self, filepath):
        try:
            with open(filepath) as f:
                reader = csv.reader(f)
                return next(reader)
                
        except FileNotFoundError as e:
            raise Exception(f"could not load header data: {e}") 


    def _parse_config(self, config):
        """ take config dict and store a dict
            create and store empty vocab
        """
        self.all_columns = list(config['columns'].keys())
        out_dict = {
            'one_hot' : [],     # one vocabulary
            'bow' : [],         # vocabulary for each
            'minmax' : [],      # one vocabulary
            'y_vals' : None
        }

        self.ignore_columns = []
        self.vect_params = {} # used as params in _make_pipe vectorizers
        for col, attr in config['columns'].items():
            if attr['encoding'] == 'one-hot':
                out_dict['one_hot'].append(col)
            elif attr['encoding'] == 'bow':
                out_dict['bow'].append(col)
                self.vect_params.update({
                    col : {
                        'ngram' : attr['ngram'],
                        'analyzer' : attr['analyzer'],
                        'token_pattern' : attr['token_pattern']
                    }
                })
            elif attr['encoding'] == 'minmax':
                out_dict['minmax'].append(col)
            elif attr['type'] == 'class':
                out_dict['y_vals'] = col
            elif attr['type'] == 'meta':
                self.ignore_columns.append(col)

        self.config = out_dict

        # empty vocabulary dict
        self.full_vocab = {
            'one-hot' : None,
            'minmax'  : None
        }
        for col in out_dict['bow']:
            self.full_vocab.update({
                col : None
            })      

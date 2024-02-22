# pylint: disable=import-error
import sys, os, json

projRoot = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports (horible module management, ugh)

from training.data import ModelData
from training.model import Train
from training.model import TaggingModel
import csv

import unittest
from copy import deepcopy

# global setup
CFG = os.path.join(projRoot,'config/tab_config_train.json') # path to default TAB config file
CFG_EXC = os.path.join(projRoot,'config/tab_exc_config_train.json') # path to TAB exclusion model config
PRED_FIELDS = (3,7)  # { tag, prob, rej?, ver, exc_tag?, exc_prob?, exc_ver? } (interval [3,7])


# top level tests for methods at the interface level (TaggingModel class)
# assumes problems in base classes will bubble to the top (probably a bad assumption)

class TestTaggingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ build a model using default config and test data in same directory
            should take about ~25 seconds to build
        """
        print("building model for tests")
        cls.test_data = ModelData(config_path=CFG)
        cls.test_data.load_csv('./dat_unit.csv')
        cls.test_data.fit_transform()

        cls.train_model = Train()
        cls.train_model.load_data(cls.test_data)
        cls.train_model.train_model(c_penalty=0.1, weights='balanced', version='unittest')
        cls.train_model.save_model('./test_model.gz')

        cls.model_under_test = TaggingModel()
        cls.model_under_test.load_model('./test_model.gz')
        cls.model_under_test.load_exc_model_obj("../data/tab_exc_mod_rf.gz")

        # some test data which is both wellformed and malformed
        try:
            cls.well_formed_record = deepcopy(cls.test_data.raw_dat[1])  # should work
            cls.malformed_record_add = deepcopy(cls.well_formed_record); cls.malformed_record_add.update({'bad_field': 'icky value'})  # should break
            cls.malformed_record_sub = {k: cls.well_formed_record[k] for k in cls.well_formed_record if k != 'tx_desc'}  # should break
            cls.well_formed_lessmeta_record = {k: cls.well_formed_record[k] for k in cls.well_formed_record if k != 'brand_name'}  # should work

            # grab exclusion test record
            with open("./exc_rec_test.json") as f:
                cls.exc_record = json.load(f)

        except Exception:  # not worried about naked except here
            raise("could not construct data sets for testing")


    @classmethod
    def tearDownClass(cls):
        """ May not be necessary but in the interest of explicitly freeing memory """
        del cls.test_data
        del cls.train_model
        del cls.model_under_test
        del cls.well_formed_record
        del cls.malformed_record_add
        del cls.malformed_record_sub
        del cls.well_formed_lessmeta_record

        if os.path.exists('./test_model.gz'):
            os.remove('./test_model.gz')

        if os.path.exists('./dat_unit_add.csv'):
            os.remove('./dat_unit_add.csv')

        if os.path.exists('./dat_unit_sub.csv'):
            os.remove('./dat_unit_sub.csv')


    # model exists
    def test_model_exists(self):
        self.assertIsNotNone(self.model_under_test, msg="model was not created")

    # meta tests
    def test_meta(self):
        self.assertIsNotNone(self.model_under_test.meta, msg="meta was not created")
        self.assertFalse(self.model_under_test.meta == {}, msg="meta is empty")
        self.assertFalse(self.model_under_test.meta.get('data config', 'ERROR') == 'ERROR', msg="data config not present in meta")
        self.assertFalse(self.model_under_test.meta.get('version', 'ERROR') == 'ERROR', msg="version not present in meta")
        self.assertEqual(self.model_under_test.meta['version'], 'unittest', msg="version not stored correctly in meta")

    # getter tests
    def test_get_config(self):
        self.assertFalse(self.model_under_test.get_config() == {}, msg="get_config returned empty dict")
        self.assertFalse(self.model_under_test.get_config().get('columns', {}).get('tx_desc', 'ERROR') == 'ERROR', msg="get_config returned missing tx_desc") # assumed constant
        self.assertEqual(self.model_under_test.get_config(), self.test_data.CONFIG, msg="get_config doesn't match data config")

    def test_get_version(self):
        self.assertFalse(self.model_under_test.get_version() == '', msg="get_version returned empty string")
        self.assertIsNotNone(self.model_under_test.get_version(), msg="get_version returned None")
        self.assertEqual(self.model_under_test.get_version(), 'unittest', msg="get_version did not return: unittest as expected")
        self.assertEqual(self.model_under_test.get_version(), self.train_model._version, msg="get_version doesn't match train version")
        self.assertFalse(self.model_under_test.get_exc_version() == '', msg="get_exc_version returned empty string")
        self.assertIsNotNone(self.model_under_test.get_exc_version(), msg="get_exc_version returned None")

    # property tests
    def test_properties_version(self):
        self.assertIsNotNone(self.model_under_test._version, msg="version was not created")
        self.assertEqual(self.model_under_test._version, 'unittest', msg="stored version is not unittest")
        self.assertIsNotNone(self.model_under_test._exc_model_obj["_version"], msg="exc model version was not created")

    def test_properties_config(self):
        self.assertIsNotNone(self.model_under_test.raw_config, msg="raw_config was not created")
        self.assertNotEqual(self.model_under_test.raw_config, {}, msg="raw_config is empty dict")
        self.assertEqual(self.model_under_test.raw_config, self.test_data.CONFIG, msg="raw_config doesn't match data config")
        self.assertIsNotNone(self.model_under_test._exc_model_obj["raw_config"], msg="excluded model raw_config was not created")

    ### predict one tests ###
    def test_predict_one_wf(self):
        # well formed tests
        self.assertIsNotNone(self.model_under_test.predict_one(self.well_formed_record), msg="pred_one returned None")
        well_formed_result = self.model_under_test.predict_one(self.well_formed_record)
        self.assertTrue(len(well_formed_result.keys()) in range(PRED_FIELDS[0], PRED_FIELDS[1]), msg=f"expected {PRED_FIELDS} to {PRED_FIELDS} keys recieved {well_formed_result}")
        self.assertEqual(well_formed_result.get('version', None), 'unittest', msg="version not returned by predict_one")

    def test_predict_one_mf(self):
        # malformed records
        self.assertRaises(Exception, self.model_under_test.predict_one, self.malformed_record_add)
        self.assertRaises(Exception, self.model_under_test.predict_one, self.malformed_record_sub)

    def test_predict_one_ignore_meta(self):
        # meta is ignored
        well_formed_result = self.model_under_test.predict_one(self.well_formed_record)
        self.assertIsNotNone(self.model_under_test.predict_one(self.well_formed_lessmeta_record), msg="pred_one returned None for meta value exclusion")
        self.assertEqual(self.model_under_test.predict_one(self.well_formed_lessmeta_record), well_formed_result, msg="meta exclusion result doesn't match well formed")

    def test_predict_one_excluded(self):
        # check exclusion model results
        excluded_result = self.model_under_test.predict_one(self.exc_record)
        self.assertIsNotNone(excluded_result, msg="pred_one with exclusion returned None")

        # does the result contain the keys { exclusion_tag, exclusion_prob, exclusion_version }
        returned_keys = excluded_result.keys()
        expected_keys = ["exclusion_tag", "exclusion_prob", "exclusion_version"]
        for k in expected_keys:
            self.assertTrue(k in returned_keys)

        # ensure each of the keys contain a value (not none or empty)
        for k in expected_keys:
            self.assertFalse(excluded_result[k] in [None, ''])


    # cannot pass bad config so no need to test the converse case

    def test_predict_one_threshold(self):
        # invariant under threshold
        thresholds = [x/10 for x in range(0,11,1)]
        pred_error, t_thr = False, None
        well_formed_result = self.model_under_test.predict_one(self.well_formed_record)

        for t in thresholds:
            pred = self.model_under_test.predict_one(record=self.well_formed_record, thr=t)
            t_vals = {pred['tag'], pred.get('rejected', None)}
            wf_vals = {well_formed_result['tag'], well_formed_result.get('rejected', None)}
            set_diff = (t_vals ^ wf_vals) - {'', None}

            if set_diff:  # {e, t, N, t*} ^ {e, t, N}  looking for t*
                pred_error, t_thr = True, t
                break

        self.assertFalse(pred_error, msg=f"pred_one spurious tag: {set_diff}, using thr={t_thr}")


    @staticmethod
    def mutate_file(inpath, outpath, add=None, sub=None):
        """ mutate data file by adding or subtracting fields
            add: dict  eg. {'bad column': 'icky data'}
            sub: list of keys to remove
        """
        # load file
        pre_mut = []
        with open(inpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pre_mut.append(row)

        if add:
            for rec in pre_mut:
                rec.update(add)

        if sub:
            for i, rec in enumerate(pre_mut):
                pre_mut[i] = {k: rec[k] for k in rec if k not in sub}

        fields = pre_mut[0].keys()
        with open(outpath, 'w') as f:
                writer = csv.DictWriter(f, fieldnames = fields)
                writer.writeheader()
                writer.writerows(pre_mut)


    @staticmethod
    def set_compare(test_set, ctrl_set):
        """ check that the return set difference for each record is null
            return error
        """
        if len(test_set) != len(ctrl_set):
            return f"mis-matched counts: {len(test_set)} vs. {len(ctrl_set)}"

        for i in range(len(test_set)):
            t_vals = { test_set[i]['tag'], test_set[i].get('rejected', None) }
            wf_vals = { ctrl_set[i]['tag'], ctrl_set[i].get('rejected', None) }
            diff = (t_vals ^ wf_vals) - {'', None}
            if diff:
                return f"{i} idx, wth diff: {diff}"

        return None
    
    
    ### predict many ###
    # predict_many(self, records_path=None, data_obj=None, thr=0.49, chunk_size=10000)
    def test_predict_many_path(self):
        self.assertRaises(Exception, self.model_under_test.predict_many, '/garbage/path')

    def test_predict_many_empty(self):
        self.assertIsNotNone(self.model_under_test.predict_many(data_obj=self.test_data), msg="pred_many returned None for well formed data")

    def test_predict_many_pred_counts(self):
        well_formed_preds = self.model_under_test.predict_many(data_obj=self.test_data)
        len_wf_preds, len_raw_dat = len(well_formed_preds), len(self.test_data.raw_dat)
        self.assertEqual(len_wf_preds, len_raw_dat, msg=f"pred_many returned: {len_wf_preds}, expected {len_raw_dat}")

    def test_predict_many_return(self):
        well_formed_preds = self.model_under_test.predict_many(data_obj=self.test_data)
        cnt_check, row, count = False, None, None
        for i, res in enumerate(well_formed_preds):
            if len(res.keys()) not in range(PRED_FIELDS[0], PRED_FIELDS[1]):
                cnt_check = True
                row = i + 1
                count = len(res.keys())
                break

        self.assertFalse(cnt_check, msg=f"expected {PRED_FIELDS[0]} to {PRED_FIELDS[1]} fields in return records, got: {count} at rec: {row}")

    def test_predict_many_return_version(self):
        well_formed_preds = self.model_under_test.predict_many(data_obj=self.test_data)
        ver_err = False
        for rec in well_formed_preds:
            if 'version' not in rec:
                ver_err = True
                break

        self.assertFalse(ver_err, msg="predict_many did not return a version")

    def test_predict_many_novocab(self):
        # predictions against mutated data object without vocab
        novocab_data = deepcopy(self.test_data)
        novocab_data.full_vocab, novocab_data.vocabulary_ = None, None
        novocab_preds = self.model_under_test.predict_many(data_obj=novocab_data)
        self.assertIsNone(novocab_preds, msg="pred_many fed no vocab data and returned a result, should be None")

    def test_predict_many_mutated_add(self):
        # predictions against data with malformed fields (and matching configs)
        self.mutate_file('./dat_unit.csv', 'dat_unit_add.csv', add={'bad field':'icky value'})
        config_add = deepcopy(self.model_under_test.raw_config)
        config_add['columns'].update({
            "bad field": {
                "type": "feature",
                "encoding": "one-hot"
            }
        })
        # calling predict_many on malformed data csv should fail
        self.assertRaises(Exception, self.model_under_test.predict_many, './dat_unit_add.csv')
        malf_dat_add = ModelData(config_obj=config_add)
        malf_dat_add.load_csv('./dat_unit_add.csv')
        malf_dat_add.fit_transform(thr=1)  # keep all of the columns
        # with mis-matched configs should return None with logger warning
        self.assertIsNone(self.model_under_test.predict_many(data_obj=malf_dat_add), msg="pred_many returned a result for malformed data")

    def test_predict_many_mutated_sub(self):
        self.mutate_file('./dat_unit.csv', 'dat_unit_sub.csv', sub=['tx_amt'])
        config_sub = deepcopy(self.model_under_test.raw_config)
        config_sub['columns'].pop('tx_amt', None)

        # calling predict_many on malformed data csv should fail
        self.assertRaises(Exception, self.model_under_test.predict_many, './dat_unit_sub.csv')

        malf_dat_sub = ModelData(config_obj=config_sub)
        malf_dat_sub.load_csv('./dat_unit_sub.csv')
        malf_dat_sub.fit_transform()
        # with mis-matched configs should return None with logger warning
        self.assertIsNone(self.model_under_test.predict_many(data_obj=malf_dat_sub), msg="pred_many returned a result for malformed data")

    def test_predict_many_chunks(self):
        # invariant under chunk size
        well_formed_preds = self.model_under_test.predict_many(data_obj=self.test_data)

        chk_vals = [x for x in range(1000,11000,1000)]
        chk_error, chk_iter, err = False, None, None
        for chk in chk_vals:
            preds = self.model_under_test.predict_many(data_obj=self.test_data, chunk_size=chk)
            if len(preds) != len(well_formed_preds):
                chk_error, chk_iter = True, chk
                break
            err = self.set_compare(preds, well_formed_preds)
            if err:
                chk_error, chk_iter = True, chk
                break
        # predict many should return the same result regardless of chunk size
        self.assertFalse(chk_error, msg=f"pred_many returned incorrect results at chk={chk_iter}, {err}")

    def test_predict_many_threshold(self):
        # invariant under threshold
        well_formed_preds = self.model_under_test.predict_many(data_obj=self.test_data)
        thresholds = [x/10 for x in range(0,11,1)]
        thr_error, thr_val, err = False, None, None
        for t in thresholds:
            preds = self.model_under_test.predict_many(data_obj=self.test_data, thr=t)
            err = self.set_compare(preds, well_formed_preds)
            if err:
                thr_error, thr_val = True, t
                break
        # predict many should return a set-equivelent result regarless of threshold (threshold modifies the status of tag only, rejected/tag)
        self.assertFalse(thr_error, msg=f"pred_many produced spurious resuts at thr={thr_val}, {err}")



# directly calling evokes the testing routine
if __name__ == '__main__':
    unittest.main()
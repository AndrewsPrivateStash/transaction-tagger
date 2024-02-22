# pylint: disable=import-error
import sys
from os import path
import datetime, pprint, json
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

projRoot = path.join(path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports

from training.data import ModelData
from training.model import Train
from training.model import TaggingModel

CFG = path.join(projRoot,'config','tab_config_train.json')     # path to tab config file

def get_config():
    with open(CFG) as f:
        return json.load(f)

def check_config(dat_config):
    diffs = dat_config.keys() ^ get_config().keys()
    if diffs:
        return True
    return False

def cross_val(model=None, dat=None, kfolds=10, njobs=-1):
    """
    perform stratified kfold cross validation to produce accuracy estimate
    Parameters
    ----------
    model : obj
        the trained model to be tested (self.model)
    dat : obj
        the data object used to train the model
    kfolds : int
        number of stratisfied kfolds to generate in cross validation
    njobs : int
        number of jobs to spawn, -1 is all
    """
    if model is None:
        print("cross validation called without estimation model")
        return

    print(f'cross validating model with {kfolds} folds')
    start = datetime.datetime.now()
    cross_val = cross_val_score(model, dat.xmat, dat.ys, cv=kfolds, n_jobs=njobs, verbose=3)
    end = datetime.datetime.now()

    print(f'cross val finished in: {str(end - start)} with niave accuracy: {cross_val.mean()}')
    res =   {
            'cross_validation': {
                'strat kfolds': kfolds,
                'accuracy': "{:0.4f} (+/- {:0.4f})".format(cross_val.mean(), cross_val.std() * 2),
                'cross time': str(end - start)
                }
            }

    return res


def main():
    """ ARGS:
            Str model, the path in /data to the trained model to be tested
            Int folds, the number of folds to use in cross val (0 means skip)
            Int cores, the number of cores to use in cross val (-1 is all)
            Str dat_path, the path to the data object or the data csv
            Str unseen_path, the path to unseen data to test (empty is no test)

        always provide model (first arg)
        if folds is not zero then cross validation is performed
            any valid value may be passed if folds is zero for the cores and dat_path (but not empty)
        if unseen_path containes a path then OOS testing is done
        
    """

    if len(sys.argv)-1 < 2:
        print("usage $ python model_testing.py <model path> <folds> <cores> <dat path> <unseen path>\n")
        sys.exit(1)

    if not str.isdigit(sys.argv[2].strip()):
        print("second argument is the number of folds to use in x-val, please provide an integer value\n")
        sys.exit(1)

    if not str.isdigit(sys.argv[3].strip()):
        print("third argument is the number of cores to use in x-val (-1 is all), please provide an integer value\n")
        sys.exit(1)

    MODELPATH = path.join(projRoot,'data', sys.argv[1].strip())
    OUTMODPATH = path.join(projRoot,'data', path.splitext(path.split(MODELPATH)[1])[0] + "_tested.gz")
    FOLDS = int(sys.argv[2].strip())
    CORES = int(sys.argv[3].strip())
    DATAPATH = None
    if FOLDS != 0 and len(sys.argv)-1 >= 4:
        DATAPATH = path.join(projRoot,'data', sys.argv[4].strip())

    UNSEENPATH = None
    if len(sys.argv)-1 >= 5:
        UNSEENPATH = path.join(projRoot,'data', sys.argv[5].strip())

    # print what was recieved for clarity
    print(f"received:\nmodel path: {MODELPATH}\n  x-val\n\t{FOLDS} folds\n\t{CORES} cores\n\tpath: {DATAPATH}\n  unseen-test\n\t{UNSEENPATH}")

    print("loading base model for update and use..")
    mod = TaggingModel()
    mod.load_model(model_path=MODELPATH)


    ##### cross validation #####
    if FOLDS != 0:

        dat = ModelData(config_path=CFG)
        if path.splitext(DATAPATH)[1][1:] == "gz":
            dat.load_data_obj(DATAPATH)
            if dat.xmat is None:
                print(f"loaded data object {DATAPATH} has no xmat (not fit)")
                sys.exit(1)
            if check_config(dat.CONFIG):
                print(f"config in {DATAPATH} doesn't match train config: {CFG}")
                sys.exit(1)

        elif path.splitext(DATAPATH)[1][1:] == "csv":
            dat.load_csv(DATAPATH)
            dat.fit_transform()
        else:
            print(f"{DATAPATH} was not a .gz or .csv file, exiting..")
            sys.exit(1)

        print("starting cross validation..\n")
        res = cross_val(model=mod.uncal_model, dat=dat, kfolds=FOLDS, njobs=CORES)
        pprint.pprint(res)
        mod.meta.update(res)        

        # free memory
        dat = None


    ##### unseen test #####
    if UNSEENPATH:
        print("starting unseen test..")
        res = mod.test(test_path=UNSEENPATH, thr=0.85)
        pprint.pprint(res)
        mod.meta.update({
            "OOS Test" : res
        })


    ##### Output #####
    print("saving test results to model meta with postfix '_tested'..")
    mod.save_model(OUTMODPATH)
    print("\ntests complete!")


if __name__ == '__main__':
    main()
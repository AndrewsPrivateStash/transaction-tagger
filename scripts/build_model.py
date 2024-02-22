# pylint: disable=import-error
import sys
from os import path
import json

projRoot = path.join(path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports

from training.data import ModelData
from training.model import Train

CFG = path.join(projRoot,'config','tab_config_train.json')     # path to config file (config excludes txnid)

def get_config():
    with open(CFG) as f:
        return json.load(f)

def check_config(dat_config):
    diffs = dat_config.keys() ^ get_config().keys()
    if diffs:
        return True
    return False


def main():
    """ ARGS:
            Data Type:  { obj, csv }
            Data Path
            Model Out Path
            Model Version
    """

    if len(sys.argv)-1 != 4:
        # first argument is csv or obj declaring what type of data to load
        print("usage $ python build_model.py {csv,obj} data_path model_out_path version_name\n")
        sys.exit(1)
    
    if sys.argv[1].lower() not in["csv", "obj"]:
        print("first argument is either csv or obj")
        sys.exit(1)

    # data and model files assumed in: ./data/
    DATAPATH = path.join(projRoot,'data', sys.argv[2].strip())
    MODPATH = path.join(projRoot,'data', sys.argv[3].strip())
    VERSION = sys.argv[4]
    
    print(f"received:\n\tdatatype {sys.argv[1]}\n\tdata path {DATAPATH}\n\tmodel path {MODPATH}\n\tmodel version {VERSION}\n")
    
    # construct or load data object
    dat = ModelData(config_path=CFG)
    if sys.argv[1].lower() == "csv":
        dat.load_csv(DATAPATH)
        dat.fit_transform()
    elif sys.argv[1].lower() == "obj":
        dat.load_data_obj(DATAPATH)
        if dat.xmat is None:
            print(f"loaded data object {sys.argv[2]} has no xmat (not fit)")
            sys.exit(1)
        if check_config(dat.CONFIG):
            print(f"config in {DATAPATH} doesn't match train config: {CFG}")
            sys.exit(1)
    else:
        print("first argument is either csv or obj")
        sys.exit(1)

    # train model
    print("training model..")
    tr = Train()
    tr.load_data(dat)
    tr.train_model(
        c_penalty=0.1,
        weights='balanced',
        #output_path=MODPATH,
        version=VERSION
    )

    # perform model score calculation and save model
    print("calculating model score..")
    tr.calc_model_score()
    tr.save_model(MODPATH)



if __name__ == '__main__':
    main()
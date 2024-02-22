# pylint: disable=import-error
import sys
from os import path
import json, csv

projRoot = path.join(path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports

from training.data import ModelData
from training.model import TaggingModel

CFG = path.join(projRoot,'config','tab_exc_config_train.json')     # path to config file (config excludes txnid)
BND = path.join(projRoot,'data','brands.json')

def getBrands(brand_path=BND):
    with open(brand_path, 'r') as f:
        return json.load(f)

def load_csv(filepath):
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

def write_dict(outpath, dict_data):
    fields = dict_data[0].keys()
    with open(outpath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames = fields)
        writer.writeheader()
        writer.writerows(dict_data)

def main():
    """ ARGS:
            Str model, the path in /data to the trained model to be tested
            Str unseen_path, the path to unseen data csv
            Str output_path, the path to the output file

            The fields are expected to be { tx_desc, tx_amt, brand_name }
            returns a csv file with tag, brand and prob to the input file
    """

    if len(sys.argv)-1 < 3:
        print("usage $ python predict_tags_file.py <model path> <unseen path> <output path>\n")
        sys.exit(1)

    MODELPATH = path.join(projRoot,'data', sys.argv[1].strip())
    UNSEENPATH = path.join(projRoot,'data', sys.argv[2].strip())
    OUTPUT = path.join(projRoot,'data', sys.argv[3].strip())

    # print what was recieved for clarity
    print(f"received:\nmodel path: {MODELPATH}\nunseen data\t{UNSEENPATH}\noutput\t{OUTPUT}")

    print("loading base model for update and use..")
    mod = TaggingModel()
    mod.load_model(model_path=MODELPATH)

    ##### unseen predictions #####
    if UNSEENPATH:
        print("starting unseen predictions..")
        res = mod.predict_many(records_path=UNSEENPATH, thr=0.80)
        print(f"processed: {len(res)} records")
        #print(res[0].keys())
        
        # combine results and unseen ids
        infile = list(load_csv(UNSEENPATH))
        out_dict_temp = [{
            "tx_desc" : i["tx_desc"],
            "tx_amt" : i["tx_amt"],
            "brand_name" : i["brand_name"],
            "tag" : t["tag"],
            "prob" : t["prob"]
            }
            for (i, t) in zip(infile, res)    
        ]

        # only show the mismatched records
        out_dict = []
        for r in out_dict_temp:
            if r["brand_name"] != r["tag"]:
                out_dict.append(r)


    ##### Output #####
    if out_dict:
        print(f"saving predictions to: {OUTPUT} ..")
        write_dict(OUTPUT, out_dict)
    else:
        print("no bad tags found!")


if __name__ == '__main__':
    main()
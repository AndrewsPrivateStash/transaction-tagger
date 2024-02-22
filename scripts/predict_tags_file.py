# pylint: disable=import-error
import sys
from os import path
import json, csv

projRoot = path.join(path.dirname(__file__), '..')
sys.path.append(projRoot)
# hack to include the parent for imports

from training.data import ModelData
from training.model import TaggingModel

CFG = path.join(projRoot,'config','tab_config_train.json')     # path to config file (config excludes txnid)
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

            assumes data file has transactionId present to join after predictions
            The fields are expected to be { tx_desc, tx_amt, transactionId }
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
        res = mod.predict_many(records_path=UNSEENPATH, thr=0.66)
        print(f"processed: {len(res)} records")
        
        #augment the results with the brand names
        brands = getBrands()
        res_b = []
        for rec in res:
            res_b.append({
                "tag" : rec["tag"],
                "brand" : brands.get(rec["tag"], "NULL"),
                "prob" : rec["prob"]
            })


        # combine results and unseen ids
        infile = list(load_csv(UNSEENPATH))
        out_dict = [{
            "transactionId" : i["transactionId"],
            "tx_desc" : i["tx_desc"],
            "tx_amt" : i["tx_amt"],
            "tag" : t["tag"],
            "brand" : t["brand"],
            "prob" : t["prob"]
            }
            for (i, t) in zip(infile, res_b)    
        ]

    ##### Output #####
    print(f"saving predictions to: {OUTPUT} ..")
    write_dict(OUTPUT, out_dict)
    


if __name__ == '__main__':
    main()
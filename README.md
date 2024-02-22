# transaction-tagger
classes used in training tagging models and predicting values.

### ModelData Class
Pre-processing class for subsequent use by a Train, TaggingModel and GreedyTaggingModel classes.
Responsible for processing a set of records into a document term matrix and performing feature reduction via a chi2 process and storing associated vocabularies.
Expects source data with headers and encoding defined in config json.
y-values are optional (needed for subsequent training/testing, but not predictions).

* typical method sequence: {init, load\_csv, fit\_transform}

##### Methods:
* init: ModelData(vocab=None, config\_path='./config/data\_config.json', config\_obj=None)
* load\_record(record)
* load\_csv(filepath)
* load\_data\_obj(data\_path)
* save\_data\_obj(output\_path)
* fit\_transform(thr=0.05, overwrite=False)


### Train Class
Wraps sklearn classes and controls the training, testing, and saving of models.

* typical method sequence: {init, load\_dat, train, save\_model, test}

##### Methods:
* init: Train()
* load\_data(data\_obj)
* train\_model(c\_penalty=1, weights=None, output\_path=None, version=None) // svm model
* train\_model\_rf(n\_est=100, max\_depth=None, max\_leaf=None, njobs=None, verbose=0, version=None, output\_path=None) // random forest model
* calc\_model\_score(chunk\_size=10000)
* save\_model(output\_path)
* load\_model(model\_path, data\_obj)
* cross\_val(kfolds=10, njobs=-1)
* test(thr=0.49, ret=False, cls\_dim=False, chunk\_size=10000)


### TaggingModel Class
Wraps sklearn model to allow for predictions and/or testing against transaction records.   
This is the class used in production for tagging inbound transactions

* typical method sequence: {init, load\_model, predict\_one}

##### Methods:
* init: TaggingModel(model\_obj=None)
* predict\_one(record, thr=0.49)
* predict\_many(records\_path=None, data\_obj=None, thr=0.49, chunk\_size=10000)
* exc\_predict\_one(record, thr=0.85)  //called automatically from predict\_one, if exclusion model loaded
* test(test\_path=None, data\_obj=None, thr=0.49, cls\_dim=False)
* load\_model(model\_path)
* save\_model(model\_path)
* load\_exc\_model\_obj(model\_path)  //side loads an exclusion model into the object
* get\_version()
* get\_exc\_version()
* get\_config()
* get\_exc\_config()

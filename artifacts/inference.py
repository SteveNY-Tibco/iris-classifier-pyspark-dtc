# inference.py

import sys
import os

from pyspark.ml.classification import DecisionTreeClassificationModel

import pyspark
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

#if ('sc' not in locals() or 'sc' not in globals()):
#    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python2'

sc = pyspark.SparkContext('local[*]')
spark = SQLContext(sc)

class Inference:
    def __init__(self, config):
        self.model = DecisionTreeClassificationModel.load(config['Model'])
        self.labels = config['Labels'].split(',')
    
    def evaluate(self, payload):
        d = [{'features': Vectors.dense(payload), 'labelIndex' : 0.0}]
        data = spark.createDataFrame(d)
        prediction = self.model.transform(data)
        return self.labels[int(prediction.select('prediction').first()[0])]

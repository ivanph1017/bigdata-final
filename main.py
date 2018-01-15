from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression  #encontraremos una recta (h) para decidir si ciertos valores se clasifican para un lado o para el otro
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql import Row
from operator import add
import math
import re


## split y parseo
def clean_process(sc, url):
    rdd = sc.textFile(url).map(lambda linea: linea.split(","))
    return rdd

## VectorAssembler: Convertir el dataframe en uno que tenga una columna llamada output_col compuesta por las columnas @features_array.
def convert_dataframe(data, features_array, output_col, label_col):
    assembler = VectorAssembler(inputCols= features_array, outputCol= output_col)
    return assembler.transform(data).select(output_col, label_col)
    #test_data = assembler.transform(test).select("features","PRE_POS")

## Obtenemos un modelo de Regresion Logistica dependiendo de familyName
## con cross validation
def logistic_regression(data, label_col, metric_name, familyName):
    lr = LogisticRegression(maxIter=100, labelCol=label_col, family=familyName)

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.elasticNetParam, [0.1, 0.01]) \
        .build()

    crossVal = CrossValidator(estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=MulticlassClassificationEvaluator(
            labelCol=label_col,
            metricName=metric_name,
            predictionCol='prediction'
        ),
        numFolds=4
    )

    return crossVal.fit(data)

##Mostrar parametros multiclase
def parameters_lr_multiclass(lr_model):
    print("Coefficients: " + str(lr_model.bestModel.coefficientMatrix))
    print("Intercept: " + str(lr_model.bestModel.interceptVector))

##Metodo que realizar una evaluacion
def evaluate_model_regression(label_col, metric_name, data_to_validate):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, \
                                                  metricName=metric_name, predictionCol='prediction')
    value = evaluator.evaluate(data_to_validate)
    print("{}:{}".format(metric_name, value))
    ## devuelve el valor del evaluator, accuracy
    return value

##Ejecuciòn de modelo  multiclase de regresion logistica
def execute_logistic_regression_multiclass(sc, spark):
    ''' Regresion Multiclase '''
    print("------------ Regresion multiclase --------------")

    lyrics_rdd = clean_process(sc, 'data/lyrics.csv')

    # Limpieza de dataset
    base_rdd = lyrics_rdd.map(lambda x: (''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[0]).split()),
                                         ''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[1]).split()),
                                         ''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[2]).split()),
                                         ''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[3]).split()),
                                         ''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[4]).split()),
                                         ''.join(re.sub(r'[^a-zA-Z0-9\._-]', '', x[5]).split()))) \
        .filter(lambda x: x is not None) \
        .filter(lambda x: x[0] is not None and x[1] is not None and \
                x[2] is not None and x[3] is not None and \
                x[4] is not None and x[5] is not None) \
        .filter(lambda x: len(x[0]) > 0 and len(x[1]) > 0 \
                and len(x[2]) > 0 and len(x[3]) > 0 and len(x[4]) > 0 and len(x[5]) > 0)

    base_features_rdd = base_rdd.map(lambda x: (x[2], x)) \
        .groupByKey() \
        .mapValues(list) \
        .zipWithIndex() \
        .flatMap(lambda x: generator(x))

    # total de canciones
    songs_total = base_features_rdd.map(lambda x: (x[1], x)) \
		.groupByKey() \
		.mapValues(list) \
		.count()

    print("Total de canciones... " + str(songs_total))

    # Los generos (labels)
    genres = base_features_rdd.map(lambda x: x[5]) \
		.distinct() \
		.collect()

    print("Init df_dict...")
    # Número total de canciones que contienen la palabra
    df_dict = base_features_rdd.map(lambda x: (x[2], x)) \
    	.sortByKey() \
		.groupByKey() \
		.mapValues(list) \
		.map(lambda x: (x[0], len(x[1]))) \
		.collectAsMap()

    # Headers
    headers = ['genre']
    headers.extend(list(map(lambda x: str(x), df_dict.keys())))

    # Procesar canciones
    rdd_song_ly = base_features_rdd.map(lambda x: (x[1], x)) \
    	.sortByKey() \
	    .groupByKey() \
		.mapValues(list) \
		.map(lambda x: map_song(x, genres, df_dict, songs_total))

    data = spark.createDataFrame(rdd_song_ly, headers).na.fill(0)
    # data.show()
    train, test = data.randomSplit([0.7,0.3], seed=12345)
    # train.show()

    features = headers[1:]
    output = 'features'
    label_col = 'genre'
    train_data = convert_dataframe(train, features, output, label_col)
    train_data.show()

    print("Encontrando h ...")

    metric_name = 'accuracy'
    lr_model_multiclass = logistic_regression(train_data, label_col,
        metric_name, 'multinomial')
    parameters_lr_multiclass(lr_model_multiclass)

    print("Testing model ...")

    test_data = convert_dataframe(test, features, output, label_col)

    data_to_validate = lr_model_multiclass.transform(test_data)
    # data_to_validate.show()

    ## pinta el valor del evaluator, valor de 'accuracy'
    evaluate_model_regression(label_col, metric_name, data_to_validate)

# Se mapea los campos
def generator(x):
    for lyric in x[0][1]:
        yield (lyric[0], lyric[1], x[1], lyric[3], lyric[4], lyric[5])

# procesar lyric_row
def map_song(row, genres, df_dict, songs_total):
	genre = genres.index(row[1][0][5])
	song_list = [genre]
	word_list = []
	for word_index in df_dict.keys():
		word_list.append(0)
	for lyric in row[1]:
		word_index = lyric[2]
		count = lyric[3]
		# se calcula el TF-IDF
		word_list[word] = calc_tf_idf(count, word_index, df_dict, songs_total)
	song_list.extend(word_list)
	return song_list

# calcular tf_idf
def calc_tf_idf(count, word_index, df_dict, songs_total):
	tf = int(count)
	w = tf * math.log10(songs_total / df_dict[word_index])
	return w

def main():

    conf = SparkConf().setAppName('Songs').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    ##Llamada a funciones de regresiones logisticas
    execute_logistic_regression_multiclass(sc, spark)

if __name__ == '__main__':
    main()
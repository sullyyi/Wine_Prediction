#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(test_path, model_path):
    spark = SparkSession.builder.appName("WineQualityPredict").getOrCreate()

    #load test data (semicolon-delimited, strip quotes)
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .option("sep", ";")
          .option("quote", '"')
          .csv(test_path))

    #index the 'quality' column into 'label'
    idx = StringIndexer(inputCol="quality", outputCol="label")
    df  = idx.fit(df).transform(df)

    #assemble the features vector
    feature_cols = [c for c in df.columns if c not in ("quality","label")]
    assembler    = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data         = assembler.transform(df)

    #load the trained model
    model = LogisticRegressionModel.load(model_path)

    #predict + evaluate
    preds     = model.transform(data)
    evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="f1")
    f1 = evaluator.evaluate(preds)
    print(f"Test F1 score = {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine quality prediction")
    parser.add_argument("test_csv",  help="Path to test CSV")
    parser.add_argument("model_dir", help="Path to saved model directory")
    args = parser.parse_args()
    main(args.test_csv, args.model_dir)

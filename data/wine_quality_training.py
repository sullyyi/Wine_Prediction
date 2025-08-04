#!/usr/bin/env python3

import argparse
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    # Initialize Spark
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    # Load datasets
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    val_df   = spark.read.csv(val_path,   header=True, inferSchema=True)

    # Identify feature columns (all except the label 'quality')
    feature_cols = [c for c in train_df.columns if c != "quality"]

    # Index the label column
    idx = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = idx.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    # Assemble features into a vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = assembler.transform(train_df)
    val_data   = assembler.transform(val_df)

    # Train a logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    model = lr.fit(train_data)

    # Evaluate on validation set
    preds = model.transform(val_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score = {f1:.4f}")

    # Save the model (overwrite if exists)
    if spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        ).exists(spark._jvm.org.apache.hadoop.fs.Path(output_path)):
        shutil.rmtree(output_path)
    model.save(output_path)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

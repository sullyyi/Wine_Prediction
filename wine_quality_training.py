#!/usr/bin/env python3
import argparse
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    #load CSVs with semicolon + quote handling
    train_df = (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("sep", ";")
                .option("quote", '"')
                .csv(train_path))

    val_df = (spark.read
              .option("header", True)
              .option("inferSchema", True)
              .option("sep", ";")
              .option("quote", '"')
              .csv(val_path))

    #index the label column (quality to label)
    idx = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = idx.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    #assemble feature vector
    feature_cols = [c for c in train_df.columns if c not in ("quality", "label")]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = assembler.transform(train_df)
    val_data   = assembler.transform(val_df)

    #train logistic regression
    lr    = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    model = lr.fit(train_data)

    #persist model on disk
    model.write().overwrite().save(output_path)

    #evaluate on validation set
    preds     = model.transform(val_data)
    evaluator = MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score = {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a wine quality model")
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save the trained model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    spark = SparkSession.builder.appName("WineQualityRFTraining").getOrCreate()

    #load CSVs (semicolon-delimited, strip quotes)
    def load(path):
        return (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("sep", ";")
                .option("quote", '"')
                .csv(path))
    train_df = load(train_path)
    val_df   = load(val_path)

    #index the label column
    idx = StringIndexer(inputCol="quality", outputCol="label").fit(train_df)
    train_df = idx.transform(train_df)
    val_df   = idx.transform(val_df)

    #assemble feature vector
    feature_cols = [c for c in train_df.columns if c not in ("quality","label")]
    assembler    = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data   = assembler.transform(train_df)
    val_data     = assembler.transform(val_df)

    #train a Random Forest classifier
    rf    = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=100,
                maxDepth=5,
                seed=42)
    model = rf.fit(train_data)

    #persist the model
    model.write().overwrite().save(output_path)

    #evaluate on validation set
    preds     = model.transform(val_data)
    evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="f1")
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score (RF) = {f1:.4f}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save a wine-quality model using Random Forest")
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save the trained RF model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

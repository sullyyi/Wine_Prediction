#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    spark = SparkSession.builder.appName("WineQualityTrainingGBT").getOrCreate()

    #load CSVs (semicolon-delimited, strip quotes)
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

    #index the label column (quality → label)
    idx = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = idx.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    #assemble feature vector
    feature_cols = [c for c in train_df.columns if c not in ("quality", "label")]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = assembler.transform(train_df)
    val_data   = assembler.transform(val_df)

    #build a multiclass GBT via OneVsRest
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )
    ovr = OneVsRest(
        classifier=gbt,
        labelCol="label",
        featuresCol="features"
    )
    model = ovr.fit(train_data)

    #save the model
    model.write().overwrite().save(output_path)

    #evaluate on validation set
    preds = model.transform(val_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score = {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wine Quality GBT (multiclass via OneVsRest)")
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save the trained model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

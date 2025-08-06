import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    spark = SparkSession.builder \
        .appName("WineQuality_GBT_Direct") \
        .getOrCreate()

    # Load CSVs with header, schema inference, and semicolon delimiter
    def load(path):
        return (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("sep", ";")
                .option("quote", '"')
                .csv(path))

    train_df = load(train_path)
    val_df   = load(val_path)

    # Index the "quality" column into numeric labels
    indexer = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = indexer.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    # Assemble all feature columns into a single vector
    feature_cols = [c for c in train_df.columns if c not in ("quality", "label")]
    assembler    = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data   = assembler.transform(train_df)
    val_data     = assembler.transform(val_df)

    # Instantiate GBTClassifier with best-found hyperparameters
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        seed=42,
        maxDepth=5,
        maxIter=50,
        stepSize=0.1
    )

    # Wrap in One-vs-Rest for multiclass support
    ovr = OneVsRest(classifier=gbt, labelCol="label")

    # Train the final model
    ovrModel = ovr.fit(train_data)

    # Save the model for serving
    ovrModel.write().overwrite().save(output_path)

    # Evaluate on validation set
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    preds = ovrModel.transform(val_data)
    f1 = evaluator.evaluate(preds)
    print(f"[GBT_Direct] Validation F1 score = {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct GBT training with best hyperparameters"
    )
    parser.add_argument(
        "--train", 
        default="data/TrainingDataset-1.csv",
        help="Path to training CSV (default: data/TrainingDataset-1.csv)"
    )
    parser.add_argument(
        "--validate", 
        default="data/ValidationDataset-1.csv",
        help="Path to validation CSV (default: data/ValidationDataset-1.csv)"
    )
    parser.add_argument(
        "--output",   
        default="model_gbt",
        help="Directory to save the GBT model (default: model_gbt)"
    )
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

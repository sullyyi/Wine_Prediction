#this model is a tuned GBT classifier using One-vs-Rest strategy
#it uses cross-validation to find the best hyperparameters
#it is designed to work with the wine quality dataset
#the model is saved to the specified output directory
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def main(train_path, val_path, output_path):
    spark = SparkSession.builder.appName("WineQualityTrainingGBT_CV").getOrCreate()

    #load CSVs
    def load(path):
        return (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("sep", ";")
                .option("quote", '"')
                .csv(path))
    train_df = load(train_path)
    val_df   = load(val_path)

    #index labels
    idx = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = idx.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    #assemble features
    feature_cols = [c for c in train_df.columns if c not in ("quality","label")]
    assembler    = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data   = assembler.transform(train_df)
    val_data     = assembler.transform(val_df)

    #build a tuned GBT within One-vs-Rest
    gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
    ovr = OneVsRest(classifier=gbt, labelCol="label")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")

    #param grid: attempt to try a small search
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [3, 5, 7])
                 .addGrid(gbt.maxIter, [20, 50, 100])
                 .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
                 .build())

    cv = CrossValidator(estimator=ovr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3,
                        parallelism=4)  #models to train in parallel

    #fit and select best model
    cvModel = cv.fit(train_data)

    #save the best OvR-GBT model
    cvModel.bestModel.write().overwrite().save(output_path)

    #evaluate on validation
    preds = cvModel.transform(val_data)
    f1 = evaluator.evaluate(preds)
    print(f"[CV-GBT] Validation F1 score = {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBT + CV for Wine Quality")
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save the best model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

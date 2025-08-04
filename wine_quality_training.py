import argparse
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main(train_path, val_path, output_path):
    #initialize Spark session
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    #load datasets
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    val_df   = spark.read.csv(val_path,   header=True, inferSchema=True)

    #identify feature columns except the label for quality
    feature_cols = [c for c in train_df.columns if c != "quality"]

    #index the label column
    idx = StringIndexer(inputCol="quality", outputCol="label")
    idx_model = idx.fit(train_df)
    train_df = idx_model.transform(train_df)
    val_df   = idx_model.transform(val_df)

    #assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_data = assembler.transform(train_df)
    val_data   = assembler.transform(val_df)

    #logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    model = lr.fit(train_data)

    #evaluation on validation set
    preds = model.transform(val_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score = {f1:.4f}")

    #save the model
    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    model_path = spark._jvm.org.apache.hadoop.fs.Path(output_path)
    if fs.exists(model_path):
        fs.delete(model_path, True)
    model.save(output_path)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a wine quality model")
    parser.add_argument("--train",    required=True, help="Path to training CSV")
    parser.add_argument("--validate", required=True, help="Path to validation CSV")
    parser.add_argument("--output",   required=True, help="Directory to save the trained model")
    args = parser.parse_args()
    main(args.train, args.validate, args.output)

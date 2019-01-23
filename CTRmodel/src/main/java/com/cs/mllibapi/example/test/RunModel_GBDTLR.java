package com.cs.mllibapi.example.test;

import com.cs.mllibapi.evaluation.RankingEvaluator;
import com.cs.mllibapi.rankmodel.GBDTAndLRRankModel;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;


// spark-submit --driver-memory 128G --executor-memory 128G --conf spark.driver.maxResultSize=128g  --master localhost[4] --class "com.cs.mllibapi.example.RunModel"  CTRmodel-1.0-SNAPSHOT.jar 1
public final class RunModel_GBDTLR {

    //输入特征
    private final static String feature_save_file = "data/dataprocess/result/features";

    public static void main(String[] args) throws Exception {
        Integer gbdt_MaxDepth = Integer.parseInt(args[0]);
        Integer gbdt_numTrees = Integer.parseInt(args[1]);

        Logger logger = Logger.getLogger("org");
        logger.setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("app").setMaster("local[4]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // 加载数据，来自data_process.FeatureEngineer
        JavaRDD<LabeledPoint> data = jsc.objectFile(feature_save_file);

        // 划分训练、测试数据
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 12345);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1].cache();

        GBDTAndLRRankModel rankModel = new GBDTAndLRRankModel();
        System.out.println("使用模型：" + rankModel.getClass().getSimpleName());

        //   rankModel.setGbdt_numTrees(gbdt_numTrees);
        //   rankModel.setGbdt_MaxDepth(gbdt_MaxDepth);

        //  System.out.println("MaxDepth=" + rankModel.getGbdt_MaxDepth() + ",numTrees=" + rankModel.getGbdt_numTrees());

        rankModel.train(training);

        //执行预测（label）
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = rankModel.transform(test);
        predictionAndLabels.cache();

        // 评测
        RankingEvaluator rankingEvaluator = new RankingEvaluator();
        rankingEvaluator.evaluate("GBDT+LR", predictionAndLabels);


        jsc.close();

    }
}

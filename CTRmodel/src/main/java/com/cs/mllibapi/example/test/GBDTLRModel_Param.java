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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

//参数调优
// spark-submit --driver-memory 128G --executor-memory 128G --conf spark.driver.maxResultSize=128g  --master localhost[8] --class "com.cs.mllibapi.example.GBDTLRModel_Param"  CTRmodel-1.0-SNAPSHOT.jar
public final class GBDTLRModel_Param {

    //输入特征
    private final static String feature_save_file = "data/dataprocess/result/features";

    public static void main(String[] args) throws Exception {
        Logger logger = Logger.getLogger("org");
        logger.setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("app").setMaster("local[4]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        jsc.setLogLevel("WARN");

        // 加载数据【来自data_process.FeatureEngineer】
        JavaRDD<LabeledPoint> data = jsc.objectFile(feature_save_file);

        // 划分训练、测试数据
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 12345);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1].cache();

        System.out.println("训练集数量=" + training.count() + ",测试集数量=" + test.count());


        GBDTAndLRRankModel model = new GBDTAndLRRankModel();
        System.out.println("Rank Model = " + model.getClass().getSimpleName());

        Integer[] gbdt_MaxDepth_list = {8, 13, 21, 34, 55};
        Integer[] gbdt_numTrees_list = {2, 5, 8, 13, 21};

        for (Integer maxdepth : gbdt_MaxDepth_list) {
            for (Integer numtrees : gbdt_numTrees_list) {

                // model.setGbdt_MaxDepth(maxdepth);
                //  model.setGbdt_numTrees(numtrees);

                model.train(training);

                //执行预测（label）
                JavaRDD<Tuple2<Object, Object>> predictionAndLabels = model.transform(test);
                predictionAndLabels.cache();

                // 评测
                RankingEvaluator rankingEvaluator = new RankingEvaluator();
                String result = rankingEvaluator.evaluate("GBDT+LR", predictionAndLabels);

                //如果文件存在，则追加内容；如果文件不存在，则创建文件
                FileWriter fw = null;
                try {
                    File f = new File("model_log.txt");
                    fw = new FileWriter(f, true);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                PrintWriter pw = new PrintWriter(fw);

                pw.println();
                //    pw.println("gbdt maxdepth=" + model.getGbdt_MaxDepth());
                //     pw.println("gbdt numtrees=" + model.getGbdt_numTrees());
                pw.println(result);

                pw.flush();
                try {
                    fw.flush();
                    pw.close();
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("\n\n");
                predictionAndLabels.unpersist();
            }
        }


        jsc.close();

    }
}

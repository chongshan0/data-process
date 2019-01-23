package com.cs.mlapi.example;

import com.cs.mlapi.evaluation.RankingEvaluator;
import com.cs.mlapi.rankmodel.BaseRankModel;
import com.cs.mlapi.rankmodel.LogisticRegressionRankModel;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;


public final class RunModel_Ml {

    //特征
    private static final String ctrmodel_feature_file = "CTRmodel/data/sample_libsvm_data.txt";

    public static void main(String[] args) throws Exception {
        Logger logger = Logger.getLogger("org");
        logger.setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf().setAppName("app").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new SQLContext(jsc);

        // 加载数据
        DataFrame features = sqlContext.read().format("libsvm").load(ctrmodel_feature_file);
        System.out.println("排序模型特征：");
        features.printSchema();
        features.show(10);

        // 划分训练、测试数据
        DataFrame[] splits = features.randomSplit(new double[]{0.7, 0.3}, 12345);
        DataFrame training = splits[0];
        DataFrame test = splits[1];

        //test
//        System.out.println("test");
//        StringIndexer labelIndexer = new StringIndexer()
//                .setInputCol("label")
//                .setOutputCol("indexedLabel");
//        labelIndexer.fit(training).transform(training).javaRDD().collect().forEach(s -> System.out.println(s.get(0) + " " + s.get(2)));

        BaseRankModel rankModel;    //排序模型抽象类

        //选择模型
        rankModel = new LogisticRegressionRankModel();
        //rankModel = new GBDTRankModel(); //error
        //rankModel = new GBDTLRRankModel();
        System.out.println("Rank Model = " + rankModel.getClass().getSimpleName());

        //训练
        rankModel.train(training);

        //执行预测（label），返回包含预测结果的DataFrame
        DataFrame predictions = rankModel.transform(test);
        System.out.println("预测结果：");
        predictions.printSchema();
        predictions.show();

        //评测
        RankingEvaluator rankingEvaluator = new RankingEvaluator();
        System.out.println("评测：");
        rankingEvaluator.evaluate(predictions);


        jsc.close();

    }
}

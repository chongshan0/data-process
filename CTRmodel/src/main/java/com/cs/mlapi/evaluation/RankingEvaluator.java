package com.cs.mlapi.evaluation;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.sql.DataFrame;
import scala.Tuple2;

//排序模型评测器
public class RankingEvaluator {

    /**
     * 评测排序模型
     *
     * @param predictions 预测结果DataFrame，“包含” label、probability
     *                    label：真实值
     *                    features：输入特征
     *                    probability：二元vector，0和1的预测概率
     *                    prediction：预测label
     */
    public void evaluate(DataFrame predictions) {

        //找到预测值和真实值
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = predictions.select("label", "probability")
                .javaRDD()
                .map(row -> new Tuple2<>(((DenseVector) row.get(1)).apply(1), row.get(0)));

        //test
        System.out.println("（预测值，真实值）：");
        scoreAndLabels.collect().forEach(s -> System.out.println(s));

        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(scoreAndLabels.rdd());

        System.out.println("评测结果：");    // https://www.cnblogs.com/charlesblc/p/6252759.html
        System.out.println("AUC under PR = " + metrics.areaUnderPR());   //越大越好
        System.out.println("AUC under ROC = " + metrics.areaUnderROC());    //越大越好
    }
}

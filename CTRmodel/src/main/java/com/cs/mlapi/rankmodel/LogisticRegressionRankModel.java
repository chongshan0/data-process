package com.cs.mlapi.rankmodel;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.sql.DataFrame;


//逻辑回归
public class LogisticRegressionRankModel extends BaseRankModel {

    /**
     * 训练模型得到pipelineModel
     *
     * @param training 训练数据DataFrame
     *                 包括2列：label: double、features: vector
     *                 注意：数据是已经处理好的，执行train()时，不会对label和特征进行变换
     */
    @Override
    public void train(DataFrame training) {

        //创建LogisticRegression实体（一个Estimators，将一个DataFrame转换为一个Model实体）
        LogisticRegression logisticRegression = new LogisticRegression()
                .setMaxIter(10)     //最大迭代次数
                .setRegParam(0.01)  //正则化系数
                .setElasticNetParam(0.0)    //弹性参数，调节L1和L2正则化之间的比例，两种正则化比例加起来是1。0表示只用L2，1表示只用L1
                .setFeaturesCol("features")   //指明Feature在哪一列
                .setLabelCol("label");   //指明label在哪一列

        //获取PipelineStage[]，这里不需要处理数据，所以只有一个stage
        PipelineStage[] pipelineStages = new PipelineStage[]{logisticRegression};
        //装配PipelineStage，生成需要的Pipeline
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);

        //调用fit()获得pipelineModel
        pipelineModel = pipeline.fit(training);
    }

}

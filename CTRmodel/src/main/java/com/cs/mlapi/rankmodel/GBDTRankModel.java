package com.cs.mlapi.rankmodel;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.DataFrame;


//梯度增强决策树
// todo 确认在spark-1.6.1版本，GBTClassifier和GBTRegressor，均只能获得预测label，无法获得预测的probability。
public class GBDTRankModel extends BaseRankModel {

    /**
     * 训练模型得到pipelineModel
     *
     * @param training 训练数据DataFrame
     *                 包括2列：label: double、features: vector
     *                 注意：数据是已经处理好的，执行train()时，不会对label和特征进行变换
     */
    @Override
    public void train(DataFrame training) {

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        // todo 这个仍有问题，目前（可能）会把0变成1，1变成0，原因见StringIndexer文档
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel");

        //创建GBTRegressor实体（一个Estimators，将一个DataFrame转换为一个Model实体）
        GBTClassifier gbtClassifier = new GBTClassifier()
                .setMaxIter(10)     //最大迭代次数
                .setFeaturesCol("features")   //指明Feature在哪一列
                .setLabelCol("indexedLabel");   //指明label在哪一列

        //获取PipelineStage[]
        PipelineStage[] pipelineStages = new PipelineStage[]{labelIndexer, gbtClassifier};
        //装配PipelineStage，生成需要的Pipeline
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);

        //调用fit()获得pipelineModel
        pipelineModel = pipeline.fit(training);
    }

}

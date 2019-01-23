//package com.cs.mlapi.rankmodel;
//
//
//import org.apache.spark.ml.Pipeline;
//import org.apache.spark.ml.PipelineStage;
//import org.apache.spark.sql.DataFrame;
//
//
//public class GBDTLRRankModel extends BaseRankModel {
//
//    /**
//     * 训练模型得到pipelineModel
//     *
//     * @param training 训练数据DataFrame
//     *                 包括2列：label: double、features: vector
//     *                 注意：数据是已经处理好的，执行train()时，不会对label和特征进行变换
//     */
//    @Override
//    public void train(DataFrame training) {
//
//        GBTLRClassifier gbtlrClassifier = new GBTLRClassifier()
//                .setGBTMaxIter(10)  //最大迭代次数
//                .setLRMaxIter(100)  //最大迭代次数
//                .setRegParam(0.01)  //正则化
//                .setElasticNetParam(0.5)
//                .setFeaturesCol("features")   //Feature在哪一列
//                .setLabelCol("label");   //label在哪一列
//
//
//        //获取PipelineStage[]
//        PipelineStage[] pipelineStages = new PipelineStage[]{gbtlrClassifier};
//        //装配PipelineStage，生成需要的Pipeline
//        Pipeline pipeline = new Pipeline().setStages(pipelineStages);
//
//        //调用fit()获得pipelineModel
//        pipelineModel = pipeline.fit(training);
//    }
//
//}

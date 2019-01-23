package com.cs.mlapi.rankmodel;

import lombok.Getter;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.DataFrame;


//排序模型的基类
@Getter
public abstract class BaseRankModel {

    //PipelineModel是一个Transformer（将一个DataFrame转换为另一个DataFrame）
    //是训练好的模型本身
    //protected子类可访问，private仅自身可访问
    protected PipelineModel pipelineModel;

    //训练模型
    public abstract void train(DataFrame training);

    /**
     * 调用pipelineModel.transform()，执行预测（label），返回包含预测结果的DataFrame
     *
     * @param samples 需要预测的DataFrame
     * @return DataFrame新增预测结果列
     */
    public DataFrame transform(DataFrame samples) {
        return pipelineModel.transform(samples);
    }

}

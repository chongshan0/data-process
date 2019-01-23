package com.cs.mllibapi.rankmodel;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.HashMap;


public interface RankModel extends Serializable {

    //训练
    void train(JavaRDD<LabeledPoint> training);

    //预测label
    JavaRDD transform(JavaRDD<LabeledPoint> samples);

    //返回参数字符串
    String getParams();

    //设置参数
    void setParams(HashMap<String, Object> paramSet);
}

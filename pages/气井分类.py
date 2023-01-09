import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import os

from sklearn.metrics import silhouette_samples

data_path = 'dataset/GasProduction.csv'


def predict_by_wellNo(well, df, scaler, classifier):
    # 根据气井号对气井进行分类
    # 输入：要预测的气井号（列表）wells，已知数据集df，最小最大归一化scaler，拟合后的KMedoids聚类模型classifier
    # 输出：气井号所属类别列表
    df_well = df[df['WellNo'] == well]
    length = 30
    if df_well.shape[0] < 30:
        length = df_well.shape[0]
    production_hours_recently_mean = df_well.iloc[-length:, 3].mean()
    production_hours_recently_25per = df_well.iloc[-length:, 3].quantile(0.25)
    production_hours_recently_50per = df_well.iloc[-length:, 3].quantile(0.50)
    production_hours_recently_75per = df_well.iloc[-length:, 3].quantile(0.75)

    production_recently_mean = df_well.iloc[-length:, 4].mean()
    production_recently_25per = df_well.iloc[-length:, 4].quantile(0.25)
    production_recently_50per = df_well.iloc[-length:, 4].quantile(0.50)
    production_recently_75per = df_well.iloc[-length:, 4].quantile(0.75)
    elapsed_production = df_well['ElapsedProduction'].max()

    stat_well = pd.DataFrame([[well, production_hours_recently_mean, production_hours_recently_25per,
                               production_hours_recently_50per, production_hours_recently_75per,
                               production_recently_mean, production_recently_25per, production_recently_50per,
                               production_recently_75per, elapsed_production]],
                             columns=['WellNo', '近期平均开井时间', 'ProductionHoursRecently25Per',
                                      'ProductionHoursRecently50Per', 'ProductionHoursRecently75Per',
                                      '近期平均产量', 'ProductionRecently25Per',
                                      'ProductionRecently50Per', 'ProductionRecently75Per',
                                      '累计生产天数'])

    stat_scaled = scaler.transform(stat_well.drop('WellNo', axis=1))  # 对数据进行归一化
    label = classifier.predict(stat_scaled)  # 根据聚类模型进行预测

    return label


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 读取已知数据集，并进行格式转化
    df = pd.read_csv(data_path, usecols=['WellNo', 'Date', 'ElapsedProduction', 'Daily_h', 'Daily_104m3'])
    df['WellNo'] = df['WellNo'].astype("category")
    df["Date"] = pd.to_datetime(df["Date"])
    # ------------------------------------------------------------------
    # 挑选出所有投产时长大于60天的气井作为训练数据，计算每口气井的聚类特征，包括累计产量、平均产量、峰值产量、平均生产时长
    stats = pd.DataFrame()
    wells = np.array((df['WellNo'].unique()))
    for well in wells:
        df_well = df[df['WellNo'] == well]
        length = 30
        if df_well.shape[0] < 30:
            length = df_well.shape[0]
        production_hours_recently_mean = df_well.iloc[-30:, 3].mean()
        production_hours_recently_25per = df_well.iloc[-30:, 3].quantile(0.25)
        production_hours_recently_50per = df_well.iloc[-30:, 3].quantile(0.50)
        production_hours_recently_75per = df_well.iloc[-30:, 3].quantile(0.75)

        production_recently_mean = df_well.iloc[-30:, 4].mean()
        production_recently_25per = df_well.iloc[-30:, 4].quantile(0.25)
        production_recently_50per = df_well.iloc[-30:, 4].quantile(0.50)
        production_recently_75per = df_well.iloc[-30:, 4].quantile(0.75)
        elapsed_production = df_well['ElapsedProduction'].max()
        stat_well = pd.DataFrame([[well, production_hours_recently_mean, production_hours_recently_25per,
                                   production_hours_recently_50per, production_hours_recently_75per,
                                   production_recently_mean, production_recently_25per, production_recently_50per,
                                   production_recently_75per, elapsed_production]],
                                 columns=['WellNo', '近期平均开井时间', 'ProductionHoursRecently25Per',
                                          'ProductionHoursRecently50Per', 'ProductionHoursRecently75Per',
                                          '近期平均产量', 'ProductionRecently25Per',
                                          'ProductionRecently50Per', 'ProductionRecently75Per',
                                          '累计生产天数'])

        stats = pd.concat([stats, stat_well], axis=0)
    stats['WellNo'] = stats['WellNo'].astype('category')
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()  # 对聚类特征进行最小最大归一化
    stats_scaled = scaler.fit_transform(stats.drop('WellNo', axis=1))
    classifier = KMedoids(n_clusters=3, random_state=0).fit(stats_scaled)  # 构建KMedoids聚类模型

    wells_all = np.array(stats['WellNo'].unique())

    wells_class1 = wells_all[classifier.labels_ == 0]  # 第一类气井的气井号列表
    wells_class2 = wells_all[classifier.labels_ == 1]  # 第二类气井的气井号列表
    wells_class3 = wells_all[classifier.labels_ == 2]  # 第三类气井的气井号列表
    np.save('dataset/WellsClass1.npy', wells_class1)
    np.save('dataset/WellsClass2.npy', wells_class2)
    np.save('dataset/WellsClass3.npy', wells_class3)

    st.subheader("根据气井号输出类别")
    with st.form("my_form"):
        well_input = st.selectbox(label='请选择气井号', options=wells_all)
        submitted = st.form_submit_button("submit")
        if submitted:
            label = predict_by_wellNo(well_input, df, scaler, classifier)
            st.write('*'+well_input+'*' + "   ---->   *Class{}*".format(label[0] + 1))

    col1, col2, col3 = st.columns(3)
    col1.metric("Class1", str(len(wells_class1)))
    col2.metric("Class2", str(len(wells_class2)))
    col3.metric("Class3", str(len(wells_class3)))

    stats['class'] = 0
    for i in range(stats.shape[0]):
        if classifier.labels_[i] == 0:
            stats.iloc[i, -1] = 'class1'
        if classifier.labels_[i] == 1:
            stats.iloc[i, -1] = 'class2'
        if classifier.labels_[i] == 2:
            stats.iloc[i, -1] = 'class3'

    fig = px.scatter_3d(
        stats,
        x="近期平均开井时间",
        y="累计生产天数",
        z="近期平均产量",
        color='class'
    )
    fig.update_layout(width=900, height=900)
    st.plotly_chart(fig, width=900, height=900)

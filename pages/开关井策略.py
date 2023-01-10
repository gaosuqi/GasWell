# -*- coding: UTF-8 -*-
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random


def deal_C1(strategy):
    # 读取C1站气井号
    wells_C1 = np.load('WellsC1.npy', allow_pickle=True)
    model_path = 'liquidC1.bin'
    # lightGBM模型参数
    # 调参技巧 https://blog.csdn.net/weixin_44696221/article/details/104044951
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'huber',
        'metric': 'mae',
        'min_data_in_leaf': 8,  # 叶子结点的最小数据数，避免过拟合
        'max_depth': 8,  # 单颗树的最大深度
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'learning_rate': 0.05,
        'max_bin': 1000,
        'n_estimators': 6000,
        'boost_from_average': False,
        'verbose': -1,
    }
    # 读取数据
    dict_data = dict()
    for i in range(wells_C1.size):
        if wells_C1[i] in strategy.keys():
            dict_data[wells_C1[i] + '_h'] = [strategy[wells_C1[i]]]
        else:
            dict_data[wells_C1[i] + '_h'] = [0]
    df_C1 = pd.DataFrame(dict_data, columns=dict_data.keys())
    # df_C3 = pd.read_csv('./GasLiquidProductionC3Ori.csv', usecols=columns_C3 + ['LiquidProduction'])
    regressor = pickle.loads(open(model_path, 'rb').read())
    output = regressor.predict(df_C1)[0]
    return output


def deal_C2(strategy):
    # 读取C2站气井号
    wells_C2 = np.load('WellsC2.npy', allow_pickle=True)
    model_path = 'liquidC2.bin'
    # lightGBM模型参数
    # 调参技巧 https://blog.csdn.net/weixin_44696221/article/details/104044951
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'huber',
        'metric': 'mae',
        'min_data_in_leaf': 8,  # 叶子结点的最小数据数，避免过拟合
        'max_depth': 8,  # 单颗树的最大深度
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'learning_rate': 0.05,
        'max_bin': 1000,
        'n_estimators': 6000,
        'boost_from_average': False,
        'verbose': -1,
    }
    # 读取数据
    dict_data = dict()
    for i in range(wells_C2.size):
        if wells_C2[i] in strategy.keys():
            dict_data[wells_C2[i] + '_h'] = [strategy[wells_C2[i]]]
        else:
            dict_data[wells_C2[i] + '_h'] = [0]
    df_C2 = pd.DataFrame(dict_data, columns=dict_data.keys())
    # df_C3 = pd.read_csv('./GasLiquidProductionC3Ori.csv', usecols=columns_C3 + ['LiquidProduction'])
    regressor = pickle.loads(open(model_path, 'rb').read())
    output = regressor.predict(df_C2)[0]
    return output


def deal_C3(strategy):
    # 读取C3站气井号
    wells_C3 = np.load('WellsC3.npy', allow_pickle=True)
    model_path = 'liquidC3.bin'
    # lightGBM模型参数
    # 调参技巧 https://blog.csdn.net/weixin_44696221/article/details/104044951
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'huber',
        'metric': 'mae',
        'min_data_in_leaf': 8,  # 叶子结点的最小数据数，避免过拟合
        'max_depth': 8,  # 单颗树的最大深度
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'learning_rate': 0.05,
        'max_bin': 1000,
        'n_estimators': 6000,
        'boost_from_average': False,
        'verbose': -1,
    }
    # 读取数据
    dict_data = dict()
    for i in range(wells_C3.size):
        if wells_C3[i] in strategy.keys():
            dict_data[wells_C3[i] + '_h'] = [strategy[wells_C3[i]]]
        else:
            dict_data[wells_C3[i] + '_h'] = [0]
    df_C3 = pd.DataFrame(dict_data, columns=dict_data.keys())
    # df_C3 = pd.read_csv('./GasLiquidProductionC3Ori.csv', usecols=columns_C3 + ['LiquidProduction'])
    regressor = pickle.loads(open(model_path, 'rb').read())
    output = regressor.predict(df_C3)[0]
    return output


def evaluate_liquid_product(strategies, order):  # strategy是列表里面装列表 order是列表 表示井的顺序
    if len(strategies) == 0:
        return 10000
    min_liquid = 10000
    final_strategy = strategies[0]
    for strategy in strategies:
        lst = [0] * 907
        for i in strategy:
            lst[i] = 1
        data_dict = dict()
        for i in range(len(order)):
            data_dict[order[i]] = lst[i]
        liquid = deal_C3(data_dict) + deal_C2(data_dict) + deal_C1(data_dict)
        if liquid < min_liquid:
            min_liquid = liquid
            final_strategy = strategy

    return final_strategy, min_liquid


def generate_alternative_strategy(initial_choose, res_choose, m):
    # 输入是两个set
    # 输出是一个set
    change_quantity = random.choice(list(range(m)))
    if change_quantity > len(initial_choose):
        change_quantity = len(initial_choose)
    initial_choose_change = set(random.sample(initial_choose, k=change_quantity))
    res_choose_change = set(random.sample(res_choose, k=change_quantity))
    initial_choose = (initial_choose - initial_choose_change).union(res_choose_change)
    return initial_choose


def evaluate_current_pro(production, strategy):
    return sum([production[i] for i in strategy])


# 先生成一个可行的方案
def generate_feasible_strategy(amount, production, initial_num):
    '''
        amount:目标总产量
        production:某一天907口井的产量
    '''
    length = len(production)
    all_choose = set(list(range(length)))  # 907
    if initial_num > length:
        initial_num = length
    initial_choose = set(random.sample(list(range(length)), k=initial_num))  # 450
    res_choose = all_choose - initial_choose  # 457
    current_production = evaluate_current_pro(production, initial_choose)
    last_production = current_production
    last_choose = initial_choose
    while True:
        if current_production == amount:
            return initial_choose
        elif current_production < amount < last_production:
            return last_choose
        elif current_production > amount > last_production:
            return initial_choose
        elif current_production < amount and last_production < amount:
            last_production = current_production
            last_choose = initial_choose
            choose_one = set(random.sample(res_choose, k=1))  # 在剩余的关着的井中选择一口井打开
            choose_pro = production[list(choose_one)[0]]
            current_production += choose_pro
            initial_choose = initial_choose.union(choose_one)
        elif current_production > amount and last_production > amount:
            last_production = current_production
            last_choose = initial_choose
            choose_one = set(random.sample(initial_choose, k=1))
            choose_pro = production[list(choose_one)[0]]
            current_production = current_production - choose_pro
            initial_choose = initial_choose - choose_one
        else:
            print("Exception happened...................!")


# 在可行的方案基础上再进行调整，选择产液量最少的方案
def generate_strategy(amount, production, order, initial_num, last_choose, m=200, iteration=1, exist=False):
    '''
        amount:目标总产量
        production:某一天907口井的产量
    '''
    final_all_strategy = []
    length = len(production)
    all_choose = set(list(range(length)))  # 907
    if not exist:
        initial_choose = generate_feasible_strategy(amount, production, initial_num)
    else:
        initial_choose = set(last_choose)
    res_choose = all_choose - initial_choose
    for _ in range(iteration):
        strategy = generate_alternative_strategy(initial_choose, res_choose, m)
        current_pro = evaluate_current_pro(production, strategy)
        if current_pro < amount:
            continue
        final_all_strategy.append(list(strategy))

    if len(final_all_strategy) == 0:
        final_all_strategy.append(initial_choose)
    smallest_liquid_choose, liquid = evaluate_liquid_product(final_all_strategy, order)

    return smallest_liquid_choose, liquid


def main(amount):
    '''
    amount:目标总产气量
    '''
    df = pd.read_csv("test_result.csv")
    # 所有井的序号---列表  907
    order = [x for x in list(df[:907]["identifier"])]
    # 所有井第一天的产气量-2022.08.09  907
    one_day = [x for x in list(df[:907]["forecast"])]
    percent = float(amount) / sum(one_day)
    initial_num = int(percent * 907)
    # 所有井第二天的产气量-2022.08.10  907
    two_day = [x for x in list(df[907:907 * 2]["forecast"])]
    # 所有井第三天的产气量-2022.08.11  907
    three_day = [x for x in list(df[907 * 2:907 * 3]["forecast"])]
    # 所有井第四天的产气量-2022.08.12  907
    four_day = [x for x in list(df[907 * 3:907 * 4]["forecast"])]
    # 所有井第五天的产气量-2022.08.13  907
    five_day = [x for x in list(df[907 * 4:907 * 5]["forecast"])]
    # 所有井第六天的产气量-2022.08.14  907
    six_day = [x for x in list(df[907 * 5:907 * 6]["forecast"])]
    # 所有井第七天的产气量-2022.08.15  907
    seven_day = [x for x in list(df[907 * 6:]["forecast"])]
    one_strategy, one_liquid = generate_strategy(amount, one_day, order, initial_num, [])
    two_strategy, two_liquid = generate_strategy(amount, two_day, order, initial_num, one_strategy)
    three_strategy, three_liquid = generate_strategy(amount, three_day, order, initial_num, two_strategy)
    four_strategy, four_liquid = generate_strategy(amount, four_day, order, initial_num, three_strategy)
    five_strategy, five_liquid = generate_strategy(amount, five_day, order, initial_num, four_strategy)
    six_strategy, six_liquid = generate_strategy(amount, six_day, order, initial_num, five_strategy)
    seven_strategy, seven_liquid = generate_strategy(amount, seven_day, order, initial_num, six_strategy)
    return order, one_strategy, two_strategy, three_strategy, four_strategy, five_strategy, six_strategy, seven_strategy


def transform(x, length):
    day = [0] * length
    for i in x:
        day[i] = 1
    return day


def int2str(x):
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = "关"
        else:
            x[i] = "开"
    return x


def show():
    st.title("开关井策略")
    with st.form("my_form"):
        amount = st.text_input(
            "请输入目标产量"
        )
        submitted = st.form_submit_button("生成策略")
        if submitted:
            amount = int(amount)
            if amount <= 0 or amount > 1200:
                st.write("请输入合法数据（大于0小于1200）")
            else:

                order, one_strategy, two_strategy, three_strategy, four_strategy, five_strategy, six_strategy, seven_strategy = \
                    main(amount)
                length = len(order)
                first_day = int2str(transform(one_strategy, length))
                second_day = int2str(transform(two_strategy, length))
                third_day = int2str(transform(three_strategy, length))
                fourth_day = int2str(transform(four_strategy, length))
                fifth_day = int2str(transform(five_strategy, length))
                sixth_day = int2str(transform(six_strategy, length))
                seventh_day = int2str(transform(seven_strategy, length))
                first_day_df = pd.DataFrame({
                    "Date": ["2022-08-09"] * length, "Well Name": order, "Status": first_day
                })
                second_day_df = pd.DataFrame({
                    "Date": ["2022-08-10"] * length, "Well Name": order, "Status": second_day
                })
                third_day_df = pd.DataFrame({
                    "Date": ["2022-08-11"] * length, "Well Name": order, "Status": third_day
                })
                fourth_day_df = pd.DataFrame({
                    "Date": ["2022-08-12"] * length, "Well Name": order, "Status": fourth_day
                })
                fifth_day_df = pd.DataFrame({
                    "Date": ["2022-08-13"] * length, "Well Name": order, "Status": fifth_day
                })
                sixth_day_df = pd.DataFrame({
                    "Date": ["2022-08-14"] * length, "Well Name": order, "Status": sixth_day
                })
                seventh_day_df = pd.DataFrame({
                    "Date": ["2022-08-15"] * length, "Well Name": order, "Status": seventh_day
                })
                df = pd.concat(
                    [first_day_df, second_day_df, third_day_df, fourth_day_df, fifth_day_df, sixth_day_df,
                     seventh_day_df])
                st.table(df)


if __name__ == '__main__':
    show()


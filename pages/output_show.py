# -*- coding: UTF-8 -*-
import pandas as pd
import streamlit as st


def transform(x, length):
    day = [0] * length
    for i in x:
        day[i] = 1
    return day


def int2str(x):
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = "close"
        else:
            x[i] = "open"
    return x


def main(filename):
    with open(filename, "r") as f:
        amount = int(f.readline())
        order = f.readline().split(" ")
        length = len(order)
        first_day = [int(i) for i in f.readline().split(" ")[:-2]]
        first_day_lst = int2str(transform(first_day, length))
        first_day_df = pd.DataFrame({
            "Date": ["2022-08-09"] * length, "Well Name": order, "Status": first_day_lst
        })

        second_day = [int(i) for i in f.readline().split(" ")[:-2]]
        second_day_lst = int2str(transform(second_day, length))
        second_day_df = pd.DataFrame({
            "Date": ["2022-08-10"] * length, "Well Name": order, "Status": second_day_lst
        })

        third_day = [int(i) for i in f.readline().split(" ")[:-2]]
        third_day_lst = int2str(transform(third_day, length))
        third_day_df = pd.DataFrame({
            "Date": ["2022-08-11"] * length, "Well Name": order, "Status": third_day_lst
        })

        fourth_day = [int(i) for i in f.readline().split(" ")[:-2]]
        fourth_day_lst = int2str(transform(fourth_day, length))
        fourth_day_df = pd.DataFrame({
            "Date": ["2022-08-12"] * length, "Well Name": order, "Status": fourth_day_lst
        })

        fifth_day = [int(i) for i in f.readline().split(" ")[:-2]]
        fifth_day_lst = int2str(transform(fifth_day, length))
        fifth_day_df = pd.DataFrame({
            "Date": ["2022-08-13"] * length, "Well Name": order, "Status": fifth_day_lst
        })

        sixth_day = [int(i) for i in f.readline().split(" ")[:-2]]
        sixth_day_lst = int2str(transform(sixth_day, length))
        sixth_day_df = pd.DataFrame({
            "Date": ["2022-08-14"] * length, "Well Name": order, "Status": sixth_day_lst
        })

        seventh_day = [int(i) for i in f.readline().split(" ")[:-2]]
        seventh_day_lst = int2str(transform(seventh_day, length))
        seventh_day_df = pd.DataFrame({
            "Date": ["2022-08-15"] * length, "Well Name": order, "Status": seventh_day_lst
        })
        df = pd.concat([first_day_df,second_day_df,third_day_df,fourth_day_df,fifth_day_df, sixth_day_df, seventh_day_df])

    st.title("开关井策略")
    with st.form("my_form"):
        amount = st.text_input(
            "请输入目标产量"
        )
        submitted = st.form_submit_button("生成策略")
        if submitted:
            st.table(df)


if __name__ == '__main__':
    main("output.txt")

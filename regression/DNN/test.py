# coding:utf-8
# Created by chen on 23/08/2018
# email: q.chen@student.utwente.nl
import pandas as pd
df = pd.read_csv("C:/work/ecare/data/allCom_v3_final.csv")
df = df.rename({'duration_next_week':'label'}, axis='columns')
print(df.shape)
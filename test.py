# -*- coding: utf-8 -*-
# @Time : 2023/6/9 10:51
# @Author : Wangbb
# @FileName: test.pyimport numpy as np
# from ny_model import nengyuan_model
# import tempfile
import numpy as np
import pandas as pd
from ny_model import  provice_model

df_zhej = pd.read_excel("D:\Desktop\副本能源数据-更新至23年第一季度.xlsx",sheet_name='浙江省')
a = np.arange(-1,1,1)
provice_model(df_zhej, 0.8, a , None)

from ny_model import nengyuan_model
import tempfile
print(np.arange(-1,1,1))
a = ['全地市', '浙江省']
if ['全地市', '浙江省'] in ['杭州市', '宁波市', '温州市', '绍兴市', '湖州市', '嘉兴市',
                                    '金华市', '衢州市', '台州市', '丽水市', '舟山市',
                                    '全地市', '浙江省']:
    print(1)
# df_result = nengyuan_model('杭州市', pd.read_excel("D:\Desktop\副本能源数据-更新至23年第一季度.xlsx",sheet_name='杭州市'), 0.6, np.arange(-1,1,1), '交互项')

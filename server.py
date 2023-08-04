# -*- coding: utf-8 -*-
# @Time : 2023/6/8 14:43
# @Author : Wangbb
# @FileName: 学习1.py
import pandas as pd
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio import start_server
import pywebio
import numpy as np
from ny_model import nengyuan_model, provice_model
import tempfile
import io
import json

def input_check(data):
    # input的合法性校验
    if '全地市' in data['diqu_input'] and '浙江省' in data['diqu_input']:
        return "浙江和地市二选一"
    if data['max_value'] < data['min_value']:
        return "次方项最大范围必须大于次方项最小范围"
    if data['gap'] > (data['max_value'] - data['min_value']):
        return "间隔太小"
    if data['r'] < 0 or data['r'] > 1:
        return "阈值r的范围在0-1之间"

def energy_consum():
    info = input_group("建模参数", [
        input('输入任务uid',typr=TEXT, name='uid'),
        checkbox("选择建模地市", options=['杭州市', '宁波市', '温州市', '绍兴市', '湖州市', '嘉兴市',
                                    '金华市', '衢州市', '台州市', '丽水市', '舟山市',
                                    '全地市', '浙江省'], multiple=True, name='diqu_input'),
        input("输入次方项最小值", type=FLOAT, name='min_value', value=-2),
        input("输入次方项最大值", type=FLOAT, name='max_value', value=2),
        input("输入次方项间", type=FLOAT, name='gap',value=1),
        input("r2阈值", type=FLOAT, name='r', help_text='0-1之间，阈值越高输出的模型越少,输出越快',value=0.8),
        checkbox("选择模型参数", options=['归一化', '交互项', '异常值替换'], multiple=True, name='params_input',help_text='如都选，数据处理顺序先异常值-交互项-归一化'),
        file_upload("上传数据文件", name='file')
    ], validate=input_check
                       )
    # 上传文件
    file_path = info['file']['content']
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_path)
        file_path = temp_file.name

    # 输出字节流处理
    output1 = io.BytesIO()
    output2 = io.BytesIO()
    excel_file = pd.ExcelWriter(output1, engine='xlsxwriter')
    jiaohu_file = pd.ExcelWriter(output2, engine='xlsxwriter')


    # 次方项间隔获取
    jiange = np.arange(info['min_value'], info['max_value'] + 0.1, info['gap'])

    # 进度条
    put_progressbar('bar')
    i = 0
    if '全地市' in info['diqu_input']:
        for diqu in ['杭州市', '宁波市', '温州市', '绍兴市', '湖州市', '嘉兴市',
                     '金华市', '衢州市', '台州市', '丽水市', '舟山市']:
            set_progressbar('bar', i / 11)
            df = pd.read_excel(file_path, sheet_name=f'{diqu}')
            df_result = nengyuan_model(diqu, df, info['r'], jiange, info['params_input'])
            df_result[0].to_excel(excel_file, sheet_name=f'{diqu}')
            df_result[1].to_excel(jiaohu_file, sheet_name=f'{diqu}')
            i += 1

    elif '浙江省' in info['diqu_input']:
        df = pd.read_excel(file_path, sheet_name='浙江省')
        df_result = provice_model(df, info['r'], jiange, info['params_input'])
        df_result[0].to_excel(excel_file)
        df_result[1].to_excel(jiaohu_file)


    else:
        for diqu in info['diqu_input']:
            set_progressbar('bar', i / len(info['diqu_input']))
            df = pd.read_excel(file_path, sheet_name=f'{diqu}')
            df_result = nengyuan_model(diqu, df, info['r'], jiange, info['params_input'])
            df_result[0].to_excel(excel_file, sheet_name=f'{diqu}')
            df_result[1].to_excel(jiaohu_file, sheet_name=f'{diqu}')
            # put_text(f'建模进度{i / len(1)}')
            i += 1
    set_progressbar('bar', 1)
    # 保存文件
    excel_file.close()
    jiaohu_file.close()

    # 转换为字节流
    bytes_excel = output1.getvalue()
    bytesjiaohu = output2.getvalue()

    # 关闭
    output1.close()
    output2.close()

    # 输出

    put_row([put_file(f"{info['uid']}拟合结果.xlsx", bytes_excel, f"{info['uid']}拟合结果"), None,
             put_file(f"{info['uid']}解释数据.xlsx", bytesjiaohu, f"{info['uid']}解释数据")])



if __name__ == '__main__':
    start_server(
        applications=energy_consum,
        # debug=True,
        # port=1216,
        # auto_open_webbrowser=True,
        remote_access=True,
        # reconnect_timeout=3,
    )

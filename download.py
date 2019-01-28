import tushare as ts
import numpy as np
import xlrd
import os


####常量设置####
fileName = 'E:/Python/tushare/data/everyday/'

####链接地址的token,'a82de...............................'这部分是密钥####
pro = ts.pro_api('a82de...............................')

####读取全部的股票代码，'/5%.xlsx'表示需要下载的股票数据列表####
def download(time,start_time,market):
    readbook = xlrd.open_workbook(fileName+ time + '/5%.xlsx')
    rtable = readbook.sheet_by_name(market)
    data = []
    row = rtable.nrows
    col = rtable.ncols
    for i in range(row-1):
        data.append([])
        for j in range(col):
            data[i].append('s')
    for i in range(row-1):
        for j in range(col):
            data[i][j] = rtable.cell(i+1,j).value
    print(len(data))

####下载个股信息####
    if os.path.exists(fileName + time +'/'+ market ) == False:
        os.makedirs(fileName + time +'/'+ market )
    for i in range(len(data)):
        num_code = data[i][0]+".SZ"
        name = data[i][1]
        if True:
            df = pro.daily_basic(ts_code=num_code,start_date='20190120')
            df.to_excel(fileName + time + '/' + market + '/' + num_code + '_' + name + '_daily_basic.xlsx')
            df = pro.daily(ts_code=num_code,start_date='20190120')
            df.to_excel(fileName + time + '/' + market + '/' + num_code + '_' + name + '_daily.xlsx')
            print('doenload Number',i,num_code,name)

if __name__ == "__main__":
    download("2019012","20190120","zhong")

import tensorflow as tf
import tushare as ts
import numpy as np
import xlrd
import xlwt
import os
import time


fileName = 'E:/Python/tushare/data/everyday/'

def merge(time,market):
    readbook = xlrd.open_workbook(fileName + time + '/5%.xlsx')
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

    for n in range(len(data)):
        num_code = data[n][0]+".SZ"
        name = data[n][1]
        if True:
            readbook_d = xlrd.open_workbook(fileName + time + '/' + market + '/' + num_code + '_' + name + '_daily.xlsx')
            rtable_d = readbook_d.sheet_by_name('Sheet1')
            row_d = rtable_d.nrows
            col_d = rtable_d.ncols
            data_d = []
            for i in range(row_d):
                data_d.append([])
                for j in range(col_d):
                    data_d[i].append('s')
            for i in range(row_d):
                for j in range(col_d):
                    data_d[i][j] = rtable_d.cell(i,j).value

            readbook_db = xlrd.open_workbook(fileName + time + '/' + market + '/' + num_code + '_' + name + '_daily_basic.xlsx')
            rtable_db = readbook_db.sheet_by_name('Sheet1')
            row_db = rtable_db.nrows
            col_db = rtable_db.ncols
            data_db = []
            for i in range(row_db):
                data_db.append([])
                for j in range(col_db):
                    data_db[i].append('s')
            for i in range(row_db):
                for j in range(col_db):
                    data_db[i][j] = rtable_db.cell(i,j).value
            data_db = np.delete(data_db,[0,1,2,3],1)
            for i in range(min(row_db,row_d)):
                data_d[i].extend(data_db[i])

            for i in range(min(row_db,row_d)-2):
                data_d[i+2].append(data_d[i+1][9])
            if len(data_d)>1:
                data_d[1].append(0.00)
            data_d[0].append('Y')

            writebook = xlwt.Workbook(encoding = 'ascii')
            wtable = writebook.add_sheet('Sheet1')
            for i in range(len(data_d)):
                for j in range(len(data_d[0])):
                    if data_d[i][j] == '':
                        data_d[i][j] = 0.00
                    wtable.write(i,j,data_d[i][j])
            writebook.save(fileName + time + '/' + market + '/' + num_code + '_' + name + '_data_merge.xlsx')
            print('merge Number',n,num_code,name)

if __name__ == "__mian__":
    merge("2019012","zhong")


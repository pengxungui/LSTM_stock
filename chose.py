import tensorflow as tf
import numpy as np
import xlrd
import xlwt
import os
import time

fileName = 'E:/Python/tushare/data/everyday/'

def chose(time,market):
    readbook = xlrd.open_workbook(fileName + time + '/5%.xlsx')
    rtable = readbook.sheet_by_name(market)
    writebook = xlwt.Workbook(encoding = 'ascii')
    wtable = writebook.add_sheet('Sheet1')
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
    print('1',len(data),len(data[0]))


####选取一天涨停的数据####
    temp = []
    for i in range(1):
        temp.append([])
        for j in range(26):
            temp[i].append('A')
    
    for n in range(len(data)):
        num_code = data[n][0] + ".SZ"
        name = data[n][1]
        if True:
            readbook_d = xlrd.open_workbook(fileName + time + '/' + market + '/' + num_code + '_' + name + '_data_merge.xlsx')
            rtable_d = readbook_d.sheet_by_name('Sheet1')
            row_d = rtable_d.nrows
            col_d = rtable_d.ncols
            print(row_d,col_d)
            data_d = []
            for i in range(row_d):
                data_d.append([])
                for j in range(col_d):
                    data_d[i].append('s')
            for i in range(row_d):
                for j in range(col_d):
                    data_d[i][j] = rtable_d.cell(i,j).value
            for i in range(row_d,2,-1):
                if data_d[i-2][9]>9.4:
                    temp.append(data_d[i-1])
                    temp.append(data_d[i-2])
            print(len(temp))
            print('Number',n,num_code,name)

####xlrd xlwt 最大行65535####
    temp_num = len(temp)
    if temp_num > 65535:
        temp_num = 65535
    for i in range(temp_num):
        for j in range(len(temp[0])):
            if temp[i][j] == '':
                temp[i][j] = '0.00'
            wtable.write(i,j,temp[i][j])
    writebook.save(fileName + time + '/' + market + '/' + '1up_data.xlsx')
    print('1up Successy!')

####选取两天涨停的数据####
    writebook = xlwt.Workbook(encoding = 'ascii')
    wtable = writebook.add_sheet('Sheet1')
    temp = []
    for i in range(1):
        temp.append([])
        for j in range(26):
            temp[i].append('A')

    for n in range(len(data)):
        num_code = data[n][0] + ".SZ"
        name = data[n][1]
        if True:
            readbook_d = xlrd.open_workbook(fileName + time + '/' + market + '/' + num_code + '_' + name + '_data_merge.xlsx')
            rtable_d = readbook_d.sheet_by_name('Sheet1')
            row_d = rtable_d.nrows
            col_d = rtable_d.ncols
            print(row_d,col_d)
            data_d = []
            for i in range(row_d):
                data_d.append([])
                for j in range(col_d):
                    data_d[i].append('s')
            for i in range(row_d):
                for j in range(col_d):
                    data_d[i][j] = rtable_d.cell(i,j).value
            for i in range(row_d,2,-1):
                if data_d[i-2][9]>9.4 and data_d[i-1][9]>9.4:
                    temp.append(data_d[i-1])
                    temp.append(data_d[i-2])
            print(len(temp))
            print('Number',n,num_code,name)

    temp_num = len(temp)
    if temp_num > 65535:
        temp_num = 65535
    for i in range(temp_num):
        for j in range(len(temp[0])):
            if temp[i][j] == '':
                temp[i][j] = '0.00'
            wtable.write(i,j,temp[i][j])
    writebook.save(fileName + time + '/' + market + '/' + '2up_data.xlsx')
    print('2up Successy!')

####选取2天涨幅大于5%的数据####
    writebook = xlwt.Workbook(encoding = 'ascii')
    wtable = writebook.add_sheet('Sheet1')
    temp = []
    for i in range(1):
        temp.append([])
        for j in range(26):
            temp[i].append('A')

    for n in range(len(data)):
        num_code = data[n][0] + ".SZ"
        name = data[n][1]
        if True:
            readbook_d = xlrd.open_workbook(fileName + time + '/' + market + '/' + num_code + '_' + name + '_data_merge.xlsx')
            rtable_d = readbook_d.sheet_by_name('Sheet1')
            row_d = rtable_d.nrows
            col_d = rtable_d.ncols
            print(row_d,col_d)
            data_d = []
            for i in range(row_d):
                data_d.append([])
                for j in range(col_d):
                    data_d[i].append('s')
            for i in range(row_d):
                for j in range(col_d):
                    data_d[i][j] = rtable_d.cell(i,j).value
            for i in range(row_d,2,-1):
                if data_d[i-2][9]>5 and data_d[i-1][9]>5:
                    temp.append(data_d[i-1])
                    temp.append(data_d[i-2])
            print(len(temp))
            print('Number',n,num_code,name)

    temp_num = len(temp)
    if temp_num > 65535:
        temp_num = 65535
    for i in range(temp_num):
        for j in range(len(temp[0])):
            if temp[i][j] == '':
                temp[i][j] = '0.00'
            wtable.write(i,j,temp[i][j])
    writebook.save(fileName + time + '/' + market + '/' + '5%2up_data.xlsx')
    print('5%2up Successy!')

if __name__ == "__main__":
    chose("2019012","zhong")

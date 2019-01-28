import tensorflow
import download
import merge
import chose
import LSTM

market = ["chuang","zhu","zhong"]
time = "20190125"
start_time = "20190122"

for i in range(len(market)):
   download.download(time,start_time,market[i])
    merge.merge(time,market[i])
    chose.chose(time,market[i])

for i in range(len(market)):
    LSTM.two_up(time,market[i])

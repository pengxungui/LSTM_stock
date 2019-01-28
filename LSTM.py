import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import xlrd
import xlwt
import os
import time
from sklearn.preprocessing import minmax_scale

def two_up(time,market):
####设置常量####
        input_size = 19
        time_steps = 2
        hiddens = 64
        layers = 1
        output_size = 3
        batch_size = 16
        learning_rate = 0.001
        #keep_prob = 0.5
        decay_rate = 0.8
        decay_step = 50
        alpha = 1.0
        scale = 1e-4
        train = False
        count = 0
        test_num = 20
        fileName = "E:/Python/tushare/data/everyday/"
        MODEL_NAME = "dichan_LSTM"
        if market == "zhong":
           MODEL_SAVE_PATH = 'E:/Python/tushare/model/中小板/2up'
        if market == "chuang":
           MODEL_SAVE_PATH = 'E:/Python/tushare/model/创业板/2up'
        if market == "zhu":
           MODEL_SAVE_PATH = 'E:/Python/tushare/model/主板/2up'

####从Excel中读出数据####
        readbook = xlrd.open_workbook(fileName + time + '/' + market + '/' + '2up_data.xlsx')
        table = readbook.sheet_by_name('Sheet1')
        writebook = xlwt.Workbook(encoding = 'ascii')
        wtable = writebook.add_sheet('Sheet1')
        row = table.nrows
        col = table.ncols

        stock_inf = []
        for i in range(row-1):
                stock_inf.append([])
                for j in range(2):
                        stock_inf[i].append('s')
        for i in range(row-1):
                for j in range(2):
                        stock_inf[i][j] =table.cell(i + 1,j + 1).value

        data_x = np.zeros((row-1,22))
        data_y = np.zeros((row-1,output_size))
        for i in range(row-1):
                for j in range(23):
                        if j != 22:
                                temp = table.cell(i + 1,j + 3).value
                                if type(temp) == str:
                                        data_x[i][j] = float(temp)
                                if type(temp) != str:
                                        data_x[i][j] = temp
                        else:
                                a = table.cell(i + 1,j + 3).value
                                if a >= 5:
                                        data_y[i] = [1,0,0]
                                elif a > 0 and a < 5:
                                        data_y[i] = [0,1,0]
                                elif a <= 0 and a > -5:
                                        data_y[i] = [0,0,1]
                                else:
                                        data_y[i] = [0,0,1]

####当数据不足被batch_size整除时，末尾添加####

####数据处理，删除data_x中不要列####
        data_x = np.delete(data_x,[17,18,19],1)
        print(data_x.shape)
        print(data_y.shape)

####对输入数据的每一列标注化处理####
        for i in range(input_size):
                data_x[:,i] = minmax_scale(data_x[:,i],copy = True,feature_range = (-1,1))

####把数据变成三维数据####
        data_x = data_x.reshape((-1,time_steps,input_size))
        data_y = data_y.reshape((-1,time_steps,output_size))
        data_x1 = data_x
        data_y1 = data_y
        
####把数据添加为整数个batch_size,用于预测数据####
        while data_x.shape[0] % batch_size != 0:
                data_x = np.row_stack((data_x,data_x[-1].reshape((-1,time_steps,input_size))))
                data_y = np.row_stack((data_y,data_y[-1].reshape((-1,time_steps,output_size))))

##随机打乱训练数据顺序##
        perm = np.random.permutation(data_x1.shape[0])
        data_x1 = data_x1[perm,:,:]
        data_y1 = data_y1[perm,:,:]
        ##train_x = train_x.reshape((-1,input_size))
        ##train_y = train_y.reshape((-1,output_size))

        test_x = data_x1[0:batch_size*test_num,:,:]
        test_y = data_y1[0:batch_size*test_num,:,:]
        train_x = data_x1[batch_size*test_num:data_x.shape[0],:,:]
        train_y = data_y1[batch_size*test_num:data_x.shape[0],:,:]

        print(train_x.shape)
        print(train_y.shape)

        print(test_x.shape)
        print(test_y.shape)

####把数据分成batch_size大小####
        def Next_batch_x(data,batch_size,ii):
                num_b = ii % (data.shape[0] // batch_size)
                return data[num_b * batch_size:(num_b + 1) * batch_size]
        def Next_batch_y(data,batch_size,ii):
                num_b = ii % (data.shape[0] // batch_size)
                return data[num_b * batch_size:(num_b + 1) * batch_size,time_steps - 1,:]

####初始化权重值####
        weights = tf.Variable(tf.truncated_normal([hiddens,output_size],stddev = 0.1,name = 'w',dtype = tf.float32))
        biases = tf.Variable(tf.truncated_normal([output_size],stddev = 0.001,name = 'b',dtype = tf.float32))

        def Weights(n_in,n_out,name = None):
            return tf.Variable(tf.truncated_normal([n_in,n_out],stddev = 0.00001,name = name,dtype = tf.float32))

        def Biases(n_out,name = None):
            return tf.Variable(tf.truncated_normal([n_out],stddev = 0.00001,name = name,dtype = tf.float32))

        def Cell(x,w,b):
            return tf.matmul(x,w) + b

####定义网络结构####
        x = tf.placeholder(tf.float32,[None,time_steps,input_size],name = 'input')
        y = tf.placeholder(tf.float32,[None,output_size],name = 'output')
        keep_prob = tf.placeholder(tf.float32)

        def LSTM1(x):
            def cell():
                cell = rnn.LSTMCell(hiddens,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
                return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells = []
            for i in range(layers):
                cells.append(cell())
            lstm_cell = rnn.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32,time_major=False)
            return outputs[:,-1,:]
    #return tf.nn.relu(tf.matmul(outputs[:,-1,:], weights) + biases)
    #return tf.nn.softmax(tf.matmul(tf.tanh(outputs[:,-1,:]), weights) + biases)
        def LSTM2(x):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hiddens)
            outputs,final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,              # 选择传入的cell
                inputs=x,               # 传入的数据
                initial_state=None,         # 初始状态
                dtype=tf.float32,           # 数据类型
                time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
            )
            return outputs[:, -1, :]

        ##pred = tf.nn.softsign(y4)

        L=LSTM1(x)
        pred = tf.layers.dense(inputs=L, units=output_size)

        ####选择损失函数和优化器####
        #pred = LSTM(x,weights,biases)
        global_step = tf.Variable(0,trainable = False)
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale),tf.trainable_variables())
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
        ##cost = tf.reduce_mean(tf.square(y - pred)) + reg
        ##learning_rate = tf.train.exponential_decay(learning_rate,global_step,decay_step,decay_rate,staircase = True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step = global_step)
        ##train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step = global_step)
        ##train_op = tf.train.MomentumOptimizer(learning_rate,momentum = 0.9).minimize(cost,global_step = global_step)
        ##train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost,global_step = global_step)
        ##train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost,global_step = global_step)
        ##accurary = tf.metrics.accuracy(labels = tf.argmax(y,axis = 1),predictions = tf.argmax(pred,axis = 1))[1]
        prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
        accurary = tf.reduce_mean(tf.cast(prediction,tf.float32))
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
                sess.run(init)
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess,ckpt.model_checkpoint_path)
                if train == True:
                        for i in range(200000):
                                count = i
                                ii = np.random.randint(0,80000)
                                batch_x = Next_batch_x(train_x,batch_size,ii)
                                batch_y = Next_batch_y(train_y,batch_size,ii)
                                sess.run(train_op,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
                                if i % 100 == 0:
                                        print(i,"%.2f"%sess.run(accurary,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0}),"%.4f"%sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1}))
                                        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
                                        pred_ = sess.run(pred,feed_dict={x:train_x[0:batch_size,:,:],keep_prob:1.0})
        ##                                print("outputs:",pred_)
                                if i % 500 == 0:
                                        count = 0
                                        for i in range(test_num):
                                                accurary_, pred_ = sess.run([accurary,pred],feed_dict={x:test_x[i*batch_size:(i+1)*batch_size,:,:],y:test_y[i*batch_size:(i+1)*batch_size,time_steps - 1,:],keep_prob:1.0})
        ##                                        print("outputs: %.2f"%accurary_)
                                                count += accurary_
                                        print("arg: %.2f" %(count/test_num))
                else:
                        temp = []
                        for i in range(data_x.shape[0] // batch_size):
                                accurary_, pred_ = sess.run([accurary,pred],feed_dict={x:data_x[i*batch_size:(i+1)*batch_size,:,:],y:data_y[i*batch_size:(i+1)*batch_size,time_steps - 1,:],keep_prob:1.0})
        ##                        print("accurary: %.2f"% accurary_)
                                temp.extend(sess.run(tf.argmax(pred_,1)))
                        print(temp)
                        for i in range(len(temp)):
                                wtable.write(i,1,int(temp[i]))
                        writebook.save(fileName + time + '/' + market + '/' + '2up_data_label_pred.xlsx')
                        print('Successy!')

                        stock_chose = []
                        for i in range(len(temp)):
                                if (2*i+1)<= len(stock_inf) and (temp[i] == 0 or temp[i] == 1) and stock_inf[2*i+1][1] == time:
                                        stock_chose.append(stock_inf[2*i+1][0])
                        print("****"+market+"***2up****")
                        print(stock_chose)
        tf.reset_default_graph()

if __name__ == "__main__":
        two_up("20190125","zhong")

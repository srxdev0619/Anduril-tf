import tensorflow as tf
import numpy as np
import random
import time


class Anduril:
    def __init__(self):
        """
        This method initializes the neural network and its parameters
        """
        self.weights = []
        self.biases = []
        self.best_weights = []
        self.best_biases = []
        self.input_data = []
        self.input_train = []
        self.input_test = []
        self.output_data = []
        self.output_train = []
        self.output_test = []
        self.num_data = 0
        self.num_train = 0
        self.num_test = 0
        self.mini_batch_size = 0
        self.epochs = 100
        self.classreg = 1
        self.costfunc = 1
        self.numhid = 0
        self.arch = []
        self.restored = False
        self.net_name = ""
        self.restored_net = None
        return

    def init(self, arch, input_file, classreg = 1, costfunc = 1, sep1 = ",", sep2 = " "):
        self.numhid = len(arch)
        self.num_layers = len(arch) + 2
        self.classreg = classreg
        self.costfunc = costfunc

        f = open(input_file,"r")

        temp_data = []
        for line in f:
            temp_data.append(line)

        random.shuffle(temp_data)
        self.num_data = len(temp_data)
        in_lent = 0
        out_lent = 0
        for n in range(self.num_data):
            l = temp_data[n].split(sep2)
            input_l = l[0]
            output_l = l[1]
            in_temp = self.__fin_parser(input_l,sep1)
            out_temp = self.__fin_parser(output_l,sep1)
            if (n == 0):
                in_lent = len(in_temp)
                out_lent = len(out_temp)
            else:
                if (len(in_temp) != in_lent) or (len(out_temp) != out_lent):
                    print "The length of input or output vector varies!"
                    self.__reset("init")
                    return 
            self.input_data.append(in_temp)
            self.output_data.append(out_temp)
        self.arch = arch
        self.arch.insert(0,in_lent)
        self.arch.append(out_lent)
        self.num_train = int(self.num_data*0.8)
        self.num_test = self.num_data - self.num_test
        self.input_train = self.input_data[:self.num_train]
        self.input_test = self.input_data[self.num_train:]
        self.output_train = self.output_data[:self.num_train]
        self.output_test = self.output_data[self.num_train:]
        self.input_train = np.array(self.input_train, dtype=np.float32)
        self.input_test = np.array(self.input_test, dtype=np.float32)
        self.output_train = np.array(self.output_train, dtype=np.float32)
        self.output_test = np.array(self.output_test, dtype=np.float32)


    def train(self,epochs,Opt = "Adam", learning_rate = 0.0, hist_file = None, log_file = None, net_name = None):
        self.epochs = epochs
        if hist_file:
            f = open(hist_file, "w")
            f.close()
        if log_file:
            f = open(log_file + "_test.dat", "w")
            f.close()
            f = open(log_file + "_train.dat", "w")
            f.close()
        with tf.Graph().as_default():
            seed = int(time.time())
            tf.set_random_seed(seed)
            #sess = tf.Session()
            #if self.restored:
            #    saver = tf.train.Saver()
            #    load_path = "." + self.net_name + "_ckpt"
            #    saver.restore(sess, load_path)
            #else:
            #    init = tf.initialize_all_variables()
            #    sess.run(init)
            input_placeholder = tf.placeholder(tf.float32, shape=(None,self.arch[0]))
            output_placeholder = tf.placeholder(tf.float32, shape=(None,self.arch[-1]))

            net_out = self.__feed_forward(input_placeholder)

            net_error = self.__errors(net_out,output_placeholder)

            net_train = self.__trainbatch(net_error, Opt, learning_rate)

            net_eval = self.__errors(net_out, output_placeholder)

            sess = tf.Session()
            if self.restored:
                saver = tf.train.Saver()
                load_path = "." + self.net_name + "_ckpt"
                saver.restore(sess, load_path)
            else:
                init = tf.initialize_all_variables()
                sess.run(init)

            min_test_rmse = float(100e10)
            for n in range(self.epochs):
                start = 0
                end = 10

                while end < self.num_train:
                    train_batch_input = self.input_train[start:end]
                    train_batch_output = self.output_train[start:end]
                    feed_dict = {input_placeholder: train_batch_input, output_placeholder: train_batch_output}
                    _,batch_train_rmse = sess.run([net_train,net_error], feed_dict=feed_dict)
                    start = start + 10
                    end = end + 10
                feed_test_dict = {input_placeholder: self.input_test, output_placeholder: self.output_test}
                test_rmse = (sess.run(net_eval, feed_dict=feed_test_dict))**(0.5)
                if log_file:
                    feed_train_dict = {input_placeholder: self.input_train, output_placeholder: self.output_train}
                    train_rmse = (sess.run(net_eval, feed_dict=feed_train_dict))**(0.5)
                    f_test = open(log_file + "_test.dat","a")
                    f_test.write(str(n) + " " + str(test_rmse) + "\n")
                    f_train = open(log_file + "_train.dat","a")
                    f_train.write(str(n) + " " + str(train_rmse) + "\n")
                    f_test.close()
                    f_train.close()

                if (test_rmse < min_test_rmse):
                    if net_name:
                        save_params = tf.train.Saver()
                        save_params.save(sess,"." + net_name + "_ckpt")
                        f_net = open("." + net_name + "_config", "w")
                        f_net.write(str(self.arch[1:-1]).replace("[","").replace("]",""))
                        f_net.close()
                    if len(self.best_weights) > 0:
                        del self.best_weights[:]
                        del self.best_biases[:]
                    for k in range(len(self.arch)-1):
                        self.best_weights.append(self.weights[k])
                        self.best_biases.append(self.biases[k])
                    min_test_rmse = test_rmse
                print "Epoch number: ",str(n)
                print "The current train rmse is: ", train_rmse
                print "The current test rmse is: ", test_rmse
                print " "
            if hist_file:
                net_best_out = self.__best_activations(sess)
                net_hist_errors = self.__get_hist_errors(net_best_out, self.output_test, sess)
                self.__histtofile(hist_file, net_hist_errors)

    def restore(self, net_name, input_file = ""):
        self.restored = True
        self.net_name = net_name
        if input_file != "":
            f_net = open("." + net_name + "_config", "r")
            arch = f_net.readline().split(",")
            for n in range(len(arch)):
                arch[n] = int(arch[n])
            f_net.close()
            self.init(arch, input_file)
            return
        else:
            return

                    
    def __feed_forward(self,inputs):
        for n in range(self.num_layers - 1):
            weight_name = "weights" + str(n)
            bias_name = "biases" + str(n)
            self.weights.append(tf.Variable(tf.truncated_normal([self.arch[n],self.arch[n+1]], dtype=np.float32), name=weight_name))
            self.biases.append(tf.Variable(tf.truncated_normal([self.arch[n+1]], dtype=np.float32), name=bias_name))
            
        activations = inputs

        if self.classreg == 1:
            for n in range(self.num_layers - 1):
                activations = tf.add(tf.nn.tanh(tf.matmul(activations,self.weights[n]) + self.biases[n]), 0.1*(tf.matmul(activations,self.weights[n]) + self.biases[n]))
        return activations

    """def __errors(self,pred_output, act_output):
        if self.classreg == 1:
            #l2 = tf.nn.l2_loss(tf.sub(pred_output, act_output))
            diff = tf.sub(pred_output, act_output)
            errors = tf.reduce_mean(l2)
        return errors"""

    def __errors(self,pred_output, act_output):
        if self.classreg == 1:
            #l2 = tf.nn.l2_loss(tf.sub(pred_output, act_output))
            diff = tf.sub(pred_output, act_output)
            errors = tf.reduce_mean(tf.mul(diff,diff))
        return errors

    def __trainbatch(self, errors, Opt, learning_rate = 0.0):
        if Opt == "Adam":
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(errors)
        return train_op

    def __best_activations(self,sess):
        activations = self.input_test
        if self.classreg == 1:
            for n in range(self.num_layers - 1):
                activations = tf.add(tf.nn.tanh(tf.matmul(activations,self.best_weights[n]) + self.best_biases[n]), 0.1*(tf.matmul(activations,self.best_weights[n]) + self.best_biases[n])).eval(session=sess)
        return activations

    
    def __get_hist_errors(self, pred_output, act_output, sess):
        return tf.sub(pred_output, act_output).eval(session=sess)

    
    def __histtofile(self, hist_file, errors):
        f = open(hist_file, "w")
        for n in errors:
            f.write(str(n[0]) + "\n")
        return

            
    def __fin_parser(self, strng, sep):
        l1 = strng.split(sep)
        temp = []
        for n in l1:
            temp.append(float(n))
        return temp

    def __reset(self,meth_name):
        if (meth_name == "init"):
            self.weights = []
            self.biases = []
            self.best_weights = []
            self.best_biases = []
            self.input_data = []
            self.input_train = []
            self.input_test = []
            self.output_data = []
            self.output_train = []
            self.output_test = []
            self.num_data = 0
            self.num_train = 0
            self.num_test = 0
            self.mini_batch_size = 0
            self.epochs = 100
            self.classreg = 1
            self.costfunc = 1
            self.numhid = 0
            self.arch = []
            

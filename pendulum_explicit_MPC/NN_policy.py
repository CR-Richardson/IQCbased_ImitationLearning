# Modified: 09/03/23 - CRR - Modified code to allow arbitrary choice of NN depth and width.
# Modified: 09/03/23 - CRR - Added train_split variable.

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import inspect
import pickle
import logz
import scipy.linalg

sys.path.append('C:\\Program Files\\MATLAB\\R2022a\\extern\\engines\\python\\')
import matlab.engine

######################################################################################
######################################################################################
######################################################################################

# function for bulding up neural network
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation, output_activation):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass)

        Hint: use tf.layers.dense
    """
    # raise NotImplementedError
    with tf.variable_scope(scope):
        sy_input = input_placeholder

        # Hidden layers
        hidden_layer = tf.layers.dense(inputs=sy_input,
                                        units=size,
                                        activation=activation,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32), 
                                        use_bias=False)

        for _ in range(n_layers - 1):
            hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                            units=size,
                                            activation=activation,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                            use_bias=False)

        # Output layer
        output_placeholder = tf.layers.dense(inputs=hidden_layer,
                                            units = output_size,
                                            activation=output_activation,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32), 
                                            use_bias=False)
        return output_placeholder

def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean

def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir) # Configure output directory for logging

def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].

    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

#############################################################################
#############################################################################
# define the agent
class Agent(object):
    def __init__(self, ob_dim, ac_dim, n_layers, batch_size, activation, output_activation, sdp_var, iter,
                hyper_param, x1bound, x2bound):
        super(Agent, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.size = hyper_param["size"]
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.sdp_var = sdp_var
        self.iter = iter
        self.rho = hyper_param["rho"]
        self.eta_NN = hyper_param["eta_NN"]
        self.x1bound = x1bound
        self.x2bound = x2bound
        self.nphi = int(n_layers*hyper_param["size"])

    def init_variable(self):
        tf.get_default_session().run(tf.global_variables_initializer()) #pylint: disable=E1101
        self.saver = tf.train.Saver()

    def fN_compute(self):
        """Function which returns the f(N) matrix in the form f(N) = [fNux, fNuw; fNvx, fNvw].
        See Yin et al: Imitation learning with stability guarantees."""

        N = block_diagonal(self.W) # returns N = diag(W1, ..., Wn) = [Nvx, Nvw; Nux, Nuw]
        Nvx = N[:self.nphi, :self.ob_dim]
        Nvw = N[:self.nphi, self.ob_dim:]
        Nux = N[self.nphi:, :self.ob_dim]
        Nuw = N[self.nphi:, self.ob_dim:]

        Alpha = self.Alpha_compute()
        Beta = tf.eye(self.nphi)
        intermediate = tf.linalg.inv(tf.eye(self.nphi) - 1/2*tf.linalg.matmul(Nvw,Alpha+Beta))
        fNvx = tf.linalg.matmul(intermediate, Nvx)
        fNvw = 1/2*tf.linalg.matmul(tf.linalg.matmul(intermediate, Nvw), Beta-Alpha)
        fNux = Nux + tf.linalg.matmul(1/2*tf.linalg.matmul(Nuw,Alpha+Beta), fNvx)
        fNuw = 1/2*tf.linalg.matmul(Nuw, Beta-Alpha) + tf.linalg.matmul(1/2*tf.linalg.matmul(Nuw, Alpha+Beta), fNvw)
        # fN = [fNux, fNuw; fNvx, fNvw];
        fN = tf.concat([tf.concat([fNux, fNvx], 0), tf.concat([fNuw, fNvw], 0)], 1)
        return fN

    def Alpha_compute(self):
        """Function for computing the lower bound on the output of the activation functions.
        See [1] Yin et al: stability analysis using QC for systems with NN controllers
        and [2] Gowal et al: Effectiveness of interval bound propagation ..."""

        # Equilibrium points of the closed loop system - from (5) and (12) in [1]
        # All eq. points of NN are zero if bias' are all zero. Hence v_eq and w_eq are left empty for now.
        x_eq = [0.0, 0.0]
        v_eq = [] # activation function inputs
        w_eq = [] # activation function outputs

        x_ub = tf.constant([[self.x1bound], [self.x2bound]])
        x_lb = -x_ub

        # arrays to store the propagated upper and lower bounds of the NN activation functions
        # input and output
        v_lb = []
        v_ub = []
        w_lb = []
        w_ub = []

        # propagating the bounds through the NN
        mu = 0.5*(x_ub + x_lb)
        r = 0.5*(x_ub - x_lb)
        mu = tf.linalg.matmul(self.W[0], mu)
        r = tf.linalg.matmul(tf.math.abs(self.W[0]), r)
        v_lb.append(mu - r)
        v_ub.append(mu + r)
        w_lb.append(self.activation(v_lb[0]))
        w_ub.append(self.activation(v_ub[0]))

        for j in range(1,len(self.W)):
            mu = 0.5*(w_ub[j-1] + w_lb[j-1])
            r = 0.5*(w_ub[j-1] - w_lb[j-1])
            mu = tf.linalg.matmul(self.W[j], mu)
            r = tf.linalg.matmul(tf.math.abs(self.W[j]), r)
            v_lb.append(mu - r)
            v_ub.append(mu + r)
            if j < len(self.W)-1: # no activation function on the output layer
                w_lb.append(self.activation(v_lb[j]))
                w_ub.append(self.activation(v_ub[j]))

        alpha = []
        for j in range(len(self.W)-1): # output layer has no activation function
            a = tf.math.minimum(tf.math.divide(w_ub[j], v_ub[j]), tf.math.divide(w_lb[j], v_lb[j]))
            a = tf.reshape(a, [self.n[j], ])
            a = tf.matrix_diag(a)
            alpha.append(a)
        
        Alpha = block_diagonal(alpha)
        return Alpha

    def save_variables(self, logdir):
        self.saver.save(tf.get_default_session(), os.path.join(logdir, 'model.ckpt'))

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        # raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        learning_rate_ph = tf.placeholder(tf.float32, shape=[])

        return sy_ob_no, learning_rate_ph

    def build_computation_graph(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.learning_rate_ph = self.define_placeholders()
        self.ac_prediction = build_mlp(
                                input_placeholder = self.sy_ob_no,
                                output_size = self.ac_dim,
                                scope = "nn_action",
                                n_layers=self.n_layers,
                                size=self.size,
                                activation=self.activation,
                                output_activation=self.output_activation)

        
        # initialize the weights in the NN
        tf.get_default_session().run(tf.global_variables_initializer()) #pylint: disable=E1101
        self.W = []
        self.n = []
        for v in tf.global_variables():
            if v.name == "nn_action/dense/kernel:0":
                self.W.append(tf.transpose(v))
                self.n.append(self.W[0].shape[0])

            for j in range(1,self.n_layers+1):    
                if v.name == "nn_action/dense_{}/kernel:0".format(j):
                    self.W.append(tf.transpose(v))
                    self.n.append(self.W[j].shape[0])
        
        self.ac_data = tf.placeholder(shape=[None, self.ac_dim], name="ac_true", dtype=tf.float32)
        if self.iter == 0: # nominal imitation learning
            self.loss = tf.losses.mean_squared_error(self.ac_data, self.ac_prediction)
            self.mse = self.loss
        else: # safe imitation learning
            self.Q1 = np.array(self.sdp_var["Q1"],dtype='float32')
            self.Q2 = np.array(self.sdp_var["Q2"],dtype='float32')
            self.L1 = np.array(self.sdp_var["L1"],dtype='float32')
            self.L2 = np.array(self.sdp_var["L2"],dtype='float32')
            self.L3 = np.array(self.sdp_var["L3"],dtype='float32')
            self.L4 = np.array(self.sdp_var["L4"],dtype='float32')
            self.Yk = np.array(self.sdp_var["Yk"],dtype='float32')
            Q = scipy.linalg.block_diag(self.Q1, self.Q2)
            L = np.block([[self.L1, self.L2], [self.L3, self.L4]])
            
            self.fN = self.fN_compute()

            self.mse = tf.losses.mean_squared_error(self.ac_data, self.ac_prediction)
            self.loss = self.eta_NN*tf.losses.mean_squared_error(self.ac_data, self.ac_prediction) + tf.linalg.trace(tf.matmul(np.transpose(self.Yk), tf.matmul(self.fN, Q)-L)) + 0.5*self.rho*tf.math.square(tf.norm(tf.matmul(self.fN, Q)-L, 'fro', (0, 1)))
           
        self.update_op = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.loss)

    def update_NN(self, ob_no, ac_na, learning_rate):
        tf.get_default_session().run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.ac_data: ac_na, self.learning_rate_ph: learning_rate})

    def compute_ac(self, ob_no):
        ac = tf.get_default_session().run([self.ac_prediction], feed_dict={self.sy_ob_no: ob_no})
        return ac

    def compute_loss(self, ob_no, ac_na):
        temp_loss = tf.get_default_session().run([self.loss], feed_dict={self.sy_ob_no: ob_no, self.ac_data: ac_na})
        temp_mse = tf.get_default_session().run([self.mse], feed_dict={self.sy_ob_no: ob_no, self.ac_data: ac_na})
        return temp_loss, temp_mse

# solve the NN training problem using SGD
def solve_NNfit(ob_dim, ac_dim, n_layers, batch_size, activation, output_activation, xu_data, train_split, 
                n_epoch, sdp_var, iter, hyper_param, logdir, x1bound, x2bound):
    # setup logger
    setup_logger(logdir, locals())

    agent = Agent(ob_dim, ac_dim, n_layers, batch_size, activation, output_activation, 
    sdp_var, iter, hyper_param, x1bound, x2bound)

    # tensorflow: config, session initialization
    tf_config = tf.ConfigProto(inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1)
    tf_config.gpu_options.allow_growth = True # may need if using GPU
    sess = tf.Session(config=tf_config)
    with sess:
        # build computaion graph
        agent.build_computation_graph()

        # tensorflow: variable initialization
        agent.init_variable()

        # number of data points
        n_train_samples = int(len(xu_data[:,0])*train_split)
        x_train = xu_data[:n_train_samples,:2]
        u_train = xu_data[:n_train_samples,2:]
        x_test = xu_data[n_train_samples:,:2]
        u_test = xu_data[n_train_samples:,2:]

        train_loss_list = []
        test_loss_list = []
        for epoch in range(n_epoch):
            rand_index = np.random.choice(n_train_samples, size=batch_size) # random data samples
            x_batch = x_train[rand_index, :]
            u_batch = u_train[rand_index, :]

            learning_rate = 1e-3/(1 + 3*epoch/n_epoch)
            agent.update_NN(x_batch, u_batch, learning_rate)
            train_loss, train_mse = agent.compute_loss(x_batch, u_batch)
            test_loss, test_mse = agent.compute_loss(x_test, u_test)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            if epoch%(n_epoch/10) == 0:
                print("********** Iteration %i ************"%epoch)
                print('Training Loss = ', train_loss)
                print('Test Loss = ', test_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        
        agent.save_variables(logdir)
        logz.pickle_tf_vars()
        logz.save_params(hyper_param)

        if True:
            train_curve, = plt.plot(train_loss_list, 'r--', label='training Loss')
            test_curve, = plt.plot(test_loss_list, 'k--', label='test Loss')
            plt.legend(handles = [train_curve, test_curve])
            plt.xlabel('number of iterations')
            plt.ylabel('mean squared error')
            plt.title('NN policy training curve')
            plt.savefig(os.path.join(logdir, 'loss_vs_epoch'))
        
        W = []
        for j in range(len(agent.W)):
            W.append(agent.W[j].eval())

    sess.close()
    tf.reset_default_graph()
    
    return W, train_mse, test_mse

def main():

    size = 10 # size of each hidden layer
    n_layers = 5 # number of hidden layers
    n_iter = 3 #17 # number of iterations of safe imitation learning
    n_epoch = 100 #40000 # number of epochs for training the NN controller
    rho = 1
    eta_ROA = 5
    eta_NN = 100

    print(logz.colorize("Safe imitation learning begins", 'red', bold=True))
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'NN_policy' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    eng = matlab.engine.start_matlab()
    ob_dim = 2
    ac_dim = 1
    hyper_param = {"rho": rho, "eta_ROA": eta_ROA, "eta_NN": eta_NN, "size": size}
    activation = tf.nn.tanh
    output_activation = None
    x1bound = 2.5
    x2bound = 6.0
    x_eq = [[0.0], [0.0]]

    # load (x, u) data pairs
    exp_data = pd.read_csv('exp_data2.csv', sep = ',')
    data = np.array(exp_data.values)
    batch_size = 500
    train_split = 0.9

    # construct the dynamics for the pendulum
    g = 10 # gravitational coefficient
    m = 0.15 # mass
    l = 0.5 # length
    mu = 0.05 # frictional coefficient
    dt = 0.02 # sampling period
    AG = np.array([[1, dt], [g/l*dt, 1-mu/(m*l**2)*dt]])
    BG = np.array([[0], [dt/(m*l**2)]])
    nG = AG.shape[0]

    # initialize sdp_var
    sdp_var = {"Q1": np.zeros((nG,nG))}
    param = {}
    param["logdir"] = logdir
    param["AG"] = matlab.double(AG.tolist())
    param["BG"] = matlab.double(BG.tolist())
    param["rho"] = matlab.double([rho])
    param["eta_ROA"] = matlab.double([eta_ROA])
    param["eta_NN"] = matlab.double([eta_NN])
    param["x1bound"] = matlab.double([x1bound])
    param["x2bound"] = matlab.double([x2bound])
    param["x_eq"] = matlab.double(x_eq)
    param["n_iter"] = matlab.int32(n_iter)

    for i in range(n_iter):
        print(logz.colorize("safe learning iteration" + str(i), 'green', bold=True))
        
        # NN trainig step
        W, train_mse, test_mse = solve_NNfit(ob_dim, ac_dim, n_layers, batch_size, activation, output_activation,
        data, train_split, n_epoch,  sdp_var, i, hyper_param, os.path.join(logdir,'%d'%i),x1bound, x2bound)

        param["train_mse"] = matlab.double(float(train_mse[0])) # pass to matlab so all objective values can be stored after each iteration
        param["test_mse"] = matlab.double(float(test_mse[0]))
        param["iter"] = matlab.int64([i])
        param["path"] = os.path.join(logdir,'%d'%i)

        for j in range(len(W)):
            param['W{}'.format(j+1)] = matlab.double(W[j].tolist())

        # sdp step and Yk update step
        print("STARTING SDP")
        sdp_var = eng.solve_sdp(param)
        param["Yk"] = sdp_var["Yk"]
        print("FINISHING SDP")

if __name__ == "__main__":
    main()
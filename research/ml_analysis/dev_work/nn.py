"""
Module to mimic a Neural Network
"""
# import pdb
from math import tanh
import sqlite3

class NeuralNet:
    """
    Class to construct a neural network
    """
    def __init__(self, dbname):
        """
        Constructor
        """
        self.con = sqlite3.connect(dbname)
        # value lists
        self.wordids = []
        self.hiddenids = []
        self.urlids = []

        # outputs of actual nodes (input, hidden, and output)
        self.out_i = []
        self.out_h = []
        self.out_o = []

        # weights from input nodes to hidden nodes (wgt_i) and hidden nodes to output nodes (wgt_o)
        self.wgt_i = []
        self.wgt_o = []

    def __del__(self):
        """
        Close the DB
        """
        self.con.close()

    def make_tables(self):
        """
        Create the tables needed for the NN
        """
        self.con.execute('drop table if exists hiddennode')
        self.con.execute('drop table if exists wordhidden')
        self.con.execute('drop table if exists hiddenurl')
        self.con.execute('create table hiddennode(create_key)')
        self.con.execute('create table wordhidden(fromid,toid,strength)')
        self.con.execute('create table hiddenurl(fromid,toid,strength)')
        self.con.commit()

    def get_strength(self, fromid, toid, layer):
        """
        Get the strength of a given connection
        """
        # get the correct layer
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'hiddenurl'

        res = self.con.execute('select strength from %s where fromid=%d and toid=%d'
                               % (table, fromid, toid)).fetchone()
        if res is None:
            # default values
            if layer == 0:
                return -0.2
            if layer == 1:
                return 0
        return res[0]

    def set_strength(self, fromid, toid, layer, strength):
        """
        Sets the strength of a given connection between layers
        """
        # get the correct layer
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'hiddenurl'
        # See if connection exists
        res = self.con.execute('select rowid from %s where fromid=%d and toid=%d'
                               % (table, fromid, toid)).fetchone()
        if res is None:
            # If it does not exist, set it to given strength
            self.con.execute('insert into %s (fromid,toid,strength) values (%d,%d,%f)'
                             % (table, fromid, toid, strength))
        else:
            # If it exists, update it
            rowid = res[0]
            self.con.execute('update %s set strength=%f where rowid=%d'
                             % (table, strength, rowid))

    def gen_hidden_node(self, wordids, urls):
        """
        Create a hidden node for given set of inputs
        """
        if len(wordids) > 3:
            return
        # Check if we already created a node for this set of inputs
        sorted_words = [str(id) for id in wordids]
        sorted_words.sort()
        createkey = '_'.join(sorted_words)
        res = self.con.execute("select rowid from hiddennode where create_key='%s'"
                               % createkey).fetchone()

        # If it does not exits, create it
        if res is None:
            cur = self.con.execute("insert into hiddennode (create_key) values ('%s')"
                                   % createkey)
            hiddenid = cur.lastrowid
            for wordid in wordids:
                # default strength of (1 / number of nodes) going into hidden nodes
                self.set_strength(wordid, hiddenid, 0, 1.0 / len(wordids))
            for urlid in urls:
                # default strength of 0.1 coming out of hidden nodes
                self.set_strength(hiddenid, urlid, 1, 0.1)
            self.con.commit()

    def get_all_hidden_ids(self, wordids, urlids):
        """
        Find all the nodes in the hidden layer that are relevant to the specific query
        """
        nodes = {}
        for wordid in wordids:
            # get every hidden node connected to one of the inputs
            cur = self.con.execute('select toid from wordhidden where fromid=%d' % wordid)
            for row in cur:
                nodes[row[0]] = 1
        for urlid in urlids:
            # Gets every hidden node connected to one of the outputs
            cur = self.con.execute('select fromid from hiddenurl where toid=%d' % urlid)
            for row in cur:
                nodes[row[0]] = 1
        # return all the node IDs
        return nodes.keys()

    def setup_network(self, wordids, urlids):
        """
        Constructing the relevant network with the current weights from the DB
        """
        # value lists
        self.wordids = wordids
        self.hiddenids = list(self.get_all_hidden_ids(wordids, urlids))
        self.urlids = urlids

        # outputs of actual nodes (input, hidden, and output)
        self.out_i = [1.0] * len(self.wordids)
        self.out_h = [1.0] * len(self.hiddenids)
        self.out_o = [1.0] * len(self.urlids)

        # create weights matrix
        # weights from inputs to hidden layer
        self.wgt_i = [[self.get_strength(wordid, hiddenid, 0) for hiddenid in self.hiddenids]
                      for wordid in self.wordids]
        # weights from hidden layer to outputs
        self.wgt_o = [[self.get_strength(hiddenid, urlid, 1) for urlid in self.urlids]
                      for hiddenid in self.hiddenids]

    def feed_forward(self):
        """
        Takes a list of inputs, pushes them through the network, and returns the
        output of all the nodes in the output layer updating the node values in the process
        """
        # the only inputs are the query words
        for i in range(len(self.wordids)):
            self.out_i[i] = 1.0

        # hidden node activations
        # pdb.set_trace()
        for j in range(len(self.hiddenids)):
            tot_sum = 0.0
            for i in range(len(self.wordids)):
                # sum of all input node values coming to a hidden node * input connection weight
                tot_sum = tot_sum + (self.out_i[i] * self.wgt_i[i][j])
            # sum placed in the activation function (tanh) to get value for hidden node
            self.out_h[j] = tanh(tot_sum)

        # and that sum placed in the activation (in this case tanh)
        # output activations
        for k in range(len(self.urlids)):
            tot_sum = 0.0
            # sum of all node hidden values coming to an output node * output connection weights
            for j in range(len(self.hiddenids)):
                tot_sum = tot_sum + self.out_h[j] * self.wgt_o[j][k]
            # sum placed in the activation function (tanh) to get value for an output node
            self.out_o[k] = tanh(tot_sum)
        return self.out_o

    def get_result(self, wordids, urlids):
        """
        Setup the network and then call the feed forward function for the output
        """
        self.setup_network(wordids, urlids)
        return self.feed_forward()

    def back_propagate(self, targets, learn_rate=0.5):
        """
        Updating weights from outputs back to input connections
        N is an input variable to decide how quickly the updates get factored in
        """
        output_deltas = [0.0] * len(self.urlids)

        # output deltas --> basically how much our guesses were off by
        for k in range(len(self.urlids)):
            # calculate the error for each output node
            error = targets[k] - self.out_o[k]
            # calculate how much the value should change
            output_deltas[k] = dtanh(self.out_o[k]) * error

        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range(len(self.hiddenids)):
            error = 0.0
            for k in range(len(self.urlids)):
                # calculate errors for each hidden node --> sum of all output_deltas * wgts
                error = error + output_deltas[k] * self.wgt_o[j][k]
            # hidden_deltas --> dtanh(node value) * (sum)
            hidden_deltas[j] = dtanh(self.out_h[j]) * error

        # update weights from hidden to output
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                # wgt update for each node --> output_delta * hidden node value * learn_rate
                change = output_deltas[k] * self.out_h[j]
                # wgt is adjusted from current value by the calculated change
                self.wgt_o[j][k] = self.wgt_o[j][k] + learn_rate * change

        # update weights from input to hidden
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                # wgt update for each node --> hidden_delta * input node value * learn_rate
                change = hidden_deltas[j] * self.out_i[i]
                # wgt is adjusted from current value by the calculated change
                self.wgt_i[i][j] = self.wgt_i[i][j] + learn_rate * change

    def train_query(self, wordids, urlids, selected_url, prnt=True):
        """
        Take an input and labeled output and pass to back propagation to update
        node values and connection weights
        """
        # generate a hidden node if necessary
        self.gen_hidden_node(wordids, urlids)

        # load network from DB and feed forward to setup up node values based on DB
        self.setup_network(wordids, urlids)
        self.feed_forward()
        targets = [0.0] * len(urlids)

        if prnt:
            print("training input: " + str(wordids) + "    training outputs: " + str(urlids) +
                  "    labeled output: " + str(selected_url))
            print("Before:")
            self.print_weights()

        # Set the labeled url to 1 and the rest to 0 and pass that to backpropigation
        targets[urlids.index(selected_url)] = 1.0
        self.back_propagate(targets)
        # error = self.back_propagate(targets)
        self.update_database()
        # self.net_update(wordids, urlids, selected_url)

        if prnt:
            print("After:")
            self.print_weights()

    def print_weights(self):
        """
        Print all of the connection weights between layers
        """
        print("input weights --> all weights from input layer to hidden layer:")
        k = 0
        # Print the weights from the input layer to the hidden layer
        for i in self.wgt_i:
            inp_str = str(self.wordids[k]) + ": "
            for inp in i:
                inp_str += str(round(inp, 4)) + ", "
            inp_str = inp_str[:-2]
            print(inp_str)
            k += 1

        k = 0
        # Print the weights from the input layer to the hidden layer
        print("output weights --> all weights from hidden layer to output layer:")
        for wgt_out in self.wgt_o:
            out_str = str(self.get_hidden_key(self.hiddenids[k])[0]) + ": "
            # out_str = str(self.hiddenids[k]) + ": "
            for out in wgt_out:
                out_str += str(round(out, 4)) + ", "
            out_str = out_str[:-2]
            print(out_str)
            k += 1

    def update_database(self):
        """
        Updates the database with the new weight strengths
        """
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                self.set_strength(self.wordids[i], self.hiddenids[j], 0, self.wgt_i[i][j])
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                self.set_strength(self.hiddenids[j], self.urlids[k], 1, self.wgt_o[j][k])
        self.con.commit()

    def get_hidden_key(self, hidden_id):
        """
        Take a hidden node ID and return the node key
        """
        return self.con.execute("select create_key from hiddennode where rowid='%s'"
                                % hidden_id).fetchone()

def dtanh(y_val):
    """
    Calculates the slope of the tanh function
    Which determines how much the node's total output has to change
    Part of the sigmoid family of functions
    """
    return 1.0 - y_val * y_val


if __name__ == '__main__':
    MYNET = NeuralNet('nn.db')
    MYNET.make_tables()
    WWORLD, WRIVER, WBANK = 101, 102, 103
    UWORLDBANK, URIVER, UEARTH = 201, 202, 203
    # Creates a hidden node with connections from the first input list to the second output list
    MYNET.gen_hidden_node([WWORLD, WBANK], [UWORLDBANK, URIVER, UEARTH])

    # Test nodes created
    # for node in MYNET.con.execute('select * from wordhidden'):
    #     # prints all the connections from inputs to hidden nodes
    #     print(node)
    # for node in MYNET.con.execute('select * from hiddenurl'):
    #     # prints all the connections from hidden nodes to outputs
    #     print(node)

    # Test Feedforward functionality
    # print(MYNET.get_result([WWORLD, WBANK], [UWORLDBANK, URIVER, UEARTH]))

    # Train query exercise
    # MYNET.train_query([WWORLD, WBANK], [UWORLDBANK, URIVER, UEARTH], UWORLDBANK)
    # print(MYNET.get_result([WWORLD, WBANK], [UWORLDBANK, URIVER, UEARTH]))

    # Neural Network Example
    ALLURLS = [UWORLDBANK, URIVER, UEARTH]
    for _ in range(30):
        MYNET.train_query([WWORLD, WBANK], ALLURLS, UWORLDBANK)
        MYNET.train_query([WRIVER, WBANK], ALLURLS, URIVER)
        MYNET.train_query([WWORLD], ALLURLS, UEARTH)
    print(MYNET.get_result([WWORLD, WBANK], ALLURLS))
    print(MYNET.get_result([WRIVER, WBANK], ALLURLS))
    print(MYNET.get_result([WBANK], ALLURLS))

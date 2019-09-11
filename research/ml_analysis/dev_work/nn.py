"""
Module to mimic a Neural Network
"""
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
        # Check if we already created a node for this set of words
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
            # Put in some default weights for all connections going into and out
            # of hidden node
            for wordid in wordids:
                self.set_strength(wordid, hiddenid, 0, 1.0 / len(wordids))
            for urlid in urls:
                self.set_strength(hiddenid, urlid, 1, 0.1)
            self.con.commit()

if __name__ == '__main__':
    MYNET = NeuralNet('nn.db')
    MYNET.make_tables()
    WWORLD, WRIVER, WBANK = 101, 102, 103
    UWORLDBACK, URIVER, UEARTH = 201, 202, 203
    MYNET.gen_hidden_node([WWORLD, WBANK], [UWORLDBACK, URIVER, UEARTH])

    # Test nodes created
    for node in MYNET.con.execute('select * from wordhidden'):
        # prints all the connections from inputs to hidden nodes
        print(node)
    for node in MYNET.con.execute('select * from hiddenurl'):
        # prints all the connections from hidden nodes to outputs
        print(node)

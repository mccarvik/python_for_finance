"""
Class to crawl through web pages
"""
import pdb
import re
import warnings
import sqlite3
import urllib
import urllib.request
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import nn
MYNET = nn.NeuralNet('nn.db')

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Create a list of words to ignore
IGNORE_WORDS = {'the':1, 'of':1, 'to':1, 'and':1, 'a':1, 'in':1, 'is':1, 'it':1}
DATA_PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/dev_work/dev_data/"

class Crawler:
    """
    Class designed for recursively crawling throuhg webpages and index them
    """
    def __init__(self, dbname):
        """
        Constructor
        """
        # Initialize the crawler with the name of database
        self.con = sqlite3.connect(DATA_PATH + dbname)

    def __del__(self):
        """
        Close the Database
        """
        self.con.close()

    def dbcommit(self):
        """
        Commit to the DB
        """
        self.con.commit()

    def get_entry_id(self, table, field, value):
        """
        Auxilliary function for getting an entry id and adding it if it's not present
        """
        cur = self.con.execute("select rowid from %s where %s='%s'" % (table, field, value))
        res = cur.fetchone()
        # Check if id exists
        if res is None:
            # Add to table if id does not exist
            cur = self.con.execute("insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        return res[0]

    def add_to_index(self, url, soup):
        """
        Index an individual page
        """
        # Check if its been indexed
        if self.is_indexed(url):
            return
        print('Indexing ' + url)

        # Get the individual words
        text = self.get_text_only(soup)
        words = separate_words(text)

        # Get the URL id
        urlid = self.get_entry_id('urllist', 'url', url)

        # Link each word to this url
        for i, _ in enumerate(words):
            word = words[i]
            if word in IGNORE_WORDS:
                continue
            wordid = self.get_entry_id('wordlist', 'word', word)
            # insert word location and url into table
            self.con.execute("insert into wordlocation(urlid,wordid,location) values" \
                             "(%d,%d,%d)" % (urlid, wordid, i))

    def get_text_only(self, soup):
        """
        Recursively go through the tags in the html and only return plain text
        """
        val = soup.string
        # see if we have a text element
        if val is None:
            conts = soup.contents
            resulttext = ''
            # not text so continue recursing through the tags
            for tag in conts:
                subtext = self.get_text_only(tag)
                resulttext += subtext + '\n'
            return resulttext
        return val.strip()

    def is_indexed(self, url):
        """
        Return true if this url is already indexed
        """
        query = self.con.execute("select rowid from urllist where url='%s'" % url).fetchone()
        if query is not None:
            # Check if it actually has been crawled
            crawled = self.con.execute('select * from wordlocation where urlid=%d'
                                       % query[0]).fetchone()
            if crawled is not None:
                return True
        return False

    def add_link_ref(self, url_from, url_to, link_text):
        """
        Add a link between two pages and insert to linkwords indicating a query
        word was in the link text
        """
        words = separate_words(link_text)
        fromid = self.get_entry_id('urllist', 'url', url_from)
        toid = self.get_entry_id('urllist', 'url', url_to)
        if fromid == toid:
            return
        # insert into link table indicating the urls involved
        cur = self.con.execute("insert into link(fromid,toid) values (%d,%d)" % (fromid, toid))
        linkid = cur.lastrowid
        # insert into linkwords the words associated with the link
        for word in words:
            if word in IGNORE_WORDS:
                continue
            wordid = self.get_entry_id('wordlist', 'word', word)
            self.con.execute("insert into linkwords(linkid,wordid) values (%d,%d)"
                             % (linkid, wordid))

    def crawl(self, pages, depth=2):
        """
        Starting with a list of pages, do a breadth first search to the given depth,
        indexing pages as we go
        """
        # Only go so deep
        for _ in range(depth):
            newpages = {}
            for page in pages:
                try:
                    # Open page
                    contents = urllib.request.urlopen(page)
                except Exception:
                    print("Could not open %s" % page)
                    continue
                try:
                    # Read page
                    soup = BeautifulSoup(contents.read())
                    self.add_to_index(page, soup)
                    # get all the links
                    links = soup('a')
                    for link in links:
                        if 'href' in dict(link.attrs):
                            url = urljoin(page, link['href'])
                            if url.find("'") != -1:
                                continue
                            url = url.split('#')[0]  # remove location portion
                            if url[0:4] == 'http' and not self.is_indexed(url):
                                newpages[url] = 1
                            link_text = self.get_text_only(link)
                            self.add_link_ref(page, url, link_text)
                    self.dbcommit()
                except Exception:
                    print("Could not parse page %s" % page)
            pages = newpages

    def create_index_tables(self):
        """
        Create all the tables we will need in the DB
        """
        # List of urls that have been indexed
        self.con.execute('create table urllist(url)')
        # List of words
        self.con.execute('create table wordlist(word)')
        # What doc the word is and where it is in the doc
        self.con.execute('create table wordlocation(urlid, wordid, location)')
        # Indicates a link from one url to another
        self.con.execute('create table link(fromid integer, toid integer)')
        # which words are actually in a link
        self.con.execute('create table linkwords(wordid, linkid)')
        self.con.execute('create index wordidx on wordlist(word)')
        self.con.execute('create index urlidx on urllist(url)')
        self.con.execute('create index wordurlidx on wordlocation(wordid)')
        self.con.execute('create index urltoidx on link(toid)')
        self.con.execute('create index urlfromidx on link(fromid)')
        self.dbcommit()

    def calculate_page_rank(self, iterations=5):
        """
        Calculates the page rank score based on how many other pages reference
        this page and how popular those pages are, and how many links they have
        on their page
        """
        # clear out the current page rank tables
        self.con.execute('drop table if exists pagerank')
        self.con.execute('create table pagerank(urlid primary key,score)')

        # initialize every url with a page rank of 1
        for (urlid,) in self.con.execute('select rowid from urllist'):
            self.con.execute('insert into pagerank(urlid,score) values (%d,1.0)' % urlid)
        self.dbcommit()

        for i in range(iterations):
            # Need multiple iterations, as the page ranks of pages linked to this
            # one will be consistently updated on each iteration
            print("Iteration %d" % i)
            for (urlid,) in self.con.execute('select rowid from urllist'):
                # Default page rank
                page_rank = 0.15

                # Loop through all the pages that link to this one
                for (linker,) in self.con.execute('select distinct fromid from link where toid=%d'
                                                  % urlid):
                    # Get the page rank of the linker
                    linkingpr = self.con.execute('select score from pagerank where urlid=%d'
                                                 % linker).fetchone()[0]

                    # Get the total number of links from the linker
                    linkingcount = self.con.execute('select count(*) from link where fromid=%d'
                                                    % linker).fetchone()[0]
                    # add to page rank, accounting for the link count
                    page_rank += 0.85 * (linkingpr / linkingcount)
                self.con.execute('update pagerank set score=%f where urlid=%d'
                                 % (page_rank, urlid))
            self.dbcommit()


class Searcher:
    """
    Class to query and edit data from the DBs created by the Crawler class
    """
    def __init__(self, dbname):
        """
        Constructor
        """
        self.con = sqlite3.connect(DATA_PATH + dbname)

    def __del__(self):
        """
        Close the database
        """
        self.con.close()

    def get_match_rows(self, string):
        """
        Takes a query string and queries the DB for all the URLS with these words
        """
        # Strings to build the query
        fieldlist = 'w0.urlid'
        tablelist = ''
        clauselist = ''
        wordids = []

        # Split the words by spaces
        words = string.split(' ')
        tablenumber = 0

        for word in words:
            # Get the word ID
            wordrow = self.con.execute("select rowid from wordlist where word='%s'"
                                       % word).fetchone()
            # Builds the query
            if wordrow is not None:
                wordid = wordrow[0]
                wordids.append(wordid)
                if tablenumber > 0:
                    tablelist += ','
                    clauselist += ' and '
                    clauselist += 'w%d.urlid=w%d.urlid and ' % (tablenumber-1, tablenumber)
                fieldlist += ',w%d.location' % tablenumber
                tablelist += 'wordlocation w%d' % tablenumber
                clauselist += 'w%d.wordid=%d' % (tablenumber, wordid)
                tablenumber += 1

        if not tablelist:
            print("No matches found")
            return None

        # Create the query from the separate parts
        fullquery = 'select %s from %s where %s' % (fieldlist, tablelist, clauselist)
        print(fullquery)
        cur = self.con.execute(fullquery)
        rows = [row for row in cur]
        return rows, wordids

    def get_scored_list(self, rows, wordids):
        """
        Get the scores for each url and totals them based on weight
        Rows has 3 elements - [url_id, loc of word 1, loc of word 2]
        Will have every combination of locations between the two words
        """
        totalscores = dict([(row[0], 0) for row in rows])
        # This is where we'll put our scoring functions
        weights = [(1.0, location_score(rows)),
                   (1.0, frequency_score(rows)),
                   (1.0, distance_score(rows)),
                   (1.0, self.page_rank_score(rows)),
                   (1.0, self.link_text_score(rows, wordids)),
                   #  (5.0, self.nnscore(rows, wordids))
                  ]
        # Sum up weighted scores
        for (weight, scores) in weights:
            for url in totalscores:
                totalscores[url] += weight * scores[url]
        return totalscores

    def get_url_name(self, url_id):
        """
        gets the url name from the urllist table based on url id
        """
        return self.con.execute("select url from urllist where rowid=%d"
                                % url_id).fetchone()[0]

    def query(self, string):
        """
        Query for the page rank scores and return a sorted list
        """
        rows, wordids = self.get_match_rows(string)
        scores = self.get_scored_list(rows, wordids)
        rankedscores = [(score, url) for (url, score) in scores.items()]
        rankedscores.sort()
        rankedscores.reverse()
        for (score, urlid) in rankedscores[0:10]:
            print('%f\t%s' % (score, self.get_url_name(urlid)))
        return wordids, [r[1] for r in rankedscores[0:10]]

    def inbound_link_score(self, rows):
        """
        Count the number of inbound links on each page and use the total number as a metric
        """
        uniqueurls = dict([(row[0], 1) for row in rows])
        inboundcount = dict([(u, self.con.execute('select count(*) from link where toid=%d'
                                                  % u).fetchone()[0]) for u in uniqueurls])
        return normalize_scores(inboundcount)

    def link_text_score(self, rows, wordids):
        """
        Uses the text of the links to a page to decide how relevant the page is
        """
        # create default score of 0 fro each url
        linkscores = dict([(row[0], 0) for row in rows])
        for wordid in wordids:
            # get all the links
            cur = self.con.execute('select link.fromid,link.toid from linkwords,link where wordid=%d and linkwords.linkid=link.rowid' % wordid)
            for (fromid, toid) in cur:
                if toid in linkscores:
                    # if word is in the link, get the links page score
                    page_rank = self.con.execute('select score from pagerank where urlid=%d'
                                                 % fromid).fetchone()[0]
                    linkscores[toid] += page_rank
        maxscore = max(linkscores.values())
        if not maxscore:
            maxscore = 0.00001
        normalizedscores = dict([(u, float(l) / maxscore) for (u, l) in linkscores.items()])
        return normalizedscores

    def page_rank_score(self, rows):
        """
        Grabs the pageranks from the DB and then normalizes them
        """
        pageranks = dict([(row[0], self.con.execute('select score from pagerank where urlid=%d'
                                                    % row[0]).fetchone()[0]) for row in rows])
        maxrank = max(pageranks.values())
        normalizedscores = dict([(u, float(l) / maxrank) for (u, l) in pageranks.items()])
        return normalizedscores


def normalize_scores(scores, small_is_better=0):
    """
    Some functions have a lower score as better, some a higher score
    This function will always return a number between 1 and 0 where 1 is the best
    score possible
    """
    # Avoid division by zero errors
    vsmall = 0.00001
    if small_is_better:
        minscore = min(scores.values())
        return dict([(u, float(minscore) / max(vsmall, l)) for (u, l) in scores.items()])
    else:
        maxscore = max(scores.values())
        if maxscore == 0:
            maxscore = vsmall
        return dict([(u, float(c) / maxscore) for (u, c) in scores.items()])


def frequency_score(rows):
    """
    Counts how many times a url comes up amongst these words
    Not a true count as it counts combinations of the words involved not just
    the absolute count. ex: will favor a url with 3 of each of 2 words vs. 5 and 1
    """
    counts = dict([(row[0], 0) for row in rows])
    for row in rows:
        counts[row[0]] += 1
    return normalize_scores(counts)


def location_score(rows):
    """
    How close the location of the word is to the beginning of the URL
    """
    # hard max of 1,000,000
    locations = dict([(row[0], 1000000) for row in rows])
    for row in rows:
        # gets a sum of the locations of the word in each url
        loc = sum(row[1:])
        if loc < locations[row[0]]:
            locations[row[0]] = loc
    return normalize_scores(locations, small_is_better=1)


def distance_score(rows):
    """
    how far appart the words in the query are in the URL
    """
    # If there's only one word, everyone wins!
    if len(rows[0]) <= 2:
        return dict([(row[0], 1.0) for row in rows])

    # Initialize the dictionary with large values
    mindistance = dict([(row[0], 1000000) for row in rows])

    for row in rows:
        dist = sum([abs(row[i]-row[i-1]) for i in range(2, len(row))])
        # get the min distance for this url
        if dist < mindistance[row[0]]:
            mindistance[row[0]] = dist
    return normalize_scores(mindistance, small_is_better=1)


def nn_score(rows, wordids):
    """
    Grab the score from the neural network
    """
    pdb.set_trace()
    # Get unique URL IDs as an ordered list
    urlids = [urlid for urlid in dict([(row[0], 1) for row in rows])]
    nnres = MYNET.get_result(wordids, urlids)
    scores = dict([(urlids[i], nnres[i]) for i in range(len(urlids))])
    return normalize_scores(scores)


def separate_words(text):
    """
    Get a body of text and seperate into words by any non-whitespace character
    """
    splitter = re.compile('\\W*')
    return [s.lower() for s in splitter.split(text) if s != '']


if __name__ == '__main__':
    ##### Crawler demo #####
    # PAGE_LIST = ['https://en.wikipedia.org/wiki/Sport']
    CRAWL = Crawler('page_index.db')
    # CRAWL.create_index_tables()
    # CRAWL.crawl(PAGE_LIST, depth=2)
    # print([row for row in CRAWL.con.execute('select rowid from wordlocation where wordid=1')])
    # CRAWL.calculate_page_rank()

    ##### Query Demo #####
    SRCH = Searcher('page_index.db')
    # print(SRCH.get_match_rows('search lucky'))

    ##### Scoring Demo #####
    print(SRCH.query('stadiums'))

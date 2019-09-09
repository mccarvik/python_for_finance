"""
Allows for persistent on the line training
"""
import re
import pdb
import feedparser

DATA_PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/dev_work/dev_data/"

def read(feed, classifier):
    """
    Takes a filename of URL of a blog feed and classifies the entries
    """
    # Get feed entries and loop over them
    ind_f = feedparser.parse(DATA_PATH + feed)
    for entry in ind_f['entries']:
        print()
        print('-----')
        # Print the contents of the entry
        print('Title:     ' + entry['title'])
        print('Publisher: ' + entry['publisher'])
        print()
        print(entry['summary'])

        # Combine all the text to create one item for the classifier
        fulltext = '%s\n%s\n%s' % (entry['title'], entry['publisher'], entry['summary'])

        # Print the best guess at the current category
        # Need to switch between entry and fultext variables depending on feature func
        # print('Guess: ' + str(classifier.classify(fulltext)))
        print('Guess: ' + str(classifier.classify(entry)))
        # Ask the user to specify the correct category and train on that
        cat = input('Enter category: ')
        # classifier.train(fulltext, cat)
        classifier.train(entry, cat)

def entryfeatures(entry):
    """
    More advanced feature extraction than just alphanumeric
    """
    splitter = re.compile('\\W*')
    feed = {}

    # Extract the title words and annotate
    titlewords = [s.lower() for s in splitter.split(entry['title'])
                  if len(s) > 2 and len(s) < 20]
    for word in titlewords:
        feed['Title:' + word] = 1

    # Extract the summary words
    summarywords = [s.lower() for s in splitter.split(entry['summary'])
                    if len(s) > 2 and len(s) < 20]

    # Count uppercase words
    u_case = 0
    for idx, _ in enumerate(summarywords):
        word = summarywords[idx]
        feed[word] = 1
        if word.isupper():
            u_case += 1

        # Get word pairs in summary as features
        if idx < len(summarywords)-1:
            twowords = ' '.join(summarywords[idx:idx+1])
            feed[twowords] = 1

    # Keep creator and publisher whole
    feed['Publisher:' + entry['publisher']] = 1

    # UPPERCASE is a virtual word flagging too much shouting
    if float(u_case) / len(summarywords) > 0.3:
        feed['UPPERCASE'] = 1
    return feed

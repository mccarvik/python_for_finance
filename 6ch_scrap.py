import numpy as np
import pandas as pd

def intro():
    df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
    # print(df.index)                     # index vlues
    # print(df.columns)                   # columns
    # print(df.ix['c'])                   # selection via index
    # print(df.ix[['a','d']])             # selection of multiple indices
    # print(df.ix[df.index[1:3]])         # selection vie index object
    # print(df.sum())                     # sum per column
    # print(df.apply(lambda x: x ** 2))   # square of every element
    # print (df ** 2)
    df['floats'] = (1.5, 2.5, 3.5, 4.5) 
    # print(df)
    # print(df['floats'])
    df['names'] = pd.DataFrame(['Yves', 'Guido', 'Felix', 'Francesc'], 
                                index=['d', 'a', 'b', 'c'])
    # print(df)
    # Notice side effect that index gets replaced by numbered index
    # print(df.append({'numbers': 100, 'floats': 5.75, 'names': 'Henry'}, ignore_index=True))
    df = df.append(pd.DataFrame({'numbers': 100, 'floats': 5.75, 'names': 'Henry'}, index=['z',]))
    # print(df)
    
    # Join method --> does SQL type, set theory combinations (right, left, inner, outer, etc.)
    print(df.join(pd.DataFrame([1, 4, 9, 16, 25], 
                        index=['a', 'b', 'c', 'd', 'y'],
                        columns=['squares',])))
    df =  df.join(pd.DataFrame([1, 4, 9, 16, 25], 
                        index=['a', 'b', 'c', 'd', 'y'],
                        columns=['squares',]), how='outer')
    print(df)
    print(df[['numbers', 'squares']].mean())
    print(df[['numbers', 'squares']].std())

def second_steps():
    a = np.random.standard_normal((9,4))
    a.round(6)
    df = pd.DataFrame(a)
    df.columns = [['No1', 'No2', 'No3', 'No4']]
    # print(df['No2'][3])
    dates = pd.date_range('2015-1-1', periods=9, freq='M')
    print(dates)

if __name__ == "__main__":
    # intro()
    second_steps()
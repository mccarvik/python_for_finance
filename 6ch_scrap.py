import numpy as np
import pandas as pd

def intro():
    df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
    print(df)
    

if __name__ == "__main__":
    intro()
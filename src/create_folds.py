"""Creates a K fold division of the data and puts tghe folder nr on each one"""
import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train_1.csv")
    df["Kfold"] = -1
    target = input("Target column?:")
    
    df = df.sample(frac = 1).reset_index(drop = True)
    # fetch targets
    y = df.loc[:, target].values
   
    kf = model_selection.StratifiedKFold(n_splits= 5)

    for fold, (trn_, val_) in enumerate(kf.split(X = df, y = y)):
        df.loc[val_, "Kfold"] = fold

    df.to_csv("../input/train_folds.csv", index  = False) 
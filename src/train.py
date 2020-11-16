import os
import config
import model_dispatcher
import argparse 

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 


def run(fold, model):
    #Takes a fold and separate/rotulating the train and validation set of that fold
    #read the data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    target = config.TARGET
    #identifying train and validation sets for that fold
    df_train = df[df["Kfold"] != fold].reset_index(drop = True)
    df_val = df[df["Kfold"] == fold].reset_index(drop = True)

    #Create a X_train and y_train set from df_train. These files must be arrays
    y_train =  df_train.loc[:, target].values
    X_train = df_train.drop(columns = target, axis = 1).values
   
    #Create a X_val and y_val set from df_train. These files must be arrays
    y_val = df_val.loc[:, target].values
    X_val = df_val.drop(columns = target, axis = 1).values
    #Instance of the algorithm
    clf =  tree.DecisionTreeClassifier()
    clf = model_dispatcher.models[model]

    #Fit the algorithm to the train set
    clf.fit(X_train, y_train)

    #Predic with validation
    y_pred = clf.predict(X_val)
    #Compute accuracy for the prediction
    accuracy = metrics.accuracy_score(y_val, y_pred)
    print(f"Fold = {fold}, Accuracy = {accuracy}")

    #Save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"../models/dt_{fold}.bin"))

if __name__ == "__main__":
    # target = input("Target location column?:")
    #initializing ArgumentParser of argparse class
    parser = argparse.ArgumentParser()
    #Add the arguments and the type of each
    parser.add_argument("--fold", type= int)
    parser.add_argument("--model", type= str)

    #Read the arguments from the commnand line
    args = parser.parse_args()
    #Run
    run(fold = args.fold, model = args.model)




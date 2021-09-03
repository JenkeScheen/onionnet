#!/usr/bin/env python

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawDescriptionHelpFormatter
import os
from scipy import stats
from sklearn.metrics import mean_absolute_error
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
alpha = 0.8

#################### stats.

def pearsonr(y_true, y_pred):
    r_val, _ = stats.pearsonr(y_true, y_pred)
    return r_val

def mue(y_true, y_pred):
    mue = mean_absolute_error(y_true, y_pred)
    return mue

def kendalltau(y_true, y_pred):
    tau, _ = stats.kendalltau(y_true, y_pred)
    return tau

#################### utils.

def pKaKcalMol(pka_val):
    """Convert pKa value to dG in kcal/mol. Provided pKa values are in nm."""
    
    kd_val = 10**(-pka_val) *10000
    R_Kcal = 1.987204258640
    
    dG = -R_Kcal * 300 * kd_val
    
    return dG

#################### model stuff.

def pcc(y_true, y_pred):
    p = stats.pearsonr(y_true, y_pred)
    return p[0]


def pcc_rmse(y_true, y_pred):

    dev = np.square(y_true.ravel() - y_pred.ravel())
    r = np.sqrt(np.sum(dev) / y_true.shape[0])

    p = stats.pearsonr(y_true, y_pred)[0]

    return (1 - p) * alpha + r * (1 - alpha)


def PCC_RMSE(y_true, y_pred):

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    r = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    p = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    # p = tf.where(tf.is_nan(p), 0.25, p)

    return alpha * p + (1 - alpha) * r


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)

def remove_all_hydrogens(dat, n_features):
    df = np.zeros((dat.shape[0], n_features))
    j = -1
    for f in dat.columns.values:
        # remove the hydrogen containing features
        if "H_" not in f and "_H_" not in f and int(f.split("_")[-1]) > 64:
            j += 1
            # if df.shape[0] == 0:
            try:
                df[:, j] = dat[f].values
            except IndexError:
                pass
            print(j, f)

    df = pd.DataFrame(df)
    df.index = dat.index

    return df


def create_model(input_size, lr=0.0001, maxpool=True, dropout=0.1):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=1,
                                     padding="valid", input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Conv2D(64, 4, 1, padding="valid"))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Conv2D(128, 4, 1, padding="valid"))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(400, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(200,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, )
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=['mse'])

    return model


if __name__ == "__main__":
    d = """Predict the features based on protein-ligand complexes.

    Citation: Zheng L, Fan J, Mu Y. arXiv preprint arXiv:1906.02418, 2019.
    Author: Liangzhen Zheng (zhenglz@outlook.com)

    Examples:
    python predict_pKa.py -fn features_ligands.csv -model ./models/OnionNet_HFree.model \
    -scaler models/StandardScaler.model -out results.csv


    -fn : containing the features, one sample per row with an ID, 2891 feature values.
    -model: the OnionNet CNN model containing the weights for the networks
    -scaler: the scaler for dataset standardization
    -out: the output pKa, one sample per row with two columns (ID and predicted pKa)

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-fn", type=str, default="features_1.csv",
                        help="Input. The docked cplx feature training set for pKa prediction.")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-weights", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    scaler = joblib.load(args.scaler)

    Xtest = None


    df = pd.read_csv(args.fn, index_col=0, header=0).dropna()

    if 'pKa' in df.columns:
        ytest = df['pKa'].values.ravel()
        df = df.drop(['pKa'], axis=1)
    else:
        ytest = None

    Xs = scaler.transform(df.values)
    Xs = pd.DataFrame(Xs)
    Xs.index = df.index
    Xs.columns = df.columns

    # the dataset shape 60 layers with 64 atom-type combinations
    Xtest = Xs.values.reshape((-1, 64, 60, 1))

    # create the model
    base_model = create_model((64, 60, 1), dropout=0.0, maxpool=False, lr=0.001)
    base_model.load_weights(args.weights)

    #print(base_model.summary())

    ############################################
    ######################## process predictions
    lig_names = [name.split("_")[1].split(".")[0] for name in Xs.index]
    ypred = pd.DataFrame(index=lig_names)

    ypred['iML ΔG'] = [pKaKcalMol(val) for val in base_model.predict(Xtest).ravel()]

    ################################################
    ######################## Load Expt./FEP measures
    ref_data = pd.read_csv("POC/input/results_20ns.csv")
    fep_preds = ref_data[["Ligand", "Pred. ΔG"]].set_index("Ligand")
    expt = ref_data[["Ligand", "Exp. ΔG"]].set_index("Ligand")

    compiled = ypred.join([fep_preds, expt])

    print("\niML vs Expt.:")
    print(f"R: {round(pearsonr(compiled['iML ΔG'], compiled['Exp. ΔG']), 2)}")
    print(f"MUE: {round(mue(compiled['iML ΔG'], compiled['Exp. ΔG']), 2)}")
    print(f"TAU: {round(kendalltau(compiled['iML ΔG'], compiled['Exp. ΔG']), 2)}")

    print("\nFEP vs Expt.:")
    print(f"R: {round(pearsonr(compiled['Pred. ΔG'], compiled['Exp. ΔG']), 2)}")
    print(f"MUE: {round(mue(compiled['Pred. ΔG'], compiled['Exp. ΔG']), 2)}")
    print(f"TAU: {round(kendalltau(compiled['Pred. ΔG'], compiled['Exp. ΔG']), 2)}")

    # reset index for easier merging later on.
    #compiled = compiled.reset_index().rename(columns={'index': 'Ligand'})
    ############################################
    ######################## Transfer learn in loop
    overall_results = []
    for repl in range(15):
        # do the whole thing n times to get variation per n sampled.
        print("@"*50, "REPLICATE", repl)

        this_rep_results = []
        # 42 ligands total, want to ALWAYS have 15 ligands in validation set.
        # i.e. loop up to 42 - 10 - 10 = 22.
        for data_selector_num in range(22):
            # increment n sampled.

            num = data_selector_num + 1
            print(f"\nTransfer-learning on {num} ligand(s).")

            ########################
            ###### DATA SAMPLING

            Xs = scaler.transform(df.values)
            Xs = pd.DataFrame(Xs)

            Xs_train = Xs.sample(n=num)
            train_indices = Xs_train.index

            # first grab all the non-training ligands.
            Xs_test_non_train = Xs[~Xs.isin(Xs_train)].dropna()

            # create a validation set. We will train on its FEP preds.
            Xs_val = Xs_test_non_train.sample(n=10)
            val_indices = Xs_val.index

            # randomly sample 10 of these to create a test set. Validates with expt.
            Xs_non_val = Xs_test_non_train[~Xs_test_non_train.isin(Xs_val)].dropna()
            Xs_test = Xs_non_val.sample(n=10)
            test_indices = Xs_test.index

            # get labels. We already have the correct order from earlier analysis.
            # we train on FEP values, and check how well the model predicts on exp values.
            yval_train = compiled.iloc[train_indices]['Pred. ΔG'].values
            yval_val = compiled.iloc[val_indices]['Pred. ΔG'].values
            yval_test = compiled.iloc[test_indices]['Exp. ΔG'].values

            # the dataset shape 60 layers with 64 atom-type combinations
            Xtrain = Xs_train.values.reshape((-1, 64, 60, 1))
            Xval = Xs_val.values.reshape((-1, 64, 60, 1))
            Xtest = Xs_test.values.reshape((-1, 64, 60, 1))

            ########################
            ###### MODEL GENERATION
            # add first extra dense layer by replacing the last regressor neuron in the base model.
            base_output = base_model.layers[-3].output
            new_output = tf.keras.layers.Dense(activation="relu", units=100)(base_output)
            extended_model_0 = tf.keras.models.Model(inputs=base_model.inputs, outputs=new_output)

            # add a second dense layer, slightly smaller.
            base_output = extended_model_0.layers[-1].output
            new_output = tf.keras.layers.Dense(activation="relu", units=70)(base_output)
            extended_model_1 = tf.keras.models.Model(inputs=extended_model_0.inputs, outputs=new_output)

            # add a third dense layer, slightly smaller.
            base_output = extended_model_1.layers[-1].output
            new_output = tf.keras.layers.Dense(activation="relu", units=40)(base_output)
            extended_model_2 = tf.keras.models.Model(inputs=extended_model_1.inputs, outputs=new_output)

            # add a final dense regressor neuron.
            base_output = extended_model_2.layers[-1].output
            new_output = tf.keras.layers.Dense(activation="linear", units=1)(base_output)
            extended_model = tf.keras.models.Model(inputs=extended_model_2.inputs, outputs=new_output)

            # freeze the layers up to the extension.
            # Check https://www.tensorflow.org/tutorials/images/transfer_learning
            # and https://stackoverflow.com/questions/57569460/freezing-keras-layer-doesnt-change-sumarry-trainable-params
            for layer in extended_model.layers[:-5]:
                layer.trainable = False

            # simple early stopping
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', 
                                verbose=1, patience=50)

            sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, clipnorm=1.0)
            extended_model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=['mae'])

            # now ready for transfer learning.
            ########################
            ###### MODEL FITTING
            # history = extended_model.fit(Xtrain, yval_train, validation_data=(Xval, yval_val),
            #                   batch_size=5, epochs=1000, verbose=0)
            # with early stopping:
            history = extended_model.fit(Xtrain, yval_train, validation_data=(Xval, yval_val),
                              batch_size=5, epochs=1000, verbose=0, callbacks=[es])

            # at this point we could fine-tune the frozen layers to get some more
            # predictivity, but overkill for this POC.
            # instead, just predict on the test set.
            test_preds = extended_model.predict(Xtest).ravel()
            if not any(np.isnan(test_preds)):
                test_mue = round(mue(test_preds, yval_test), 3)
            else:
                test_mue = 'nan'
            ########################
            ###### DATA PROCESSING
            mues = pd.DataFrame(history.history)['val_mean_absolute_error'].values
            min_mue= round(min(mues), 3)
            print("Last val. MUE:", min_mue)
            print("Test MUE:", test_mue)

            this_rep_results.append([num, min_mue])

        overall_results.append([repl, this_rep_results])

with open("POC_PRED_MUES.csv", "w") as mue_file:
    writer = csv.writer(mue_file)
    writer.writerow(["Replicate", "Num_trained", "MUE"])
    for entry in overall_results:
        repl = entry[0]
        for data in entry[1]:
            writer.writerow([repl, data[0], data[1]])
print("DONE")







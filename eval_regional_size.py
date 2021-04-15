import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import bartlett
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


radiomics_path = './results/radiomics/radiomics.csv'
name_mapping_path = './data/name_mapping.csv'


if __name__ == '__main__':
    mapping = pd.read_csv(name_mapping_path, header=0)[['Grade', 'BraTS_2019_subject_ID']]
    radiomics = pd.read_csv(radiomics_path, header=0, index_col=0)

    df = radiomics.merge(mapping, left_index=True, right_on='BraTS_2019_subject_ID')

    y = np.array(df['Grade'])
    X = np.array(df.loc[:, ['NET_VoxelVolume', 'ED_VoxelVolume', 'ET_VoxelVolume']])

    random_state = 1173

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    accuracy_list = []
    precision_list = []
    sensitivity_list = []
    specificity_list = []
    npv_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = LogisticRegression(random_state=random_state).fit(X_train, y_train)

        y_pred  = clf.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        npv_list.append(npv)

    result = {
        'accuracy_mean': [np.mean(accuracy_list)],
        'accuracy_std': [np.std(accuracy_list)],
        'precision_mean': [np.mean(precision_list)],
        'precision_std': [np.std(precision_list)],
        'sensitivity_mean': [np.mean(sensitivity_list)],
        'sensitivity_std': [np.std(sensitivity_list)],
        'specificity_mean': [np.mean(specificity_list)],
        'specificity_std': [np.std(specificity_list)],
        'npv_mean': [np.mean(npv_list)],
        'npv_std': [np.std(npv_list)],
    }

    df = pd.DataFrame(data=result)
    df.to_csv('./results/radiomics/classification_by_regional_size.csv')

import os
import torch
import random
import radiomics
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from utils import load_json
from infer_features import save_dir_path


name_mapping_path = './data/name_mapping.csv'


def eval_feature_difference(base_features, scale, shift):
    base_data = base_features.values

    comp_features_path = os.path.join(
        save_dir_path,
        'features_scale_{:.1f}_shift_{:.1f}.csv'.format(scale, shift),
    )
    comp_features = pd.read_csv(comp_features_path, index_col=0, header=0)
    comp_data = comp_features.values

    diff = np.abs(base_data - comp_data).sum()
    denominator = base_data.sum()

    ratio = diff / denominator

    return ratio


def eval_classification_performance(scale, shift, random_state):
    comp_features_path = os.path.join(
        save_dir_path,
        'features_scale_{:.1f}_shift_{:.1f}.csv'.format(scale, shift),
    )
    comp_features = pd.read_csv(comp_features_path, index_col=0, header=0)

    mapping = pd.read_csv(name_mapping_path, header=0)[['Grade', 'BraTS_2019_subject_ID']]

    df = comp_features.merge(mapping, left_index=True, right_on='BraTS_2019_subject_ID')

    y = np.array(df['Grade'])
    X = np.array(df.loc[:, '0': '511'])

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
        'accuracy_mean': np.mean(accuracy_list),
        'accuracy_std': np.std(accuracy_list),
        'precision_mean': np.mean(precision_list),
        'precision_std': np.std(precision_list),
        'sensitivity_mean': np.mean(sensitivity_list),
        'sensitivity_std': np.std(sensitivity_list),
        'specificity_mean': np.mean(specificity_list),
        'specificity_std': np.std(specificity_list),
        'npv_mean': np.mean(npv_list),
        'npv_std': np.std(npv_list),
    }

    return result


if __name__ == '__main__':

    base_features_path = os.path.join(
        save_dir_path, 'features_scale_0.0_shift_0.0.csv'
    )
    base_features = pd.read_csv(base_features_path, index_col=0, header=0)

    results = []
    shift = 0.0
    for scale in np.arange(0.0, 1.1, 0.1):
        feat_diff = eval_feature_difference(base_features, scale, shift)

        result = {
            'shift': shift,
            'scale': scale,
            'ratio': feat_diff,
        }

        class_performance = eval_classification_performance(scale, shift, random_state=1173)

        result.update(class_performance)
        results.append(result)

    scale = 0.0
    for shift in np.arange(0.0, 1.1, 0.1):
        feat_diff = eval_feature_difference(base_features, scale, shift)

        result = {
            'shift': shift,
            'scale': scale,
            'ratio': feat_diff,
        }

        class_performance = eval_classification_performance(scale, shift, random_state=1173)

        result.update(class_performance)
        results.append(result)

    df = pd.DataFrame(data=results)
    df.to_csv(os.path.join(
        save_dir_path, 'features_reproducibility.csv',
    ))

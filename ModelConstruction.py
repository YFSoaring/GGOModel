import os
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from scipy.stats import levene, mannwhitneyu, pearsonr, ttest_ind

warnings.filterwarnings("ignore")


def LassoSelect(file1, label_path1, file3, label_path3, seed=9):
    np.random.seed(seed)
    random.seed(seed)

    # ========================== Process Training and Internal Test Sets ==========================
    pdfile1 = pd.read_csv(file1)
    labelData = pd.read_excel(label_path1)

    pdfile1['ID'] = pdfile1['ID'].astype(str).str.strip()
    labelData['ID'] = labelData['ID'].astype(str).str.strip()
    label_dict1 = dict(zip(labelData['ID'], labelData['Label']))
    pdfile1.insert(1, 'label', pdfile1['ID'].map(label_dict1).astype(int))

    data_all = pdfile1.set_index('ID')

    # Stratified split to preserve class proportions between train and test sets
    train_df, test_df = train_test_split(
        data_all,
        test_size=0.3,
        random_state=seed,
        stratify=data_all['label'],
    )

    xtrain = train_df.drop(columns=['label']).astype(np.float64)
    ytrain = train_df['label']

    xtest = test_df.drop(columns=['label']).astype(np.float64)
    ytest = test_df['label']

    # ========================== Process External Validation Set ==========================
    pdfile3 = pd.read_csv(file3)
    labelData3 = pd.read_excel(label_path3, sheet_name='Extra')
    pdfile3['ID'] = pdfile3['ID'].astype(str).str.strip()
    labelData3['ID'] = labelData3['ID'].astype(str).str.strip()

    labelData3['Label'] = labelData3['Label'].astype(int)
    label_dict3 = dict(zip(labelData3['ID'], labelData3['Label']))

    pdfile3.insert(
        1, 'label', pdfile3['ID'].map(label_dict3).astype(int)
    )
    data_test3 = pdfile3.set_index('ID')

    xtest3 = data_test3.drop(columns=['label']).astype(np.float64)
    ytest3 = data_test3['label']

    # ========================== Feature Selection & Standardization ==========================
    feature_path = '/FinalFeature-TP5.txt'
    if not os.path.exists(feature_path):
        print(f"Error: Feature file not found at {feature_path}")
        return

    colum_lasso = np.atleast_1d(np.loadtxt(feature_path, dtype='str'))

    if len(colum_lasso) > 0:
        print(
            '===================================== START ========================================================'
        )
        print(f'Selected Feature Count: {len(colum_lasso)}')
        print(colum_lasso)
        print(
            '====================================================================================================='
        )

        xtrain = xtrain[colum_lasso]
        xtest = xtest[colum_lasso]
        xtest3 = xtest3[colum_lasso]

        # Z-score standardization
        zscore = StandardScaler()
        xtrain = pd.DataFrame(
            zscore.fit_transform(xtrain),
            columns=colum_lasso,
            index=xtrain.index,
        )
        xtest = pd.DataFrame(
            zscore.transform(xtest), columns=colum_lasso, index=xtest.index
        )
        xtest3 = pd.DataFrame(
            zscore.transform(xtest3), columns=colum_lasso, index=xtest3.index
        )

        # ========================== Model Training & Feature Importance ==========================
        model_ET = ExtraTreesClassifier(
            n_estimators=33, max_depth=6, random_state=seed
        ).fit(xtrain, ytrain)

        all_feature_importances = np.array(
            [tree.feature_importances_ for tree in model_ET.estimators_]
        )
        feature_importance_std = np.std(all_feature_importances, axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': xtrain.columns,
            'Importance': model_ET.feature_importances_,
            'Importance Std': feature_importance_std,
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance Ranking:")
        print(feature_importance_df)

        # ========================== Model Prediction & Threshold Tuning ==========================
        train_scoreET = model_ET.predict_proba(xtrain)[:, 1]
        test_scoreET = model_ET.predict_proba(xtest)[:, 1]
        test3_scoreET = model_ET.predict_proba(xtest3)[:, 1]

        # Determine optimal cutoff threshold based on training set using Youden's Index
        fpr, tpr, thresholds = roc_curve(ytrain, train_scoreET)
        youden_index = tpr - fpr
        best_cutoff = thresholds[np.argmax(youden_index)]
        print(f"\nOptimal Cutoff Threshold: {best_cutoff:.4f}")

        new_train_predET = (train_scoreET >= best_cutoff).astype(int)
        new_test_predET = (test_scoreET >= best_cutoff).astype(int)
        new_test3_predET = (test3_scoreET >= best_cutoff).astype(int)

        # Helper function for metric evaluation
        def get_metrics(y_true, y_pred, y_score):
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score)
            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return auc, acc, sen, spec

        train_auc, train_acc, train_sen, train_spec = get_metrics(
            ytrain, new_train_predET, train_scoreET
        )
        test_auc, test_acc, test_sen, test_spec = get_metrics(
            ytest, new_test_predET, test_scoreET
        )
        test3_auc, test3_acc, test3_sen, test3_spec = get_metrics(
            ytest3, new_test3_predET, test3_scoreET
        )

        print(
            '\n******************************** ET Model Performance **********************************'
        )
        print(f'Random Seed: {seed}')
        print(
            f'Train Set     -> AUC: {train_auc:.4f}, ACC: {train_acc:.4f}, SEN: {train_sen:.4f}, SPEC: {train_spec:.4f}'
        )
        print(
            f'Test Set      -> AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, SEN: {test_sen:.4f}, SPEC: {test_spec:.4f}'
        )
        print(
            f'ExtraTest Set -> AUC: {test3_auc:.4f}, ACC: {test3_acc:.4f}, SEN: {test3_sen:.4f}, SPEC: {test3_spec:.4f}'
        )


if __name__ == '__main__':
    file1 = '/Original_TP5_feature.csv'
    label_path1 = '/Original_TP5_label.xlsx'

    file3 = '/Extra_TP5_feature.csv'
    label_path3 = '/Extra_TP5_label.xlsx'

    LassoSelect(file1, label_path1, file3, label_path3, seed=9)
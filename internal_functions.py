import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from typing import List, Optional, Union
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    f1_score)


def percentage_of_nulls(dataset: pd.DataFrame) -> pd.Series:
    """ Return percentage of missing data in features """
    return dataset.isnull().sum() * 100 / len(dataset)


def display_formatted_features_missing_pct(dataset: pd.DataFrame):
    """ Print formatted percents of missing data"""
    feat_pct_dict = dict(percentage_of_nulls(dataset))
    longest_key = len(max(feat_pct_dict, key=len))
    for i, j in feat_pct_dict.items():
        print(f'Feature: {i:{longest_key}}  Percent of missing data: {j}')


def features_with_threshold(dataset: pd.DataFrame, threshold: int) -> List:
    """ Return list of features that match the threshold"""
    feat_pct_dict = dict(percentage_of_nulls(dataset))
    features_list = []
    for key, value in feat_pct_dict.items():
        if value >= threshold:
            features_list.append(key)
    return features_list


def dataset_shrink(original_df: pd.DataFrame, modified_df: pd.DataFrame) -> str:
    """ Return dataset shrink from original in percents"""
    changed_pct = modified_df.shape[0] / original_df.shape[0] * 100
    return f'{round(100-changed_pct, 2)}%'


def title_and_labels(title: str, x_label: str, y_label: str):
    """ Defining fontsize for plots """
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)


def change_type_to_numeric(dataset: pd.DataFrame, features: List):
    """ Change features dtype to numeric"""
    for feat in features:
        dataset[feat] = pd.to_numeric(dataset[feat])


def change_type_to_categorical(dataset: pd.DataFrame, features: List):
    """ Change features dtype to category"""
    for feat in features:
        dataset[feat] = dataset[feat].astype('category')


def change_type_to_uint8(dataset: pd.DataFrame, features: List):
    """ Change features dtype to uint8"""
    for feat in features:
        dataset[feat] = dataset[feat].astype('uint8')


def get_shapes(list_of_dfs: List, names: List) -> List:
    """ Return list of shapes of datasets"""
    shapes = []
    for i, name in zip(list_of_dfs, names):
        shapes.append([name[:-4], i.shape[0], i.shape[1]])
    return shapes


def get_features_of_zeros_ones(dataset: pd.DataFrame, columns: List) -> List:
    """ Return list of features that values are 0 or 1 (categorical)"""
    categorical_features = []
    for column in columns:
        if dataset[column].isin([0,1]).all():
            categorical_features.append(column)
    return categorical_features


def num_of_outliers(dataset: pd.DataFrame, features: List, limit: int) -> List:
    """ Return list of rows indexes with at least number (limit) of outliers detected """
    outliers = []
    for col in features:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers.extend(dataset[(dataset[col] < (Q1 - 1.5 * IQR)) | (dataset[col] > (Q3 + 1.5 * IQR))].index)
    outliers = Counter(outliers)
    filter_outliers = [index for index, count in outliers.items() if count >= limit]
    return filter_outliers


def fraction_of_group_to_target(dataset: pd.DataFrame, feature: str, rotation: Optional[int] = None):
    """ Count size of group and fraction of True values. Make barplots of group size and size of fraction in %"""
    f, ax = plt.subplots(1, 2, figsize=(16, 6))

    grouped_df = dataset.groupby([feature]).agg(
        number_of_clients=(feature, 'count'),
        loans_defaulted=('TARGET', 'sum')
    )
    grouped_df['defaulted_loans_fraction'] = grouped_df['loans_defaulted'] / grouped_df['number_of_clients'] * 100
    grouped_df.drop('loans_defaulted', axis=1, inplace=True)

    features = grouped_df.columns

    for feat, ax in zip(features, ax.flat):
        sns.barplot(y=feat, x=grouped_df.index, data=grouped_df, ax=ax)
        if grouped_df[feat].dtypes == 'float64':
            ax.bar_label(ax.containers[0], fmt="%0.2f%%", fontsize=14)
        else:
            ax.bar_label(ax.containers[0], fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, fontsize=12)
        ax.set_title(' '.join(feat.split('_')).capitalize(), fontsize=14)

    plt.tight_layout()


def make_label(row: Union[str, int, float], one_value: str, zero_value: str) -> str:
    """ Return labeled value of binary category"""
    if type(row) == float and row > 0:
        return one_value
    elif row == 1 or row == 'M':
        return one_value
    else:
        return zero_value


def rename_columns(dataset: pd.DataFrame, prefix: str) -> List:
    """ Return list of renamed aggregated columns"""
    new_columns = ['SK_ID_CURR']
    for name in dataset.columns.levels[0][:-1]:
        new_columns.append(f'{prefix}_{name}_MEAN')
    return new_columns


def rename_columns_encoding(dataset: pd.DataFrame, prefix: str) -> List:
    """ Return list of renamed encoded and aggregated columns"""
    new_columns = ['SK_ID_CURR']
    for name in dataset.columns.levels[0][:-1]:
        new_columns.append(f'{prefix}_{name}_SUM')
    return new_columns


def dummies(dataset: pd.DataFrame, id_feature) -> pd.DataFrame:
    """ Return One-Hot encoded dataset with id_feature included"""
    encoded_dataset = pd.get_dummies(dataset.select_dtypes('object'))
    encoded_dataset[id_feature] = dataset[id_feature]
    return encoded_dataset


def log_transformation(dataset: pd.DataFrame, columns) -> pd.DataFrame:
    """ Return dataset with log transformation"""
    modified_df = dataset.copy()
    for column in columns:
        modified_df[f'{column}'] = np.log(1 + dataset[column])
    return modified_df


def feature_importance_iter(dataset: pd.DataFrame, target: str, id_feat: str, max_iter: int) -> pd.DataFrame:
    """ Return dataset with removed features of zero importance by LightGBM model

        Runs iterations until max_iter is reached or no features with zero importance are found """

    for iteration in range(1, max_iter + 1):
        y = dataset[target]
        X = dataset.drop([target, id_feat], axis=1)

        lgbm_model = lgb.LGBMClassifier(
            n_estimators=5000,
            objective='binary',
            n_jobs=-1,
            boosting_type='goss',
            class_weight='balanced')

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        lgbm_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            early_stopping_rounds=800,
            verbose=100)

        # getting names and importances and putting to dataframe
        features = lgbm_model.feature_name_
        importances = lgbm_model.feature_importances_
        feature_importance = pd.DataFrame(
            {'features': features, 'importance': importances}).sort_values('importance', ascending=False).reset_index(
            drop=True)
        zero_feats = feature_importance[feature_importance['importance'] == 0]['features'].to_list()
        if len(zero_feats) > 0:
            print(f'Iteration {iteration}: There are {len(zero_feats)} features with zero importance!')
            dataset.drop(zero_feats, axis=1, inplace=True)
        else:
            print(f'There is no features with zero importance anymore! It took {iteration} iterations')
            print(f'Final shape is: {dataset.shape}')
            break
    return dataset


def collinear_features(dataset: pd.DataFrame, threshold: float):
    """ Return list of features with correlation exceeded above threshold"""

    # Absolute values to find positive and negative correlations above threshold
    corr = dataset.corr().abs()

    # Copy of an array with the elements below the k-th diagonal zeroed
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.triu.html
    # Source: https://cmdlinetips.com/2020/02/lower-triangle-correlation-heatmap-python/
    correlations_df = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    collinear_above_threshold = [column for column in correlations_df.columns if
                                 any(correlations_df[column] > threshold)]

    print(f'There are {len(collinear_above_threshold)} features with correlations above threshold')

    return collinear_above_threshold


def prediction_print(classifier, predictions, predictions_prob, y_test):
    """ Prints metrics of ML model"""

    accuracy = accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
    auc_score = auc(fpr, tpr)
    class_report = classification_report(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(classifier.__class__.__name__)
    print('\n')
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc_score}')
    print(f'F1: {f1}')
    print(class_report)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues')
    plt.show()
    print(100 * '-')


def run_model(classifier, x_tr: pd.DataFrame, x_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series):
    """ Return trained model"""
    results = {}
    model = classifier
    model.fit(
        x_tr,
        y_tr,
    )
    predictions = model.predict(x_te)
    predictions_prob = model.predict_proba(x_te)[:,1]
    prediction_print(classifier, predictions, predictions_prob, y_te)
    results[classifier.__class__.__name__] = model.score(x_te, y_te)
    return model, results


def roc_auc_plot(models, X_test, y_test, optional_model, x_std, y_std):
    # Create the baseline (how the graph looks like if the model does random guess instead)
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)

    plt.figure(figsize=(12, 8))
    plt.title('ROC Curve')
    plt.plot(p_fpr, p_tpr)

    for model in models:
        # Predict the probability score
        prob_test = model.predict_proba(X_test)

        # Create test curve
        fpr_test, tpr_test, thresh_test = roc_curve(y_test, prob_test[:, 1])

        plt.plot(fpr_test, tpr_test, label=model.__class__.__name__)

    prob_test = optional_model.predict_proba(x_std)
    fpr_test, tpr_test, thresh_test = roc_curve(y_std, prob_test[:, 1])
    plt.plot(fpr_test, tpr_test, label = optional_model.__class__.__name__)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()


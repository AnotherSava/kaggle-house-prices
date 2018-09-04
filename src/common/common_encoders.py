import logging
from os import makedirs
from os.path import join, isdir, exists

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from common import make_dir
from component_classifier import DEFAULT_PRODUCT

ABSENT_VALUE = 'None'


def create_column_names(name, number, separator='_'):
    chars = len(str(number))
    return ['{}{}{:0{}d}'.format(name, separator, i, chars) for i in range(number)]


def encode_cat_feature(df, label, product=None, load_model=False, save_model=False, encoders_folder=None, verbose=False):
    if verbose:
        print("Category '{}'".format(label), end='')

    series = df[label].copy()

    if load_model:
        label_encoder = load_label_encoder(product, label, encoders_folder)
        series[~series.isin(set(label_encoder.classes_))] = ABSENT_VALUE
    else:
        label_encoder = LabelEncoder()

        # Removing labels presented only once to compact model
        counts = series.value_counts()
        exclude = set(counts[counts == 1].index)
        series[series.isin(exclude)] = ABSENT_VALUE

        # Making sure NaN is encoded
        label_encoder.fit([ABSENT_VALUE] + series.tolist())

    # First label is extra NaN
    labeled_feature = label_encoder.transform([ABSENT_VALUE] + series.tolist()).reshape(len(series) + 1, 1)

    if load_model:
        onehot_encoder = load_one_hot_encoder(product, label, encoders_folder)
    else:
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(labeled_feature)

    # Removing encoded extra NaN record and whole column related to NaN labels
    features = onehot_encoder.transform(labeled_feature)[1:,1:]

    if verbose:
        print(": {} items".format(features.shape[1]))

    if save_model:
        save_label_encoder(product, label, label_encoder, encoders_folder)
        save_one_hot_encoder(product, label, onehot_encoder, encoders_folder)

    columns = create_column_names(label, features.shape[1])
    return pd.DataFrame(features, index=df.index, columns=columns).astype(np.int)


def encode_cat_features(df, labels, product=None, load_model=False, save_model=False, encoders_folder=None, verbose=False):
    feature_dfs = [encode_cat_feature(df, label, product=product, load_model=load_model, save_model=save_model, encoders_folder=encoders_folder, verbose=verbose) for label in labels]
    return pd.concat(feature_dfs, axis=1) 


def get_feature_prefix(feature):
    return feature.split('_')[0]


def list_features(df):
    return set([get_feature_prefix(feature) for feature in df.columns.tolist()])


def collect_feature_flags(features, name):
    return [get_feature_prefix(feature) == name for feature in features]


def collect_features(features, names):
    name_list = [names] if isinstance(names, str) else names
    return [feature for feature in features if get_feature_prefix(feature) in name_list]


def create_label_encoder_name(feature):
    return 'label-encoder-{}.txt'.format(feature)


def load_label_encoder(product, feature, encoders_folder):
    if not product:
        product = DEFAULT_PRODUCT
    input_file_name = join(encoders_folder, product, create_label_encoder_name(feature))
    if not exists(input_file_name):
        logging.error('Label encoder not found: "{}"'.format(input_file_name))
        return None
    label_encoder = LabelEncoder()
    label_encoder.classes_ = joblib.load(input_file_name)
    return label_encoder


def save_label_encoder(product, feature, encoder, encoders_folder):
    make_dir(encoders_folder)

    if not product:
        product = DEFAULT_PRODUCT
    folder = join(encoders_folder, product)
    if not isdir(folder):
        makedirs(folder)

    joblib.dump(encoder.classes_, join(folder, create_label_encoder_name(feature)))


def create_one_hot_encoder_name(feature):
    return 'one-hot-encoder-{}.txt'.format(feature)


def load_one_hot_encoder(product, feature, encoders_folder):
    if not product:
        product = DEFAULT_PRODUCT
    input_file_name = join(encoders_folder, product, create_one_hot_encoder_name(feature))
    if not exists(input_file_name):
        return None
    return joblib.load(input_file_name)


def save_one_hot_encoder(product, feature, encoder, folder):
    make_dir(folder)

    if not product:
        product = DEFAULT_PRODUCT
    folder = join(folder, product)
    if not isdir(folder):
        makedirs(folder)
    joblib.dump(encoder, join(folder, create_one_hot_encoder_name(feature)))

    
def create_fasttext_embeddings(df, label, model):
    data = [model.get_sentence_vector(str(s)) for s in df[label]]
    columns = create_column_names(label, np.shape(data)[1])
    return pd.DataFrame(data, columns=columns, index=df.index)    


def create_fasttext_embeddings_single(text, label, model):
    vector = model.get_sentence_vector(text)
    columns = create_column_names(label, len(vector))
    return pd.DataFrame([vector], columns=columns)


def collect_xgboost_features_importance(df, model):
    column_list = df.columns
    unique_features = list(set([get_feature_prefix(x) for x in column_list]))
    feature_groups = [collect_feature_flags(column_list, feature) for feature in unique_features]
    number = [np.sum(feature_group) for feature_group in feature_groups]
    importance = [model.feature_importances_[feature_group].sum() for feature_group in feature_groups]
    result = pd.DataFrame({'number': number, 'importance': importance}, index=unique_features)
    return result.sort_values(by='importance', ascending=False)

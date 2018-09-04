
import re
import string
from os.path import join

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

MODEL_SUFFIX = '-model.bin'
XGB_MODEL_SUFFIX = '-joblib.dat'
LABEL_PREFIX = '__label__'
DEFAULT_PRODUCT = 'default'
PRODUCT_LABEL = 'Product'

DEFAULT_LEARNING_FOLDER = '../default/learning'

part = {
    'N': 'n',
    'V': 'v',
    'J': 'a',
    'S': 's',
    'R': 'r'
}

wnl = WordNetLemmatizer()


def convert_tag(penn_tag):
    if penn_tag[0] in part.keys():
        return part[penn_tag[0]]
    
    # other parts of speech will be tagged as nouns
    return 'n'


def lemmatize(text):
    sent = [tag for sent in sent_tokenize(text) for tag in pos_tag(word_tokenize(sent))]
    return ' '.join([wnl.lemmatize(tag[0], convert_tag(tag[1])) for tag in sent])


def preprocess(s):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    s = s.translate(table).lower().replace('\n', ' ')
    return re.sub(' +', ' ', s).strip()


def get_minor_products(df, minor_product_size):
    product_sizes = df[PRODUCT_LABEL].value_counts()
    return product_sizes[product_sizes < minor_product_size].index.tolist()

    
def filter_product(product, df, minor_product_size):
    if product == DEFAULT_PRODUCT:
        return df.drop(df[~df[PRODUCT_LABEL].isin(get_minor_products(df, minor_product_size))].index)
    
    return df.drop(df[df[PRODUCT_LABEL] != product].index)


def create_train_filename(product=DEFAULT_PRODUCT, folder=None):
    return join(folder if folder is not None else DEFAULT_LEARNING_FOLDER, product + '-data-train.csv')


def create_test_filename(product=DEFAULT_PRODUCT, folder=None):
    return join(folder if folder is not None else DEFAULT_LEARNING_FOLDER, product + '-data-test.csv')


def create_model_filename(folder, product=DEFAULT_PRODUCT):
    return join(folder, product + MODEL_SUFFIX)


def create_custom_model_filename(folder, name, product=DEFAULT_PRODUCT):
    return join(folder, product + '-' + name + MODEL_SUFFIX)


def create_xgb_model_filename(folder, product=DEFAULT_PRODUCT):
    return join(folder, product + XGB_MODEL_SUFFIX)


def insert_period(template, period=None):
    suffix = str(int(period.total_seconds())) if period is not None else 'default'
    return template.format(suffix)

import errno
import os
import pandas as pd
import numpy as np

LONG_SEPARATOR = '\n' + '-' * 120 + '\n'


def search_null(df, columns=None):
    columns_to_search = df.columns if columns is None else [columns] if isinstance(columns, str) else columns
    return df[df[columns_to_search].isnull().T.any().T]


def describe_null(df):
    null_value_stats = df.isnull().sum(axis=0)
    return null_value_stats[null_value_stats != 0]


def printt(text, tabs=1, **kwargs):
    tabs_string = '\t' * tabs
    for s in str(text).split('\n'):
        if len(s) > 0:
            print('{}{}'.format(tabs_string, s), **kwargs)


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = {key: d1[key] for key in d1_keys - d2_keys}
    removed = {key: d2[key] for key in d2_keys - d1_keys}
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def print_list(lst, separator='\n', prefix="'", suffix="'"):
    print(separator.join(map(lambda s: "{}{}{}".format(prefix, s, suffix), lst)))


def pprint_timedelta(period):
    seconds = period.seconds
    days = period.days
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '{:3d}d{:3d}h{:3d}m{:3d}s'.format(days, hours, minutes, seconds)
    elif hours > 0:
        return '{:3d}h{:3d}m{:3d}s'.format(hours, minutes, seconds)
    elif minutes > 0:
        return '{:3d}m{:3d}s'.format(minutes, seconds)
    else:
        return '{:3d}s'.format(seconds)


def get_at(lst, index=0, default=None):
    return lst[index] if max(~index, index) < len(lst) else default


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def split_dataframe(df, target_column, separator, strip=True):
    """
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """
    def split_list_to_rows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        if strip:
            split_row = map(str.strip, split_row)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def strip(l):
    return list(map(str.strip, l))


def enquote(s):
    return "'{}'".format(s)


def enlist(l):
    def prepare(item):
        if isinstance(item, str):
            return enquote(item)
        return str(item)

    return ', '.join(map(prepare, l))


def intersect(start1, end1, start2, end2):
    """
    start1, end1 = first interval
    start2, end2 = second interval
    open intervals can use None as f bound
    intersection is these intervals is returned
    """
    start = compare_none(start1, start2, np.maximum)
    end = compare_none(end1, end2, np.minimum)
    return (start, end) if start is None or end is None or start < end else None


def intersect_lists(*intervals_list):
    """
    intersect any number of interval lists
    """
    result = [(None, None)]
    for intervals in intervals_list:
        result = [intersect(i1[0], i1[1], i2[0], i2[1]) for i1 in result for i2 in intervals]
        result = filter_nones(result)
    return result


def touch(intervals, interval):
    """
    :param intervals: list of intervals to filter
    :param interval: interval that should touch them
    :return: subset of intervals that touch specific interval
    """
    return [i for i in intervals if intersect(i[0], i[1], interval[0], interval[1]) is not None]


def compare_none(a, b, f):
    """
    :param f: function to use for comparison
    :return: if one argument is None, return another, otherwise use f to compare
    """
    return a if b is None else b if a is None else f(a, b)


def filter_nones(lst):
    return [elem for elem in lst if elem is not None]


def dict_add_if_not_none(d, name, value):
    if value is not None:
        d[name] = value


def interval_length(interval):
    return interval[1] - interval[0]


def intervals_length(intervals):
    return sum(map(interval_length, intervals), pd.Timedelta('0 sec'))


def find_long_interval(intervals, length):
    return [interval for interval in intervals if (interval[1] - interval[0]) >= length]


def format_interval(interval):
    return '[{}, {}]'.format(format_timestamp(interval[0]), format_timestamp(interval[1]))


def format_intervals(intervals):
    return ', '.join(map(format_interval, intervals))


def format_timestamp(timestamp):
    return str(timestamp)


def format_timedelta(timedelta):
    return str(timedelta)


def series_not_nan(series):
    series = series.dropna()
    return None if series.empty else series.iloc[0]


def df_not_nan(df):
    return df.apply(series_not_nan)


# Root Mean Squared Logarithmic Error
def rmsle(y_test, y_pred):
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))

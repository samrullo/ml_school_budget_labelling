import pandas as pd


def combine_text_data(df: pd.DataFrame, to_drop=None):
    """

    :param df:
    :param to_drop:
    :return:
    """
    to_drop = set(to_drop) & set(df.columns.tolist())
    text_df = df.drop(to_drop, axis=1)
    text_df.fillna("", inplace=True)
    return text_df.apply(lambda x: " ".join(x), axis=1)

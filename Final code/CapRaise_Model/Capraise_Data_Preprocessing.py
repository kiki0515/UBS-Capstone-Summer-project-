"""data_prep.py: Make a labeled dataset."""
__author__ = "Hubert Wang"
__version__ = "0.1.1"

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO
import random
# Define Input and output folders
IN_DIR = "Input/"
OUT_DIR = "Output/"


def run_data_preprocessing() -> None:
    # Read original data
    pqt_1 = IN_DIR + r"model-crs-gecr-2022-02-03-train.parquet"
    df_1 = pd.read_parquet(pqt_1)

    # Creating lists for unique domain names and dates
    domain_list = list(set(df_1["domain_name"]))
    domain_list.sort()
    n_domain = len(domain_list)
    topic_list = df_1.groupby(['topic']).count().index.to_list()
    print("Topics:", topic_list)
    n_topic = len(topic_list)
    # Calculate relative gap to event date
    df_1['gap'] = (df_1['date_event'] - df_1['date_month']).astype('timedelta64[D]') / 30
    _cols = df_1.columns.tolist()
    _cols = _cols[:4] + _cols[-1:] + _cols[4:-1]
    df_1 = df_1[_cols]
    # Creating an empty dataframe
    df_trans = pd.DataFrame(np.zeros((n_domain, 3 * n_topic + 2), dtype=float))
    column_list = ['gap_' + str(s) for s in topic_list] + ['score_' + str(s) for s in topic_list] \
                  + ['rating_' + str(s) for s in topic_list] + ['label', 'event_date']
    df_trans.columns = column_list
    df_trans.index = domain_list
    # Fill the DataFrame
    df_empty = pd.DataFrame(topic_list).set_index(0)
    valid_list = []
    for i in range(n_domain):
        domain_i = domain_list[i]
        df_i = df_1.loc[df_1["domain_name"] == domain_i]
        label_i = df_i.iloc[0, -2]
        date_end = df_i.iloc[0, -3]
        # Filter by event_date
        date_start = date_end + relativedelta(months=-6)
        str_end = date_end.strftime('%Y%m')
        str_start = date_start.strftime('%Y%m')
        df_i = df_i.loc[df_i['dt'] > str_start]
        df_i = df_i.loc[df_i['dt'] < str_end]
        if len(df_i) > 0:
            valid_list.append(i)
        # Part 1: Gap
        df_c1 = df_i.groupby(['topic']).mean()['gap']
        df_c1 = df_empty.join(df_c1)
        df_c1 = df_c1.fillna(6)
        # Part 2: CapRaise Score
        df_c2 = df_i.groupby(['topic']).mean()['score']
        df_c2 = df_empty.join(df_c2)
        df_c2 = df_c2.fillna(70)
        # Part 3: Ratings
        df_c3 = df_i.groupby(['topic']).mean()['scoreq']
        df_c3 = df_empty.join(df_c3)
        df_c3 = df_c3.fillna(1)
        # Add to the big Dataframe
        df_trans.iloc[i, 0:n_topic] = df_c1.transpose()
        df_trans.iloc[i, n_topic:2*n_topic] = df_c2.transpose()
        df_trans.iloc[i, 2*n_topic:3*n_topic] = df_c3.transpose()
        df_trans.iloc[i, 3*n_topic:4*n_topic] = label_i

    df_trans = df_trans.iloc[valid_list, :]
    print("Valid entries after Processing:", len(df_trans))
    df_trans = df_trans.drop(['event_date'], axis=1)
    for i in range(len(df_trans)):
        if df_trans.iloc[i, -1] == 0:
            for j in range(6):
                drift = np.random.uniform(-0.25, 0.25)
                if df_trans.iloc[i, j] != 6:
                    df_trans.iloc[i, j] = df_trans.iloc[i, j] + drift
    df_trans.to_csv(OUT_DIR + 'preprocessed_data.csv')
    print("Successfully saved to: " + OUT_DIR)


if __name__ == "__main__":
    run_data_preprocessing()









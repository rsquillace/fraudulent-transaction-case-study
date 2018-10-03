import pandas as pd
import numpy as np
import sqlite3

'''
This script joins the transaction and ip dataframes to assign a country to each recorded transaction. About 14.5% of transactions have an ip address that doesn't fit within any of the ranges provided in the ip dataframe. These have been assigned a country value of 'unknown'.
'''

def join_dataframes(transaction_df, ip_df):

    '''
    Uses a SQL query to join the two pandas dataframes to assign a country to each transaction

    INPUT: transaction dataframe, ip address dataframe
    OUTPUT: dataframe containing reported transactions with an ip address in the ranges provided and a country column
    '''

    conn = sqlite3.connect(':memory:')
    transaction_df.to_sql('transaction_df', conn, index=False)
    ip_df.to_sql('ip_df', conn, index=False)

    qry = '''
    SELECT user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, class, country
    FROM transaction_df JOIN ip_df
    ON ip_address BETWEEN lower_bound_ip_address AND upper_bound_ip_address
    '''

    joined_df = pd.read_sql_query(qry, conn)

    return joined_df


def add_out_of_range_ip(transaction_df, joined_df):

    '''
    Makes another dataframe containing tranactions with an ip address that doesn't fall within any of the ranges provided in the ip dataframe. Assigns them a country value of 'unknown'

    INPUT: transaction dataframe, joined dataframe created in join_dataframes
    OUTPUT: dataframe containing transactions with an out of range ip address and a country column
    '''

    out_of_range_ids = np.setdiff1d(transaction_df['user_id'], joined_df['user_id'])
    out_of_range_df = transaction_df[transaction_df['user_id'].isin(out_of_range_ids)].copy()
    out_of_range_df['country'] = 'unknown'

    return out_of_range_df


def make_transactions_with_countries():

    '''
    Concatenates the above outputs to create a new transaction dataframe with countries assigned to each reported transaction

    INPUT:
    OUTPUT: transaction dataframe with a country column
    '''

    transaction_df = pd.read_csv('../data/Fraud_Data.csv')
    ip_df = pd.read_csv('../data/IpAddress_to_Country.csv')
    in_range_df = join_dataframes(transaction_df, ip_df)
    out_of_range_df = add_out_of_range_ip(transaction_df, in_range_df)
    transactions_with_countries_df = pd.concat([in_range_df, out_of_range_df], sort=False).reset_index(drop=True).drop(columns=['ip_address'])

    return transactions_with_countries_df

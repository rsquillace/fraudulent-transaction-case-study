import pandas as pd


'''
This script addresses the initial cleaning of the transaction data provided by the Ecommerce site. It simply puts the data into a workable format, and does no feature engineering
'''


def convert_datetimes(transaction_df):

    '''
    Converts the columns representing times into a datetime format

    INPUT: transaction dataframe
    OUTPUT: tranaction dataframe with proper datetimes
    '''

    transaction_df['signup_time'] = pd.to_datetime(transaction_df['signup_time'])
    transaction_df['purchase_time'] = pd.to_datetime(transaction_df['purchase_time'])

    return transaction_df


def gender_binary(transaction_df):

    '''
    Converts the gender column into a numeric binary

    INPUT: transaction dataframe
    OUTPUT: transaction dataframe with numeric gender column
    '''

    transaction_df['sex'] = transaction_df['sex'].apply({'M':1, 'F':0}.get)

    return transaction_df


def get_dummies(transaction_df):

    '''
    Makes dummy columns for the source and browser data

    INPUT: transaction dataframe
    OUTPUT: transaction dataframe with dummy columns
    '''

    return pd.get_dummies(transaction_df, columns=['source', 'browser'])


def clean_transaction_data():

    '''
    Loads dataframe from csv and performs all the above functions to produce a cleaned dataframe

    INPUT:
    OUTPUT: clean transaction dataframe
    '''

    transactions = pd.read_csv('data/transactions_with_country.csv')
    update_dates = convert_datetimes(transactions)
    update_gender = gender_binary(update_dates)
    update_dummies = get_dummies(update_gender)
    cleaned_transactions = update_dummies.copy()

    return cleaned_transactions

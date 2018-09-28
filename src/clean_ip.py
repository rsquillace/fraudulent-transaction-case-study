import pandas as pd


def fix_scientific_notation(ip_df):

    '''
     A few rows in the ip dataframe provided a lower bound in rounded scientific notation, and provided an inaccurate range. The ip dataframe is sorted (ascending), and the intervals do not overlap. To fill in these values, I replaced them with the the prior upper bound + 1

     INPUT: ip dataframe
     OUTPUT: ip dataframe with viable intervals
    '''

    for row in range(len(ip_df)):
    if ip_df['lower_bound_ip_address'].iloc[row]<ip_df['upper_bound_ip_address'].iloc[row] == False:
        ip_df['lower_bound_ip_address'].iloc[row] = ip_df['upper_bound_ip_address'].iloc[row-1]+1
    return ip_df


def clean_ip_data():

    '''
    Loads dataframe from csv and performs above function to produce clean ip dataframe

    INPUT:
    OUTPUT: clean ip dataframe
    '''

    ip = pd.read_csv('../data/IpAddress_to_Country.csv')
    clean_ip = fix_scientific_notation(ip).copy()
    return clean_ip

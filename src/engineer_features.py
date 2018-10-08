import pandas as pd
import numpy as np

'''
This script uses provided data to extract new features that allow for a better model
'''

class EngineerFeatures():

    def __init__(self, transaction_df):
        self.transaction_df = transaction_df

    def multiple_accounts(self):

        '''
        Transactions with the same device id are known to be different accounts made from the same device. This creates a column indicating if the recoded transaction was made from a device that has created multiple accounts
        '''

        self.transaction_df['multiple_accounts'] = self.transaction_df['device_id'].duplicated(keep=False)
        self.transaction_df = self.transaction_df.drop(columns=['device_id'])

        return self.transaction_df

    def time_between(self):

        '''
        This creates a column calculating the time between account signup and purchase in seconds
        '''
        
        self.transaction_df['time_between_signup_purchase'] = self.transaction_df['purchase_time'] - self.transaction_df['signup_time']
        self.transaction_df['time_between_signup_purchase'] = self.transaction_df['time_between_signup_purchase'] / np.timedelta64(1, 's')
        self.transaction_df = self.transaction_df.drop(columns=['signup_time', 'purchase_time'])

        return self.transaction_df

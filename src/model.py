import pandas as pd

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class MakeModel():

    def __init__(self, transaction_df):
        self.transaction_df = transaction_df

    def upsample_minority_class(self):

        '''
        The strong imbalance between fraudulent and non fraudulent transactions in the provided data is problematic when training models. To combat this, create a dataframe of resampled fraudulent transactions that is equal in length to non fraudulent transactions and concatenate the two.
        '''

        fraud_transactions = self.transaction_df[self.transaction_df['class']==1]
        nonfraud_transactions = self.transaction_df[self.transaction_df['class']==0]
        fraud_upsampled = resample(fraud_transactions, replace=True, n_samples=len(nonfraud_transactions), random_state=42)
        self.balanced_transactions = pd.concat([nonfraud_transactions, fraud_upsampled])


    def prepare_x_y(self):

        '''
        Defines model features and target
        '''

        self.x =  self.balanced_transactions.filter(items=['purchase_value', 'sex', 'age', 'source_Ads', 'source_Direct', 'source_SEO', 'browser_Chrome',
       'browser_FireFox', 'browser_IE', 'browser_Opera', 'browser_Safari','multiple_accounts', 'time_between_signup_purchase'])
        self.y = self.balanced_transactions['class']

    def split(self):

        '''
        Splits data into an 80-20 train test split
        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=.2, random_state = 42, stratify = self.y)

    def make_random_forest(self):

        '''
        Establishes random forest instance with 1000 trees
        '''

        self.rf = RandomForestClassifier(n_estimators=1000)

    def fit(self):

        '''
        Fits random forest model on training data
        '''

        self.rf.fit(self.X_train, self.y_train)

    def evaluate(self, test_x, test_y):

        '''
        The priority in this study is to minimize false negatives, while also keeping false positives down to a reasonable number so as to not waste time investingating transactions that aren't fraudulent. This makes confusion matrices a good evaluation metric for the model.

        INPUT: x to evaluate on, y to evaluate on **made into variables so you can easily evaluate the model on different samples of the provided data that aren't balanced 
        OUTPUT: confusion matrix elements
        '''

        true_neg, false_pos, false_neg, true_pos = confusion_matrix(test_y, self.rf.predict(test_x)).ravel()

        return 'true negatives: {}, false positives: {}, false negatives: {}, true positives: {}'.format(true_neg, false_pos, false_neg, true_pos)

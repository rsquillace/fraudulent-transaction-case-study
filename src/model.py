import pandas as pd

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


class MakeModel():

    def __init__(self, transaction_df):
        self.transaction_df = transaction_df

    def downsample_majority_class(self):

        '''
        The strong imbalance between fraudulent and non fraudulent transactions in the provided data is problematic when training models. To combat this, downsample the non fraudulent transactions until there is an equal amount between fraudulent and non fraudulent
        '''

        fraud_transactions = self.transaction_df[self.transaction_df['class']==1]
        nonfraud_transactions = self.transaction_df[self.transaction_df['class']==0]
        non_fraud_downsampled = resample(nonfraud_transactions, replace=False, n_samples=len(fraud_transactions), random_state=42)
        self.balanced_transactions = pd.concat([fraud_transactions, non_fraud_downsampled])


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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=.2, random_state = 42)

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

    def evaluate(self, probability_threshold, test_x, test_y):

        '''
        The priority in this study is to minimize false negatives, while also keeping false positives down to a reasonable number so as to not waste time investingating transactions that aren't fraudulent. Rather than relying entirely on classification, adjust the threshold for how probable fraud should be to warrent an investigation on a transaction.

        Confusion matrices a good evaluation metric for the model, as well as precision and recall scores.

        INPUT: probability of fraud required to flag transaction, x to evaluate on, y to evaluate on **made into variables so you can easily evaluate the model on different samples
        OUTPUT: confusion matrix elements, recall score, precision score
        '''

        probability = self.rf.predict_proba(self.X_test)[:,1]
        probability[probability >= probability_threshold] = 1
        probability[probability < probability_threshold] = 0

        recall = recall_score(self.y_test, probability)
        precision = precision_score(self.y_test, probability)
        true_neg, false_pos, false_neg, true_pos = confusion_matrix(test_y, self.rf.predict(test_x)).ravel()

        print ('recall score: {} \nprecision_score: {} \ntrue negatives: {}, false positives: {}, false negatives: {}, true positives: {}'.format(recall, precision, true_neg, false_pos, false_neg, true_pos))

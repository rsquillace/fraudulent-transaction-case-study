import pandas as pd

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score


class MakeModel():

    def __init__(self, transaction_df):
        self.transaction_df = transaction_df

    def prepare_x_y(self, list_of_columns):

        '''
        Defines target and model features based on the list of columns specified
        '''

        self.x =  self.transaction_df.filter(items=list_of_columns)
        self.y = self.transaction_df['class']

    def split(self):

        '''
        Splits data into an 80-20 train test split
        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=.2)

    def make_random_forest(self, num_trees):

        '''
        Creates random forest model with n trees
        '''

        self.rf = RandomForestClassifier(n_estimators=num_trees)

    def make_gradient_boost(self, num_trees, learning_rate):

        '''
        Creates gradient boosting model with n trees at a specified learninig rate
        '''

        self.gb = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=learning_rate)

    def fit(self):

        '''
        Fits random forest model on training data
        '''

        self.rf.fit(self.X_train, self.y_train)

    def evaluate(self, probability_threshold, test_x, test_y):

        '''
        The priority in this study is to minimize false negatives, while also keeping false positives down to a reasonable number so as to not waste time investingating transactions that aren't fraudulent. Rather than relying entirely on classification, adjust the threshold for how probable fraud should be to warrent an investigation on a transaction.

        Confusion matrices are a good evaluation metric for the model, as well as precision and recall scores.

        INPUT: probability of fraud required to flag transaction, x to evaluate on, y to evaluate on **made into variables so you can easily evaluate the model on different samples
        OUTPUT: confusion matrix elements, recall score, precision score
        '''

        probability = self.rf.predict_proba(test_x)[:,1]
        probability[probability >= probability_threshold] = 1
        probability[probability < probability_threshold] = 0

        recall = recall_score(test_y, probability)
        precision = precision_score(test_y, probability)
        auroc = roc_auc_score(test_y, probability)
        true_neg, false_pos, false_neg, true_pos = confusion_matrix(test_y, probability).ravel()

        print ('recall score: {} \nprecision_score: {} \narea under roc: {} \ntrue negatives: {}, false positives: {}, false negatives: {}, true positives: {}'.format(recall, precision, auroc, true_neg, false_pos, false_neg, true_pos))

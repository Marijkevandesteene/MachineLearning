import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from pathlib import Path

# Defining the learning curve function
def plot_learning_curve(model, X, y, cv, num_show, metric=None):
    """Plots learning curve given an inputted estimator and data"""    
    
    from sklearn.model_selection import learning_curve   
    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X=X, 
                                                            y=y, 
                                                            cv=cv, 
                                                            n_jobs=-1, 
                                                            scoring=metric,
                                                            return_times=False,
                                                            train_sizes=np.linspace(0.1, 1.0, num_show))
    
    # Calculating some statistics
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.grid()
    plt.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std), 
                     alpha=0.1,
                     color='red')
    plt.fill_between(train_sizes, (test_scores_mean - test_scores_std), test_scores_mean + test_scores_std, 
                     alpha=0.1,
                     color='green')
    plt.plot(train_sizes, train_scores_mean, color='red', label='Training score')
    plt.plot(train_sizes, test_scores_mean, color='green', label='Cross-validation score')
    plt.ylabel(str(metric) + ' Score')
    plt.xlabel('Training examples')
    plt.legend(loc='best')
    
    return

# Drop-column method for calculating feature importances
# Defining a function to perform the drop-column importance
def drop_col_feat_imp(model, X, y, random_state, verbose=False):
    """Feature importance calculation using drop-column feature importances"""
    from sklearn.base import clone 

    model_clone = clone(model)  # You can use this to make a copy of a model :)
    model_clone.random_state = random_state
    model_clone.fit(X=X, y=y)
    benchmark_score = model_clone.score(X=X, y=y)
    importances = [None] * len(X.columns)
    
    # For each column: drop the column and retrain a model & score
    for i, col in enumerate(X.columns):
        if verbose:
            print(f"Running for column {i}/{len(X.columns)}: {col}")
        # Drop the column we are looping over
        X_dropped = X.drop(col, axis=1)
        
        # Clone the model we already had in terms of hyperparams/settings
        model_clone = clone(model)
        model_clone.random_state = random_state  # Fixing the random state
        
        # Fit the model using the new data
        model_clone.fit(X=X_dropped, y=y)  # fit using the X with column dropped
        
        # Predict/score
        drop_col_score = model_clone.score(X=X_dropped, y=y)
        
        # Get importances and assign
        importances[i] = benchmark_score - drop_col_score
    
    # Making Pandas DF with results
    importances_df = pd.DataFrame(data={'feature': X.columns, 'importance': importances})
    importances_df_sorted = importances_df.sort_values(by='importance', ascending=False)
    
    return importances_df_sorted

#
def plot_roc(y_true, predicted_probs, positive_label=None, thresholds_every=10, title=''):
    """
    A more complex ROC visualization. It will also visualize various cutoffs with annotated text.
    """
    # Getting information we need form the roc_curve() function (fp, tp, thresholds)
    fp, tp, thresholds = roc_curve(y_true=y_true, y_score=predicted_probs, pos_label=positive_label)
    
    # Calculating the AUROC score
    roc_auc = roc_auc_score(y_true=y_true, y_score=predicted_probs)
    
    # Creating the more complicated figure
    plt.figure(figsize=(8, 6))
    plt.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    plt.xlabel('False positives rate')
    plt.ylabel('True positives rate')
    plt.xlim([-0.03, 1.0])
    plt.ylim([0.0, 1.03])
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # plot some thresholds
    len_thresholds = len(thresholds)
    colorMap=plt.get_cmap('jet', len_thresholds)
    for i in range(0, len_thresholds, thresholds_every):
        threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
        plt.text(x=fp[i] - 0.03, 
                 y=tp[i] + 0.005, 
                 s=threshold_value_with_max_four_decimals, 
                 fontdict={'size': 11}, 
                 color=colorMap(i/len_thresholds));
    plt.show()


def regression_lift(y_true, y_pred):
    """
    Compute the lift for regression.
    
    :param y_true: Array-like, true target values.
    :param y_pred: Array-like, predicted values by the model.
    :return: Lift value.
    """
    baseline_prediction = np.mean(y_true)
    mean_model_prediction = np.mean(y_pred)
    return mean_model_prediction / baseline_prediction

def classification_lift(y_true, y_pred):
    """
    Compute the lift for classification.
    
    :param y_true: Array-like, true binary labels (0 or 1).
    :param y_pred: Array-like, predicted binary labels (0 or 1).
    :return: Lift value.
    """
    # Percentage of actual positives in the entire dataset
    baseline = np.mean(y_true)
    
    # Indices where the prediction is positive
    positive_pred_indices = np.where(y_pred == 1)
    
    # Percentage of actual positives in the predicted positive group
    model_success_rate = np.mean(y_true[positive_pred_indices])
    
    return model_success_rate / baseline


def cumulative_regression_lift(y_true, y_pred):
    """
    Compute the cumulative lift for regression.
    
    :param y_true: Array-like, true target values.
    :param y_pred: Array-like, predicted values by the model.
    :return: List of cumulative lift values.
    """
    # Sort y_true and y_pred by y_pred values
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    cumulative_lifts = []
    for i in range(1, len(y_true) + 1):
        mean_true_up_to_i = np.mean(y_true_sorted[:i])
        mean_pred_up_to_i = np.mean(y_pred_sorted[:i])
        
        lift = mean_pred_up_to_i / mean_true_up_to_i if mean_true_up_to_i != 0 else 0
        cumulative_lifts.append(lift)
        
    return cumulative_lifts

def cumulative_classification_lift(y_true, y_prob):
    """
    Compute the cumulative lift for classification.
    
    :param y_true: Array-like, true binary labels (0 or 1).
    :param y_prob: Array-like, predicted probabilities for the positive class.
    :return: List of cumulative lift values.
    """
    # Sort y_true based on predicted probabilities in descending order
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n = len(y_true)
    cumulative_positive = np.cumsum(y_true_sorted)
    
    # Calculate cumulative percentage of actual positives
    cumulative_positive_percentage = cumulative_positive / np.arange(1, n + 1)
    
    # Overall percentage of actual positives in the dataset
    overall_positive_percentage = np.sum(y_true) / n
    
    cumulative_lifts = cumulative_positive_percentage / overall_positive_percentage
    
    return cumulative_lifts


def plot_lift_curve(X,actual_target,model):
    """ 
    DESCRIPTION
    ____________________________________________________________    Function that takes in X features, Y features and model object as input and creates Gain percentage and Lift.
    
    PARAMETERS
    _____________________________________________________________
       X: DataFrame
           The X features that are used by the model.
       actual_target: DataFrame
            Actual target that is used to train the model.
       model: fit object
            The fit object returned by the training algorithm.
    RETURNS
    ______________________________________________________________
       output_df:DataFrame
            Output dataframe with columns,
    
       Max_Scr : Maximum Probability Score for that Decile
       Min_Scr : Minimum Probability Score for that Decile
       Actual : Sum of targets captured by the Decile
       Total : Total population of the Decile
       Population_perc : Percentage of population in the Decile
       Per_Events : Percentage of Events in the decile
       Gain Percentage : Gain percentage
       Cumulative Population : Cumulative sum of population down the     decile
       Lift : Lift provided by that particular decile    """
    avg_tgt = actual_target.sum()/len(actual_target)
    df_data = X.copy()
    X_data = df_data.copy()
    df_data['Actual'] = actual_target
    df_data['Predict']= model.predict(X_data)
    y_Prob = pd.DataFrame(model.predict_proba(X_data))
    df_data['Prob_1']=list(y_Prob[1])
    df_data.sort_values(by=['Prob_1'],ascending=False,inplace=True)
    df_data.reset_index(drop=True,inplace=True)
    df_data['Decile']=pd.qcut(df_data.index,10,labels=False)
    output_df = pd.DataFrame()
    grouped = df_data.groupby('Decile',as_index=False)
    output_df['Max_Scr']=grouped.max().Prob_1
    output_df['Min_Scr']=grouped.min().Prob_1
    output_df['Actual']=grouped.sum().Actual
    output_df['Total']=grouped.count().Actual
    output_df["Population_perc"] = (output_df["Total"]/len(actual_target))*100
    output_df['Per_Events'] = (output_df['Actual']/output_df['Actual'].sum())*100
    output_df['Gain_Percentage'] = output_df.Per_Events.cumsum()
    output_df["Cumulative_Population"] = output_df.Population_perc.cumsum()
    output_df["Lift"] = output_df["Gain_Percentage"]/output_df["Cumulative_Population"]
    return output_df

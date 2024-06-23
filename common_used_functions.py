import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats

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
#%% import dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


#%% functions for preprocessing

#check for duplicates
def check_duplicates(df):
    """Checks and removes duplicate rows from a dataframe.

    Args:
        df (pandas.DataFrame): Raw DataFrame

    Returns:
        df (pandas.DataFrame): DataFrame with duplicate rows removed.
    """    
    has_dup = df.duplicated()
    true_dup = np.where(has_dup == True)
    if len(true_dup[0]) > 0:
        print("Data has", len(true_dup[0]), "duplicates")
        df.drop_duplicates(keep='first', inplace=True)
    else:
        print("No duplicates found")
    return df

#function for one-hot encoding
def one_hot(df, column_names):
    """Converts columns in a dataframe to one-hot encoded variants. The original columns are removed. The first one-hot encoded column is dropped.

    Args:
        df (pandas.DataFrame): Raw DataFrame
        column_names (list): list with the names of the columns to one-hot encode

    Returns:
        df (pandas.DataFrame): DataFrame with one-hot encoded columns.
    """    
    for col in column_names:
        dummies = pd.get_dummies(df[[col]].astype('category'),drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([col], axis=1)
    return df

#function for adding interaction terms between all columns
def add_full_interactions(df,degree=2):
    """Add interaction terms between all columns in a dataframe. 

    Args:
        df (pandas.DataFrame): Raw DataFrame
        degree (int, optional): Degree of polynomial to compute. Defaults to 2.

    Returns:
        df (pandas.DataFrame): DataFrame with interaction terms.
    """    
    poly = PolynomialFeatures(interaction_only=True, include_bias=False, degree=degree)
    X_poly = poly.fit_transform(df)
    column_names = list(poly.get_feature_names_out())
    X_poly = pd.DataFrame(X_poly,columns=column_names)
    return X_poly

#function for adding interaction terms between select columns
def add_interactions(df,columns,degree=2):
    """Add interaction terms between specific columns in a dataframe. 

    Args:
        df (pandas.DataFrame): Raw DataFrame
        columns (list): list of column names between which interactions are computed.
        degree (int, optional): _description_. Defaults to 2.

    Returns:
        df (pandas.DataFrame): DataFrame with interaction terms.
    """    
    poly = PolynomialFeatures(interaction_only=True, include_bias=False, degree=degree)
    X_poly = poly.fit_transform(df[columns])
    column_names = list(poly.get_feature_names_out())
    X_poly = pd.DataFrame(X_poly,columns=column_names)
    X_poly = pd.concat([df, X_poly.drop(columns,axis=1)], axis=1)
    return X_poly

#function for computing sigmoid (e.g. to look at logistic regression fit)
def sigmoid(x,b):
    """Compute a sigmoid function.

    Args:
        x (int, float): x-axis values
        b (int, float): bias term

    Returns:
        signmoid (numpy.array): the sigmoid function
    """    
    return 1 / (1 + np.exp(-(b+x)))

#min-max scale vector to pre-specified range
def linmap(vector, new_min, new_max):
    """Linearly map a vector onto a new range.

    Args:
        vector (np.array): vector of numbers
        new_min (int, float): new minimum for the vector 
        new_max (int, float): new maximum for the vector 

    Returns:
        scaled_vector: vector mapped onto the new range of values
    """    
    vector = np.array(vector)
    old_min = np.min(vector)
    old_max = np.max(vector)
    
    # Avoid division by zero if the old_min equals old_max
    if old_min == old_max:
        return np.full_like(vector, new_min if old_min == old_max else new_max)
    
    # Scale the vector
    scaled_vector = (vector - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_vector

def filter_features(X_train,X_test,thresh=0.95):
    """Filters features in a design matrix by spearman correlation coefficient. Features that correlate above a threshold are removed, column-wise. 

    Features are identified based on the training set. The same features are removed from the testing set.

    Args:
        X_train (pandas.DataFrame): Design matrix for the training set.
        X_test (pandas.DataFrame): Design matrix for the testing set.
        thresh (float, optional): Spearman's rho to use as threshold. Defaults to 0.95.

    Returns:
        X_train (pandas.DataFrame): Design matrix for the training set.
        X_test (pandas.DataFrame): Design matrix for the testing set.
    """    
    
    #compute correlations and get half the correlations
    cm = X_train.corr(method = "spearman").abs() #compute correlation matrix    
    upper = cm.where(np.triu(np.ones(cm.shape), k = 1).astype(bool)) #select upper triangle of matrix

    #find index / indices of feature(s) with correlation above threshold
    columns_to_drop = [column for column in upper.columns if any(upper[column] > thresh)]

    # Drop features
    X_train = X_train.drop(columns_to_drop, axis = 1)
    X_test = X_test.drop(columns_to_drop, axis = 1)

    # Print the number of features to be dropped
    num_features_dropped = len(columns_to_drop)
    if num_features_dropped > 0:
        print(f"Dropping {num_features_dropped} features due to high correlation.")
    else:
        print("No features dropped based on correlation.")

    return X_train, X_test, columns_to_drop

#%% functions for post-processing / prediction
   
#function to make a combined prediction with the classification and regression models
def make_combined_prediction(X,classification_model,regression_model):
    """_summary_

    Args:
        X (_type_): _description_
        classification_model (_type_): _description_
        regression_model (_type_): _description_

    Returns:
        _type_: _description_
    """    
    #first, make prediction with regression model
    r_pred = regression_model.predict(X)

    #second, make prediction with classification model
    c_pred = classification_model.predict(X)

    #set predicted delay to 0 for the flights that the classification model estimated to be on time
    return r_pred * c_pred

def compute_feature_importance(X, y, classification_model, regression_model, metric, npermutes):
    """_summary_

    Args:
        X (pandas.DataFrame): Design matrix.
        y (numpy.Array): Labels
        classification_model (model): _description_
        regression_model (model): _description_
        metric (function): _description_
        npermutes (int): Number of iterations for permutation testing

    Returns:
        _type_: _description_
    """    
    y = np.expm1(y)

    # Calculate the baseline error metric on the original data
    y_pred = make_combined_prediction(X,classification_model,regression_model)
    baseline_metric = metric( y , np.expm1(y_pred))
    print('base metric = ' + str(baseline_metric))

    feature_importance = np.empty( (npermutes,len(X.columns)) ) #will contain permuted metrics

    for permi in range(0,npermutes): #iterate over permutations
        print('permutaiton ' + str(permi + 1) + ' / ' + str(npermutes))
        for col_num, col in enumerate(X.columns): #iterate over columns in design matrix (features)

            #shuffle rows in feature (permute)
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])

            #make sure columns are the right data type after shuffling (numyp turns categorical into object)
            #make sure columns are the right data type after shuffling (numyp turns categorical into object)
            cat_columns = X_permuted.select_dtypes(include='object').columns #identify the numeric columns
            X_permuted[cat_columns] = X_permuted[cat_columns].astype('category')

            #calculate the metric on the permuted design matrix
            y_pred_permuted = make_combined_prediction(X_permuted, classification_model, regression_model)
            permuted_metric = metric(y, np.expm1(y_pred_permuted))

            #calculate feature importance (the difference with the un-shuffled metric)
            feature_importance[permi, col_num] = permuted_metric - baseline_metric #positive values mean that, when shuffling this feature, error increased (thus, the feature is important)

    #sort features by their importance
    sidx = np.argsort( np.mean(feature_importance,axis=0) ) #get indices of features sorted by importance
    sidx = sidx[::-1] #sort descending instead of ascending

    feature_importance = feature_importance[:,sidx] #sort features
    feature_labels     = X.columns[sidx] #sort labels

    return feature_importance, feature_labels

#%% functions for plotting

def prediction_plot(y_test, y_pred):
    """
    Evaluate the model by plotting scatter plots of true vs predicted values
    with a least squares regression line and MSE.
    
    Parameters:
    y_test (array-like): True values for the test set.
    y_pred (array-like): Predicted values for the test set.
    """
    plt.figure(figsize=(6, 5))
    mse_test  = mean_squared_error(y_test,  y_pred)
      
    # Scatter plot for the test set
    plt.scatter(y_test, y_pred, alpha=0.5, label='Data')
    
    # Regression line for test set
    m_test, b_test = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m_test * y_test + b_test, color='blue', label=f'Fit: y={m_test:.2f}x + {b_test:.2f}')
    
    # Perfect prediction line for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test Set (MSE: {mse_test:.2f})')
    plt.legend()
    
    plt.tight_layout()

def feature_importance_plot(feature_importance,feature_labels, N):
    mean_importance = np.mean(feature_importance, axis=0) #average across permutations
    upper_range     = np.percentile(feature_importance,axis=0, q=95 ) #upper confidence interval
    lower_range     = np.percentile(feature_importance,axis=0, q=5  ) #lower confidence interval

    # Calculate the error bars (upper and lower)
    error_bars = np.array([mean_importance - lower_range, upper_range - mean_importance])

    P = len(feature_labels)

    # Create the horizontal bar chart
    #plt.figure(figsize=(10, P/2))  # Adjust the size based on the number of features
    plt.barh(np.arange(P), np.mean(feature_importance, axis=0), xerr=error_bars, align='center', color='skyblue', ecolor='black', capsize=5)

    # Set the y-ticks and invert y-axis to have the most important features on top
    plt.yticks(np.arange(P), labels=feature_labels)
    plt.gca().invert_yaxis()

    # Add labels and title
    plt.xlabel('Feature Importance (Delta MSE)')
    plt.ylabel('Feature name')
    plt.title('Feature Importance with 95% Confidence Intervals')
    #plt.gca().invert_yaxis()

    plt.ylim([N-0.5, -0.5])
    #plt.ylim([len(X_train.columns)-N + 0.5, len(X_train.columns)])
    plt.title('Top ' + str(N) +  ' features')

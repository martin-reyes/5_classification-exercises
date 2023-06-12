

def get_baseline_model(df, target):
    '''
    Takes in a DataFrame and target column
    Makes target mode its prediction
    Returns df with baseline model predictions
    '''
    # find most frequent target class
    target_mode = df[target].value_counts().index[0]
    # predict target mode everytime
    df['model_baseline'] = target_mode
    
    return df
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_cat_and_cont_cols(df, num_unique=10):
    '''
    Identifies columns from a df as continuous or categorical
    based on the number of unique values for each column.
    Returns a list of categorical columns and a list of continuous columns
    '''
    # store column in categorical list if there are `num_unique` or less unique values
    cat_cols = [col for col in df.columns if len(df[col].unique()) <= 10]
    # store column in categorical list if there are more than `num_unique` unique values 
    cont_cols = [col for col in df.columns if len(df[col].unique()) > 10]
    
    # print continous columns that are objects
    for col in cont_cols:
        if df[col].dtype == 'O':
            print(f'{col} is continuous but not numeric. Check if column needs to be cleaned')
    
    return cat_cols, cont_cols

def explore_univariate_categorical_cols(df, cat_cols = None):
    
    # set default categorical columns
    if cat_cols == None:
        cat_cols = get_cat_and_cont_cols(df)[0]
    
    for col in cat_cols:
        print(col.upper())
        # Combine count and normalized frequency into a single DataFrame
        frequency_table = pd.concat([df[col].value_counts(), 
                                     df[col].value_counts(normalize=True)], axis=1).reset_index()
        frequency_table.columns = [col,'Count', 'Frequency']
        display(frequency_table)
        # bar plot
        plt.figure(figsize=(3, 2))
        sns.countplot(x=col, data=df)
        plt.show()
        print()
        
def explore_univariate_continuous_cols(df, cont_cols = None):
    
    # set default categorical columns
    if cont_cols == None:
        cont_cols = get_cat_and_cont_cols(df)[1]
    
    # descriptive stats
    print('Descriptive Stats:\n')
    display(df[cont_cols].describe())

    for col in cont_cols:
        print('-'*60, '\n', col.upper(), '\n')
        # most frequent values
        print('Most Frequent Values:')
        print(df[col].value_counts().head(3))
        # set figure
        fig, axes = plt.subplots(1, 2, figsize=(6, 2))
        # histogram
        sns.histplot(x=col, data=df, ax=axes[0])
        # boxplot
        sns.boxplot(x=col, data=df, ax=axes[1])

        plt.show()
        print()
        
def explore_bivariate_cont_to_cat_target(df, target, cont_cols=None):
    '''
    Explores continuous feature relationships to categorical target
    '''

    # set default categorical columns
    if cont_cols == None:
        cont_cols = get_cat_and_cont_cols(df)[1]
        
    # display descriptive stats for each target category
    display(df.groupby(target)[cont_cols].describe().T)
    
    # display, in order, pearson R correlations to the target if target is binary
    if len(df[target].unique()) == 2:
        print(f'Continuous feature correlations (Pearson R) to {target}:')
        display(df[cont_cols+[target]].corr()[target]\
                                      .sort_values(ascending=False))

    for col in cont_cols:
        plt.figure(figsize=(3, 3)) 
        sns.barplot(x=target, y=col, data=df, estimator='mean')
        plt.title(f'{col} averages')
        # add line indicating estimate of all targets
        plt.axhline(df[col].mean(), label=f'Total {col} mean', color='red')
        # plt.legend()
        plt.show()
        print()


def explore_bivariate_cat_to_cat_target(df, target, cat_cols=None):
    '''
    Explores categorical feature relationships to categorical target
    '''
    # set default categorical columns
    if cat_cols == None:
        cat_cols = get_cat_and_cont_cols(df)[0]
        
    # display, in order, pearson R "correlations" to the target if target is binary
    if len(df[target].unique()) == 2:
        print(f'Categorical feature (integer-type) "correlations" (Pearson R) to {target}:')
        display(df[cat_cols].corr(numeric_only=True)[target]\
                                  .sort_values(ascending=False))

    for col in cat_cols:
        plt.figure(figsize=(3, 3)) 
        sns.barplot(x=col, y=target, data=df, estimator='mean')
        # plt.title(f'{target} averages')
        # add line indicating estimate of all targets
        plt.axhline(df[target].mean(), label=f'Total {target} mean', color='red')
        # plt.legend()
        plt.show()
        print()


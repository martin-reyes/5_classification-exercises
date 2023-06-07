import acquire
import pandas as pd
from sklearn.model_selection import train_test_split

def prep_iris(iris=acquire.get_iris_data()):
    '''
    accepts the raw iris data
    returns the data with the transformations above applied
    '''
    # Drop the species_id and measurement_id columns.
    cols_to_drop = ['species_id', 'measurement_id']
    iris = iris.drop(columns=cols_to_drop)

    # Rename the species_name column to just species.
    iris = iris.rename(columns={'species_name':'species'})
    
    # Create dummy variables of the species name and concatenate onto the iris dataframe. 
    dummy_df = pd.get_dummies(iris['species'], drop_first=True)
    iris = pd.concat([iris, dummy_df], axis=1)
    return iris


def prep_titanic(titanic=acquire.get_titanic_data()):
    '''
    accepts the raw titanic data
    returns the data with the transformations above applied
    '''
    # dropping embarked column
    cols_to_drop = ['embarked','pclass']
    titanic = titanic.drop(columns=cols_to_drop)
    
    # Encode the categorical columns.
    dummy_df = pd.get_dummies(titanic[['embark_town', 'class', 'sex']], dummy_na=False, drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)
    
    # rename columns to be lowercased with underscores
    titanic.columns = [col.lower().replace(" ", "_") for col in titanic.columns]
    return titanic


def prep_telco(telco=acquire.get_telco_data()):
    '''
    accepts the raw telco data
    returns the data with the transformations above applied
    '''
    # Dropping foreign keys
    telco = telco.iloc[:,3:]
    
    # Encoding binary variables
    binary_cols = ['partner','dependents','phone_service', 'paperless_billing', 'churn']
    for col in binary_cols:
        telco[col] = telco[col].replace({'Yes': 1, 'No': 0})

    # Encoding multiclass variables
    dummy_df = pd.get_dummies(telco[['gender', 'multiple_lines', 'online_security', 'online_backup',
                                     'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
                                     'contract_type', 'internet_service_type', 'payment_type']],
                              dummy_na=False, drop_first=True)
    telco = pd.concat([telco, dummy_df], axis=1)
    
    # rename columns to be lowercased with underscores
    telco.columns = [col.lower().replace(" ", "_") for col in telco.columns]
    return telco


def split_data(df, test_size=.2, validate_size=.2, stratify_col=None, random_state=None):
    '''
    take in a DataFrame and return train, validate, and test DataFrames;.
    default size 
    return train, validate, test DataFrames.
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size, random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state,
                                                stratify = df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size, random_state=random_state,
                                           stratify=train_validate[stratify_col])       
    return train, validate, test

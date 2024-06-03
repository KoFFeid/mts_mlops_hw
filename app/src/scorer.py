import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier

MODEL_TRESHOLD  = 0.34

# Make prediction
def make_pred(model, df, id):

    print('Importing pretrained model...')

    # Define optimal threshold
    model_th = MODEL_TRESHOLD
    probability = pd.DataFrame({
        'client_id':  id,
        'probability': model.predict_proba(df)[:, 1]
    })

    # Make predictions dataframe
    predictions = pd.DataFrame({
        'client_id':  id,
        'preds': (model.predict_proba(df)[:, 1] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return predictions, probability
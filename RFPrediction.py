import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


def clean():
    # load main dataset and filter out teams that did not make postseason
    df = pd.read_csv('Datasets/cbb.csv')
    filtered_df = df[df['POSTSEASON'].notna()]

    # load target dataset and filter out teams that did not make postseason
    filtered_df.to_csv('Datasets/cbb_cleaned.csv', index=False)
    df = pd.read_csv('Datasets/cbb24.csv')
    df = df[df['SEED'].notna()]
    # print("Current column names:", df.columns.tolist())

    # fix column name discrepancy and save cleaned file
    df.rename(columns={'EFG%': 'EFG_O', 'EFGD%': 'EFG_D'}, inplace=True)
    df.to_csv('Datasets/cbb24_cleaned.csv', index=False)


def preprocess():
    # load dataset
    df = pd.read_csv('Datasets/cbb_cleaned.csv')

    # prepare data for training by extracting relevant features
    features = [col for col in df.columns if col not in ['RK', 'TEAM', 'G', 'POSTSEASON', 'YEAR', 'CONF']]
    X = df[features]
    y = df['POSTSEASON']  # target

    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate the model's accuracy
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # save the model
    model_filename = 'Datasets/cbb_model.joblib'
    dump(model, model_filename)


def predict():
    # load model and datasets
    df = 'Datasets/cbb_model.joblib'
    loaded_model = load(df)
    data_2024 = pd.read_csv('Datasets/cbb24_cleaned.csv')

    # select the same features from preprocessing
    features = [col for col in data_2024.columns if col not in ['RK', 'TEAM', 'G', 'POSTSEASON', 'YEAR', "CONF"]]
    X_2024 = data_2024[features]

    # predict team finishes using model
    predictions_2024 = loaded_model.predict(X_2024)
    data_2024['PredictedPostseason'] = predictions_2024
    print(data_2024[['TEAM', 'PredictedPostseason']])

    # save data
    data_2024.to_csv('output.csv', index=False)


if __name__ == "__main__":
    clean()
    preprocess()
    predict()

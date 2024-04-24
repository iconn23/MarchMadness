import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
from joblib import load


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

    # dictionary of training functions
    train_functions = {
        'Random Forest': train_RF,
        'K-Nearest Neighbors': train_KNN,
        'Support Vector Machine': train_SVM,
        'Logistic Regression' : train_LR
    }

    # iterate over training functions and train models
    for name, train_func in train_functions.items():
        print(f"Training with {name}:")
        train_func(X_train, X_test, y_train, y_test)


def train_SVM(X_train, X_test, y_train, y_test):
    # train the Support Vector Machine Classifier
    model = SVC(kernel='linear', random_state=42)  # kernel can be 'linear', 'poly', 'rbf', 'sigmoid', etc.
    model.fit(X_train, y_train)

    # evaluate the model's accuracy
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' considers class imbalance
    print(f'F1-Score: {f1}\n')

    # save the model
    model_filename = 'Models/SVM.joblib'
    dump(model, model_filename)


def train_LR(X_train, X_test, y_train, y_test):
    # train the Logistic Regression model
    model = LogisticRegression(max_iter=7500)
    model.fit(X_train, y_train)

    # evaluate the model's accuracy
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    print(f'F1-Score: {f1}\n')

    # save the model
    model_filename = 'Models/LR.joblib'
    dump(model, model_filename)


def train_KNN(X_train, X_test, y_train, y_test):
    # train the K-Nearest Neighbors Classifier
    model = KNeighborsClassifier(n_neighbors=9)  # n_neighbors can be adjusted based on your specific needs
    model.fit(X_train, y_train)

    # evaluate the model's accuracy
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' takes into account class imbalance
    print(f'F1-Score: {f1}\n')

    # save the model
    model_filename = 'Models/KNN.joblib'
    dump(model, model_filename)


def train_RF(X_train, X_test, y_train, y_test):
    # train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate the model's accuracy
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    f1 = f1_score(y_test, y_pred,
                  average='weighted')  # 'weighted' handles class imbalance by weighting the score by class support
    print(f'F1-Score: {f1}\n')

    # save the model
    model_filename = 'Models/RF.joblib'
    dump(model, model_filename)


def predict():
    # load models and datasets
    models = [os.path.join('Models', f) for f in os.listdir('Models') if f.endswith('.joblib')]
    data_2024 = pd.read_csv('Datasets/cbb24_cleaned.csv')

    # select the same features from preprocessing
    features = [col for col in data_2024.columns if col not in ['RK', 'TEAM', 'G', 'POSTSEASON', 'YEAR', "CONF"]]
    X_2024 = data_2024[features]

    # define the order of the postseason finishes
    finish_order = ['Champions', '2ND', 'F4', 'E8', 'S16', 'R32', 'R64']

    # iterate over models and make predictions
    for model_filename in models:
        loaded_model = load(model_filename)

        # predict team finishes using model
        predictions_2024 = loaded_model.predict(X_2024)
        data_2024['PredictedPostseason'] = pd.Categorical(
            predictions_2024,
            categories=finish_order,
            ordered=True
        )

        # sort data by team finish
        sorted_data = data_2024.sort_values('PredictedPostseason')
        output_data = sorted_data[['TEAM', 'PredictedPostseason']]
        champion = sorted_data.iloc[0]['TEAM']
        model_name = os.path.basename(model_filename)[:-7]
        print(f"{model_name} model predicts {champion} as champion")

        # save output file
        output_filename = f"{model_name}_output.csv"
        output_path = os.path.join('Predictions', output_filename)
        output_data.to_csv(output_path, index=False)
        # print(f"Output saved to {output_path}")


if __name__ == "__main__":
    clean()
    preprocess()
    predict()

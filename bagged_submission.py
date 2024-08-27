import pandas as pd

sgd_submission = pd.read_csv("sgd_survival.csv").rename(
    columns={"Survived": "Survived_sgd"}
)
knn_submission = pd.read_csv("knn_survival.csv").rename(
    columns={"Survived": "Survived_knn"}
)
rf_submission = pd.read_csv("rf_survival.csv").rename(
    columns={"Survived": "Survived_rf"}
)
tensorflow_submission = pd.read_csv("tensorflow_survival.csv").rename(
    columns={"Survived": "Survived_tensorflow"}
)

bagged_sumbission = (
    sgd_submission.merge(knn_submission, on="PassengerId")
    .merge(rf_submission, on="PassengerId")
    .merge(tensorflow_submission, on="PassengerId")
)


bagged_sumbission["Survived"] = (
    bagged_sumbission.loc[:, bagged_sumbission.columns.str.startswith("Survived")]
    .median(axis=1)
    .astype(int)
)

bagged_sumbission = bagged_sumbission[["PassengerId", "Survived"]]

bagged_sumbission.to_csv("bagged_survival.csv", index=False)

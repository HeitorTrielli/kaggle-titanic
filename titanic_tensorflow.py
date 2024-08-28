import os
import time

import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from preprocess import load_titanic_data, preprocess_feats

titanic_train, titanic_test = load_titanic_data()

titanic_train.columns = titanic_train.columns.str.lower()
titanic_test.columns = titanic_test.columns.str.lower()

raw_train_feats = titanic_train.drop(columns=["survived", "name"])
raw_test_feats = titanic_test.drop(columns="name")

train_label = titanic_train.survived

train_feats, test_feats = preprocess_feats(raw_train_feats, raw_test_feats)


def model_learning_rate(learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=train_feats.shape[1:]),
            tf.keras.layers.Flatten(input_shape=[28, 28]),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model


model = model_learning_rate(0.008)

t0 = time.time()
n_epochs = 300
history = model.fit(train_feats, train_label, epochs=n_epochs, validation_split=0.15)
t1 = time.time()

prediction = model.predict(test_feats)

prediction[prediction > 0.5] = 1
prediction[prediction < 0.5] = 0

df = pd.concat(
    [
        raw_test_feats[["passengerid"]].rename(columns={"passengerid": "PassengerId"}),
        pd.DataFrame(prediction, columns=["Survived"]),
    ],
    axis=1,
)

df["Survived"] = df["Survived"].apply(int)

df.to_csv("tensorflow_survival.csv", index=False)

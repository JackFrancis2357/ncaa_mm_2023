import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import numpy as np
import pandas as pd
from metadata_configs import get_training_data_col_names

a = np.load("./training_data/men_tourney/all_games.npy")
print(a.shape)

colnames = get_training_data_col_names()
df = pd.DataFrame(a, columns=colnames)
print(df.shape)


df.to_csv('test.csv')

season_info = df.iloc[:, 2].values
overall_x = df.iloc[:, 3:-1].values
overall_y = df.iloc[:, -1].values

scaler = StandardScaler()
scaled_overall_x = scaler.fit_transform(overall_x)
# scaled_overall_x = overall_x


print(scaled_overall_x.shape)

hold_out_year = 2019
x_test_ind = np.where(season_info == hold_out_year)
x_train_ind = np.where(season_info != hold_out_year)

scaled_x_test = scaled_overall_x[x_test_ind]
y_test = overall_y[x_test_ind]

scaled_x_train = scaled_overall_x[x_train_ind]
y_train = overall_y[x_train_ind]

print(scaled_x_train.shape, scaled_x_test.shape)

dropout_pct = 0.7
input_one = Input(shape=(overall_x.shape[1],))
hidden_one = Dense(128, activation="relu")(input_one)
dropout_one = Dropout(dropout_pct)(hidden_one)
hidden_two = Dense(256, activation="relu")(dropout_one)
dropout_two = Dropout(dropout_pct)(hidden_two)
hidden_three = Dense(512, activation="relu")(dropout_two)
dropout_three = Dropout(dropout_pct)(hidden_three)
hidden_four = Dense(256, activation="relu")(dropout_three)
dropout_four = Dropout(dropout_pct)(hidden_four)
hidden_five = Dense(128, activation="relu")(dropout_four)
dropout_five = Dropout(dropout_pct)(hidden_five)
hidden_six = Dense(32, activation="relu")(dropout_five)
dropout_six = Dropout(dropout_pct)(hidden_six)
output = Dense(1, activation="sigmoid")(dropout_six)

model = Model(input_one, output)
model.compile(loss="mse", optimizer=Adam(), metrics="mse")

model.fit(scaled_x_train, y_train, epochs=1000, batch_size=2048)

test_metrics = model.evaluate(scaled_x_test, y_test)
print(test_metrics)
y_pred = model.predict(scaled_x_test)
# print(y_pred)
for metric, metric_name in zip(test_metrics, model.metrics_names):
    print(f"Test {metric_name} is {round(metric, 8)}")

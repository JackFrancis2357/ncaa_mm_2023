import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


class MLModel:
    def __init__(self):
        pass

    def build_model(self):
        pass

    def run(self, input_shape):
        self.input_shape = input_shape
        self.build_model()
        self.create_callbacks()
        return self.model, self.callbacks

    def create_callbacks(self):
        lr_plateau = ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.9)
        early_stopping = EarlyStopping(monitor="val_loss", patience=25)
        self.callbacks = [lr_plateau, early_stopping]


class MenMLModel(MLModel):
    def build_model(self):
        n = 1
        self.dropout_pct = 0.3
        input_one = Input(shape=(self.input_shape,))
        hidden_one = Dense(128 * n, activation="relu")(input_one)
        dropout_one = Dropout(self.dropout_pct)(hidden_one)
        hidden_two = Dense(256 * n, activation="relu")(dropout_one)
        dropout_two = Dropout(self.dropout_pct)(hidden_two)
        hidden_three = Dense(512 * n, activation="relu")(dropout_two)
        dropout_three = Dropout(self.dropout_pct)(hidden_three)
        hidden_four = Dense(256 * n, activation="relu")(dropout_three)
        dropout_four = Dropout(self.dropout_pct)(hidden_four)
        hidden_five = Dense(128 * n, activation="relu")(dropout_four)
        dropout_five = Dropout(self.dropout_pct)(hidden_five)
        hidden_six = Dense(64 * n, activation="relu")(dropout_five)
        dropout_six = Dropout(self.dropout_pct)(hidden_six)
        output = Dense(1, activation="sigmoid")(dropout_six)

        self.model = Model(input_one, output)
        self.model.compile(loss="mse", optimizer=Adam())

        return super().build_model()


class WomenMLModel(MLModel):
    def build_model(self):
        n = 1
        self.dropout_pct = 0.5
        input_one = Input(shape=(self.input_shape,))
        hidden_one = Dense(128 * n, activation="relu")(input_one)
        dropout_one = Dropout(self.dropout_pct)(hidden_one)
        hidden_two = Dense(256 * n, activation="relu")(dropout_one)
        dropout_two = Dropout(self.dropout_pct)(hidden_two)
        hidden_three = Dense(512 * n, activation="relu")(dropout_two)
        dropout_three = Dropout(self.dropout_pct)(hidden_three)
        hidden_four = Dense(256 * n, activation="relu")(dropout_three)
        dropout_four = Dropout(self.dropout_pct)(hidden_four)
        hidden_five = Dense(128 * n, activation="relu")(dropout_four)
        dropout_five = Dropout(self.dropout_pct)(hidden_five)
        hidden_six = Dense(64 * n, activation="relu")(dropout_five)
        dropout_six = Dropout(self.dropout_pct)(hidden_six)
        output = Dense(1, activation="sigmoid")(dropout_six)

        self.model = Model(input_one, output)
        self.model.compile(loss="mse", optimizer=Adam())

        return super().build_model()

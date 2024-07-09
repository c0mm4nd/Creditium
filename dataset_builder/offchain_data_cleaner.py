# %%
# follow https://github.com/finlytics-hub/credit_risk_model/blob/master/Credit_Risk_Model_and_Credit_Scorecard.ipynb
import pandas as pd
import numpy as np

pd.options.display.max_columns = None


# %%
accepted_df = pd.read_csv("Loan_status_2007-2020Q3.csv", index_col=0, low_memory=False)
accepted_df.info

# %%
accepted_df = accepted_df[accepted_df["application_type"] == "Individual"]
accepted_df.drop(columns=["application_type"], inplace=True)
accepted_df

# %%

# %%
na_values = accepted_df.isnull().mean()
na_values[na_values > 0.3]

# %%
accepted_df = accepted_df.drop(labels=na_values[na_values > 0.3].index, axis=1)
accepted_df

# %%
"""
drop redundant and forward-looking columns
redundant like id, member_id, title, etc.
forward-looking like recoveries, collection_recovery_fee, etc.
drop sub_grade as same information is captured in grade column
drop next_pymnt_d since, given that our data is historical and this column is supposed to have future dates, will not make sense for our model
"""
accepted_df.drop(
    columns=[
        "id",
        "sub_grade",
        "emp_title",  # TOOOOOOOOOO many(450756)
        "url",
        "title",
        "zip_code",
        "recoveries",
        "collection_recovery_fee",
        "collections_12_mths_ex_med",
        "initial_list_status",
        # rm dates
        "earliest_cr_line",
        "issue_d",
        "last_pymnt_d",
        "last_credit_pull_d",
        # rm hardship
        "hardship_flag",
        "debt_settlement_flag",
        "pymnt_plan",
        "revol_util",
    ],
    inplace=True,
)
accepted_df = accepted_df.dropna(thresh=accepted_df.shape[0] * 0.8, axis=1)

# %%
# start cleaning details
accepted_df["int_rate"] = accepted_df["int_rate"].apply(lambda x: float(x[:-1]))
accepted_df = accepted_df.reset_index(drop=True)
accepted_df

# %%
# encoding
data_with_loanstatus_sliced = accepted_df[
    accepted_df["loan_status"].isin(
        [
            "Charged Off",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (31-120 days)",
            "Late (16-30 days)",
            "Default",
            "Fully Paid",
            "Does not meet the credit policy. Status:Fully Paid",
        ]
    )
]
di = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
    "Default": 1,
}  # converting target variable to boolean
accepted_df = data_with_loanstatus_sliced.replace({"loan_status": di})
accepted_df["grade"] = accepted_df["grade"].map(
    {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
)
accepted_df["home_ownership"] = accepted_df["home_ownership"].map(
    {"MORTGAGE": 6, "RENT": 5, "OWN": 4, "OTHER": 3, "NONE": 2, "ANY": 1}
)
accepted_df["emp_length"] = accepted_df["emp_length"].fillna(0)
# accetped_df["emp_length"].unique()
accepted_df["emp_length"] = accepted_df["emp_length"].map(
    {
        "10+ years": 10,
        "< 1 year": 0,  # or 0.1,
        "1 year": 1,
        "3 years": 3,
        "8 years": 8,
        "9 years": 9,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "2 years": 2,
        "7 years": 7,
    }
)
accepted_df


# %%
# function to remove 'months' string from the 'term' column and convert it to numeric
def loan_term_converter(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(" months", ""))


loan_term_converter(accepted_df, "term")
accepted_df
# %%
accepted_df["verification_status"] = accepted_df["verification_status"].map(
    {
        "Verified": 1,
        "Source Verified": 1,
        "Not Verified": 0,
    }
)
accepted_df
# %%

# remove all nan
accepted_df.fillna(0, inplace=True)

# %%
accepted_df = pd.get_dummies(
    accepted_df, columns=["purpose", "addr_state"], dtype=float
)
accepted_df

# %%
import tensorflow as tf

target = accepted_df.pop("loan_status")
target = tf.convert_to_tensor(target)
target

# %%
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(accepted_df)
accepted_tensor = normalizer(accepted_df)
accepted_tensor
# %%
from sklearn.model_selection import train_test_split

y = target.numpy()
X = accepted_tensor.numpy()
X_train, X_test_and_eval, y_train, y_test_and_eval = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test, X_eval, y_test, y_eval = train_test_split(
    X_test_and_eval, y_test_and_eval, test_size=0.5, random_state=42
)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
LRModel = LogisticRegression(max_iter=10000)
LRModel.fit(X_train, y_train)
y_pred = LRModel.predict(X_test)
print(f"accuracy_score {accuracy_score(y_test, y_pred)}")
print(f"recall_score {recall_score(y_test, y_pred)}")
print(f"f1_score macro {f1_score(y_test, y_pred, average='macro')}")
print(f"f1_score micro {f1_score(y_test, y_pred, average='micro')}")


# %%
import os

os.chdir("/home/ta/creditium")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    add,
    concatenate,
    Conv1D,
    Conv2D,
    Dropout,
    BatchNormalization,
    Flatten,
    MaxPooling2D,
    AveragePooling1D,
    AveragePooling2D,
    Activation,
    Dropout,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping


def cnn_2layer_fc_model_1d(
    n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28,)
) -> Model:
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 1:
        y = Reshape((input_shape[0], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv1D(
        filters=n1, kernel_size=(3,), strides=1, padding="same", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling1D(pool_size=(2,), strides=1, padding="same")(y)

    y = Conv1D(
        filters=n2, kernel_size=(3,), strides=2, padding="valid", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(
        units=n_classes,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    recall = tf.keras.metrics.Recall(thresholds=0)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", recall],
    )
    return model_A


# test 2_layer_cnn
# model = cnn_2layer_fc_model_1d(1, input_shape=(135,))
# model.summary()

f1 = tf.keras.metrics.F1Score()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
model.compile(
   optimizer="adam",
   loss='bce', 
    metrics=["accuracy", recall],)

# %%
# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
#     metrics=["accuracy", recall],
# )


history = model.fit(
    tf.convert_to_tensor(X_train),
    tf.convert_to_tensor(y_train),
    epochs=10,
    validation_data=(tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)),
)

result = model.evaluate(X_eval, y_eval, verbose=2)
result
# %%

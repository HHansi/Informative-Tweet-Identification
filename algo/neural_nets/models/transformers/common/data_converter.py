# Created by Hansi at 7/3/2020
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


def encode(data):
    return le.fit_transform(data)


def decode(data):
    return le.inverse_transform(data)

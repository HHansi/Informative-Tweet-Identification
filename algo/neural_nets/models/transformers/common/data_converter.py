# Created by Hansi at 7/3/2020
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}


def encode_le(data):
    return le.fit_transform(data)


def decode_le(data):
    return le.inverse_transform(data)


def encode(data):
    return [label_map[row] for row in data]


def decode(data):
    decode_label_map = dict()
    for key, value in label_map.items():
        decode_label_map[value] = key
    return [decode_label_map[row] for row in data]


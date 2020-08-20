# Created by Hansi at 7/3/2020
from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

label_map = {'UNINFORMATIVE': 0, 'INFORMATIVE': 1}
decode_label_map = {0: 'UNINFORMATIVE', 1: 'INFORMATIVE'}

#
# def encode_le(data):
#     return le.fit_transform(data)
#
#
# def decode_le(data):
#     return le.inverse_transform(data)


def encode(data):
    return [label_map[row] for row in data]


def decode(data):
    print("decoding data")
    print("data: ", data)
    # decode_label_map = dict()
    # for key, value in label_map.items():
    #     decode_label_map[value] = key
    decoded_data = [decode_label_map[row] for row in data]
    print('decoded data: ', decoded_data)
    return decoded_data



if __name__ == "__main__":
    a = [0.0000, 1.0000, 1.0000]
    print(decode(a))
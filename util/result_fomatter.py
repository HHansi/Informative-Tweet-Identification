# Created by Hansi at 8/12/2020
import csv

import pandas as pd


def format(input_file_path, output_file_path):
    # input_file = open(input_file_path, 'r', newline='', encoding='utf-8')
    # input_reader = csv.reader(input_file, delimiter='\t')

    input = pd.read_csv(input_file_path, sep='\t', encoding='utf-8')
    preds = input["predictions"]

    output_file = open(output_file_path, 'w', encoding='utf-8')
    for pred in preds:
        output_file.write(pred + '\n')
    output_file.close()


if __name__ == "__main__":
    input_file_path = "../results/dev_result.tsv"
    output_file_path = "../results/predictions.txt"
    format(input_file_path, output_file_path)
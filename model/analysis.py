import json
import os
import random

import numpy as np


def get_filepaths(directory,extension="json"):
    filepaths = []
    for root,dirs,files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root,_file))
    print("size of filepaths = ",len(filepaths))
    return filepaths

def analysis_label(converted_data_path):
    filepaths = get_filepaths(converted_data_path)
    label_sum = np.zeros(shape=(3))
    for idx,path in enumerate(filepaths):
        json_data = json.load(open(path))
        for single_data in json_data:
            win_side = single_data['win_side']
            label_sum[win_side + 1] += 1
        print(f"{idx + 1}/{len(filepaths)}")
    print(label_sum)
    print(label_sum / np.sum(label_sum))

def balance_label(converted_data_path,max_cnt=30012):
    filepaths = get_filepaths(converted_data_path)
    random.shuffle(filepaths)
    cnt = 0
    for idx, path in enumerate(filepaths):
        json_data = json.load(open(path))
        win_side = json_data[0]['win_side']
        if win_side == 0:
            cnt += 1
            if cnt > max_cnt:
                os.remove(path)
        print(f"{idx + 1}/{len(filepaths)}")

if __name__ == "__main__":
    analysis_label(converted_data_path="../dump")
    #balance_label(converted_data_path="../dump")
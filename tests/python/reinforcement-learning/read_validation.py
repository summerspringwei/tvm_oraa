import numpy

def read_validation_from_log(file_path):
    data_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find("validation p-rmse") >= 0:
                value = float(line.split(" ")[-1])
                data_arr.append(value)
    return data_arr


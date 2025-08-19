import csv
from variables import * # need variables for filename

data = [
    ['id', 'name', 'price', 'amount'],
    ['1', 'apple', '5000', '5'],
    ['2', 'pencil', '500', '42'],
    ['3', 'pineapple', '8000', '5'],
    ['4', 'pen', '1500', '10']
]



def write_csv_row(filename, datarow):
    with open('csv/%s.csv'%filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(datarow)

def append_csv_row(filename, datarow):
    with open('csv/%s.csv'%filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(datarow)


def write_csv_all(filename, data):
    with open('csv/%s.csv'%filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def save_recovery_data_row_csv(datarow):
    global codeword_len, databit_num
    filename = "n_{}_k_{}".format(codeword_len, databit_num) # _pooling_{}_thr_{}
    append_csv_row(filename, datarow)


def init_header(header_info):
    global codeword_len, databit_num
    filename = "n_{}_k_{}".format(codeword_len, databit_num) # _pooling_{}_thr_{}
    write_csv_row(filename, header_info)









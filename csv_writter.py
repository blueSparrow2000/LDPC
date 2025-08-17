import csv

data = [
    ['id', 'name', 'price', 'amount'],
    ['1', 'apple', '5000', '5'],
    ['2', 'pencil', '500', '42'],
    ['3', 'pineapple', '8000', '5'],
    ['4', 'pen', '1500', '10']
]


def format_data(data):
    return data

def write_csv(filename, data):
    with open('csv/%s.csv'%filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


write_csv("test", data)












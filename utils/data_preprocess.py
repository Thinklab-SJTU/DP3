import pandas as pd
import scipy
from scipy import io
import time


def scale_time(source_csv_file, target_csv_file, scale):
    df = pd.read_csv(source_csv_file)
    df.loc[:, 'time'] = df.loc[:, 'time'] / scale
    df.to_csv(target_csv_file, index=False)


def mat2csv(mat_file, csv_file):
    data = scipy.io.loadmat(mat_file)
    with open(csv_file, 'w') as fout:
        fout.write('id,time,event\n')
        sequence_id = 1
        print("starting...")
        for k, v in data.items():
            if 'patient' in k:
                for idx in range(data[k].shape[1]):
                    s = str(sequence_id) + ','
                    if len(data[k][1][idx].tolist()) != 0:
                        st = str(data[k][1][idx].tolist()[0])
                        st = '1' + st[1:]
                        s += (str(
                            time.mktime(time.strptime(
                                st, "%Y-%m-%d %H:%M:%S"))) + ',')
                    else:
                        s += 'N,'
                    if len(data[k][0][idx].tolist()) != 0:
                        s += (str(data[k][0][idx].tolist()[0]) + '\n')
                    else:
                        s += '8\n'
                    fout.write(s)
                sequence_id += 1
                if sequence_id % 1000 == 0:
                    print(sequence_id)


def csv_split(data, sequence_index, domain):
    """
    from csv to txt, generate file for each field
    """
    for field_name, file_name in domain.items():
        series = data.groupby(
            sequence_index)[field_name].apply(lambda x: x.tolist())
        series_data = []
        for idx in series.index:
            series_data.append(','.join(str(value)
                                        for value in series[idx]) + '\n')
        train_data_num = int(0.7 * len(series_data))
        with open(file_name + "-train.txt", 'w') as fout:
            fout.writelines(series_data[:train_data_num])
        with open(file_name + "-test.txt", 'w') as fout:
            fout.writelines(series_data[train_data_num:])
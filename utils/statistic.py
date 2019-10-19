import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


def dataset_statistic(csv_file, domain_dict, dataset_name, dataset_dir):
    """
        dataset statistic need csv file and domain_dict (describe each column of csv)
    """
    dataset = pd.read_csv(csv_file, encoding='utf-8').sort_values(
        by=[domain_dict['id'], domain_dict['timestamp']],
        ascending=[True, True]).reset_index(drop=True)
    last_id = None
    last_timestamp = None
    sequences_length = {}
    event_interval = []
    event_length = []
    event_type_dict = {}
    length = 0
    for _, row in dataset.iterrows():
        id = row[domain_dict['id']]
        timestamp = row[domain_dict['timestamp']]
        event_type = row[domain_dict['event']]
        if last_id is None or last_id != id:
            last_id = id
            if length != 0:
                sequences_length[
                    length] = 1 if length not in sequences_length.keys(
                    ) else sequences_length[length] + 1
                event_length.append(length)
            length = 1
            last_timestamp = timestamp
        else:
            if timestamp - last_timestamp > 0.01:
                event_interval.append(timestamp - last_timestamp)
            last_timestamp = timestamp
            length += 1
        if event_type not in event_type_dict.keys():
            event_type_dict[event_type] = 1
        else:
            event_type_dict[event_type] += 1
    sequences_length[length] = 1 if length not in sequences_length.keys(
    ) else sequences_length[length] + 1
    event_interval = np.log10(np.array(event_interval))
    plt.hist(event_interval,
             bins=80,
             facecolor='green',
             edgecolor='black',
             alpha=0.7)
    plt.xlabel('log event interval')
    plt.ylabel('number')
    # plt.grid(True)
    plt.savefig(dataset_dir + '/event_interval_statistic.png')

    event_length = np.log10(np.array(event_length))
    plt.hist(event_length,
             bins=20,
             facecolor='green',
             edgecolor='black',
             alpha=0.7)
    plt.xlabel('sequence length')
    plt.ylabel('number')
    plt.savefig(dataset_dir + '/sequence_length_statistic.png')

    statistic_dict = {
        'event_type_dict': event_type_dict,
        'event_interval': event_interval,
        'sequences_length': sequences_length
    }
    with open(dataset_dir + '/statistic.json', 'wb') as fout:
        pickle.dump(statistic_dict, fout)
    print('dataset ' + dataset_name + ' statistic:')
    print('event_type_statistic:\n', event_type_dict)
    print('number of event type:\t', len(event_type_dict))
    print('sequences_length_statistic:\n', sequences_length)


if __name__ == '__main__':
    domain_dict = {'id': 'id', 'timestamp': 'time', 'event': 'event'}
    dataset_statistic('../data/real/atm/atm_day.csv', domain_dict, 'atm',
                      '../data/real/atm')
    statistic_dict = pickle.load(open('../data/real/atm/statistic.json', 'rb'))
    print(len(statistic_dict['event_type_dict']))

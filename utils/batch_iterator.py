import numpy as np
import random
from .preprocess import load_sequences


class PaddedBatchIterator(object):
    def __init__(self, sequences, mark=False, diff=False, save_last_time=False):
        self.sequences = sequences
        self.mark = mark
        self.diff = diff
        self.save_last_time = save_last_time
        self.size = len(self.sequences)
        self.cursor = None
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.sequences)
        self.cursor = 0

    def next_batch(self, batch_size):
        end = False
        if self.cursor + batch_size >= self.size:
            batch_size = self.size - self.cursor
            end = True
        sequences = self.sequences[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        sequences = sorted(sequences, key=lambda item: len(item), reverse=True)
        length = [len(sequence) - 1 for sequence in sequences]
        max_length = max(length)
        last_time_sequence = None
        if self.save_last_time:
            if self.diff is True:
                last_time_sequence = np.array([sequence[-2][0] - sequence[0][0] for sequence in sequences])
            else:
                # accumulate time
                last_time_sequence = []
                for sequence in sequences:
                    time = 0
                    for item in sequence[:-1]:
                        time += item[0]
                    last_time_sequence.append(time)
                last_time_sequence = np.array(last_time_sequence)
        if self.mark is True:
            batch_sequences = np.zeros([batch_size, max_length, 2], dtype=np.float32)
            if self.diff:
                target = np.array([[sequence[-1][0] - sequence[-2][0], sequence[-1][1]] for sequence in sequences])
            else:
                target = np.array([[sequence[-1][0], sequence[-1][1]] for sequence in sequences])
        else:
            batch_sequences = np.zeros([batch_size, max_length, 1], dtype=np.float32)
            if self.diff:
                target = np.array([[sequence[-1][0] - sequence[-2][0]] for sequence in sequences])
            else:
                target = np.array([[sequence[-1][0], sequence[-1][1]] for sequence in sequences])
        for idx, batch_sequence in enumerate(batch_sequences):
            if self.mark is True:
                batch_sequence[:length[idx], :] = sequences[idx][:-1]
            else:
                batch_sequence[:length[idx], 0] = sequences[idx][:-1]

        if self.diff is True:
            if self.mark is True:
                batch_time_sequences = np.concatenate([batch_sequences[:, 0:1, 0:1] * 0,
                                                       np.diff(batch_sequences[:, :, 0:1], axis=1)], axis=1)
                batch_sequences = np.concatenate([batch_time_sequences, batch_sequences[:, :, 1:]], axis=2)
            else:
                batch_sequences = np.concatenate([batch_sequences[:, 0:1, 0:] * 0, np.diff(batch_sequences, axis=1)],
                                                 axis=1)
            for idx, sl in enumerate(length):
                if sl < max_length:
                    batch_sequences[idx, sl, 0] = 0.0
        return end, batch_sequences, target, last_time_sequence, np.array(length)


if __name__ == '__main__':
    time_sequences_file = '../data/real/ATM/dev_time.txt'
    event_sequences_file = '../data/real/ATM/dev_event.txt'
    sequences = load_sequences(time_sequences_file, event_sequences_file)
    data_batch_iterator = PaddedBatchIterator(sequences, True, False, True)
    data_batch_iterator.shuffle()
    batch_size = 10
    print(data_batch_iterator.next_batch(batch_size))


import numpy as np
np.random.seed(11)


class MDHawkesProcess(object):
    def __init__(self, dim, beta):
        self.dim = dim
        self.beta = beta

    def mle_full(self,
                 sequences,
                 T,
                 max_iteration=5000,
                 eps=1e-6,
                 iteration_threshold=50,
                 true_alpha=None):
        mu = np.random.rand(self.dim)
        alpha = np.random.rand(self.dim, self.dim)
        cnt = 0
        for iteration in range(max_iteration):
            p_list = []
            # E Step:
            for sequence in sequences:
                p = np.zeros((len(sequence), len(sequence)), dtype=np.float64)
                for idx, item in enumerate(sequence):
                    p[idx, idx] = mu[item[1]]
                    for j in range(idx):
                        p[idx, j] = alpha[item[1], sequence[j]
                                          [1]] * self.beta * np.exp(
                                              -1 * self.beta *
                                              (item[0] - sequence[j][0]))
                    p[idx] /= p[idx].sum()
                p_list.append(p)

            # M Step:
            next_mu = np.zeros(self.dim, dtype=np.float64)
            next_alpha = np.zeros((self.dim, self.dim), dtype=np.float64)
            G = np.zeros(self.dim, dtype=np.float64)
            for c, sequence in enumerate(sequences):
                for idx, item in enumerate(sequence):
                    next_mu[item[1]] += p_list[c][idx, idx]
                    G[item[1]] += (1 - np.exp(-1 * self.beta *
                                              (T[c] - item[0])))
                    for j in range(idx):
                        next_alpha[item[1], sequence[j][1]] += p_list[c][idx,
                                                                         j]

            next_mu /= T.sum()
            for d in range(self.dim):
                if G[d] != 0:
                    next_alpha[:, d] = next_alpha[:, d] / G[d]

            if ((np.abs(next_mu - mu).sum() + np.abs(next_alpha - alpha).sum())
                    / (self.dim * (self.dim + 1))) < eps:
                cnt += 1
                if cnt > iteration_threshold:
                    break
            else:
                cnt = 0

            mu = next_mu
            alpha = next_alpha

            if true_alpha is not None and iteration % 10 == 9:
                print("iteration %d: " % iteration,
                      self.relErr(true_alpha, alpha))
        return mu, alpha

    def mle_lowrank_sparse(self,
                           sequences,
                           T,
                           low_rank=True,
                           sparse=True,
                           max_admm_iteration=5000,
                           max_mm_iteration=500,
                           penalty_p=0.5,
                           lamda1=0.02,
                           lamda2=0.6,
                           eps=1e-6,
                           iteration_threshold=20,
                           true_alpha=None):
        mu = np.random.rand(self.dim)
        alpha = np.random.rand(self.dim, self.dim)

        if low_rank:
            u1 = np.zeros((self.dim, self.dim), dtype=np.float64)
            u, s, v = np.linalg.svd(alpha + u1, full_matrices=True)
            s = s - (lamda1 / penalty_p)
            s[s < 0] = 0
            s = np.diag(s)
            z1 = np.dot(u, np.dot(s, v))
        if sparse:
            u2 = np.zeros((self.dim, self.dim), dtype=np.float64)
            z2 = np.zeros((self.dim, self.dim), dtype=np.float64)
            au = alpha + u2
            upper_value = lamda2 / penalty_p
            for i in range(self.dim):
                for j in range(self.dim):
                    if au[i, j] > upper_value:
                        z2[i, j] = au[i, j] - upper_value
                    elif au[i, j] < -upper_value:
                        z2[i, j] = au[i, j] + upper_value

        admm_cnt = 0
        for admm_iteration in range(max_admm_iteration):
            # update mu, alpha
            # E step
            cnt = 0
            last_mu = mu
            last_alpha = alpha
            for mm_iteration in range(max_mm_iteration):
                p_list = []
                for sequence in sequences:
                    p = np.zeros((len(sequence), len(sequence)),
                                 dtype=np.float64)
                    for idx, item in enumerate(sequence):
                        p[idx, idx] = mu[item[1]]
                        for j in range(idx):
                            p[idx, j] = alpha[item[1], sequence[j]
                                              [1]] * self.beta * np.exp(
                                                  -1 * self.beta *
                                                  (item[0] - sequence[j][0]))
                        p[idx] /= p[idx].sum()
                    p_list.append(p)

                # M step
                B = np.zeros((self.dim, self.dim), dtype=np.float64)
                if low_rank:
                    B += penalty_p * (-z1 + u1)
                if sparse:
                    B += penalty_p * (-z2 + u2)
                C = np.zeros((self.dim, self.dim), dtype=np.float64)

                next_mu = np.zeros(self.dim, dtype=np.float64)
                for c, sequence in enumerate(sequences):
                    for idx, item in enumerate(sequence):
                        B[:, item[1]] += (1 - np.exp(-1 * self.beta *
                                                     (T[c] - item[0])))
                        next_mu[item[1]] += p_list[c][idx][idx]
                        for j in range(idx):
                            C[item[1], sequence[j][1]] += p_list[c][idx][j]
                next_mu /= T.sum()
                next_alpha = (-B + np.sqrt(B**2 + 8 * penalty_p * C)) / (
                    4 * penalty_p)

                if ((np.abs(next_mu - mu).sum() +
                     np.abs(next_alpha - alpha).sum()) /
                    (self.dim * (self.dim + 1))) < eps:
                    cnt += 1
                    if cnt > iteration_threshold:
                        break
                else:
                    cnt = 0

                mu = next_mu
                alpha = next_alpha

            if true_alpha is not None:
                print("admm_iteration %d: " % admm_iteration,
                      self.relErr(true_alpha, alpha))

            if ((np.abs(last_mu - mu).sum() + np.abs(last_alpha - alpha).sum())
                    / (self.dim * (self.dim + 1))) < eps:
                admm_cnt += 1
                if admm_cnt > iteration_threshold:
                    return mu, alpha
            else:
                admm_cnt = 0

            # update u1, z1, u2, z2
            if low_rank:
                u, s, v = np.linalg.svd(alpha + u1, full_matrices=True)
                s = s - (lamda1 / penalty_p)
                s[s < 0] = 0
                s = np.diag(s)
                z1 = np.dot(u, np.matmul(s, v))
                u1 = u1 + alpha - z1
            if sparse:
                z2 = np.zeros((self.dim, self.dim), dtype=np.float64)
                au = alpha + u2
                upper_value = lamda2 / penalty_p
                for i in range(self.dim):
                    for j in range(self.dim):
                        if au[i, j] > upper_value:
                            z2[i, j] = au[i, j] - upper_value
                        elif au[i, j] < -upper_value:
                            z2[i, j] = au[i, j] + upper_value
                u2 = u2 + alpha - z2
        return mu, alpha

    def simulation(self, mu, alpha, T, max_length=1000):
        sequence = []
        sequence_str = []
        s = 0
        length = 0
        while True:
            lam = self.intensity_value(mu, alpha, sequence, s).sum()
            s = s + (-1 * np.log(np.random.uniform(0, 1)) / lam)
            if s > T:
                break
            Dlam = np.random.uniform(0, 1) * lam
            lams = self.intensity_value(mu, alpha, sequence, s)
            lams_cumsum = np.cumsum(lams)
            idx = np.searchsorted(lams_cumsum, Dlam, 'left')
            if idx >= self.dim:
                continue
            sequence.append([s, idx])
            sequence_str.append('%.8f %d' % (float(s), int(idx)))
            length += 1
            if length >= max_length:
                break
        return sequence_str

    def predict(self, mu, alpha, history, sequence_length=10):
        sequence = history[:]
        result = []
        s = sequence[-1][0]
        length = 0
        while True:
            lam = self.intensity_value(mu, alpha, sequence, s).sum()
            s = s + (-1 * np.log(np.random.uniform(0, 1)) / lam)
            Dlam = np.random.uniform(0, 1) * lam
            lams = self.intensity_value(mu, alpha, sequence, s)
            lams_cumsum = np.cumsum(lams)
            idx = np.searchsorted(lams_cumsum, Dlam, 'left')
            if idx >= self.dim:
                continue
            sequence.append([s, idx])
            result.append([s, idx])
            length += 1
            if length > sequence_length:
                break
        return result

    def intensity_value(self, mu, alpha, sequence, t):
        lams = np.array(mu)
        for d in range(self.dim):
            for idx, item in enumerate(sequence):
                if item[0] > t:
                    break
                lams[d] += (alpha[d, item[1]] * np.exp(-1 * self.beta *
                                                       (t - item[0])))
        return lams

    def relErr(self, alpha, p_alpha):
        error = 0
        for idx in range(self.dim):
            for j in range(self.dim):
                if alpha[idx][j] == 0:
                    error += (np.abs(alpha[idx][j] - p_alpha[idx][j]))
                else:
                    error += (np.abs(alpha[idx][j] - p_alpha[idx][j]) /
                              np.abs(alpha[idx][j]))
        return error / (self.dim**2)


def set_parameters(dim, parameters_file=None):
    if parameters_file is None:
        # mu = np.random.uniform(0, 0.01, dim)
        # U = np.zeros((dim, 9), dtype=np.float64)
        # V = np.zeros((dim, 9), dtype=np.float64)
        # for i in range(9):
        #     U[i:i + 2, i] = np.random.uniform()
        #     V[i:i + 2, i] = np.random.rand(2) * 0.7
        # alpha = np.dot(U, V.T)
        mu = np.random.uniform(0, 0.01, dim)
        alpha = np.random.uniform(0, 0.1, (dim, dim))
        for i in range(dim):
            for j in range(dim):
                if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:
                    alpha[i, j] = 0
    else:
        mu = []
        alpha = []
        with open(parameters_file, 'r') as mu_alpha:
            for line in mu_alpha.readlines():
                if len(mu) == 0:
                    mu.append(list(map(float, line.strip().split(','))))
                else:
                    alpha.append(list(map(float, line.strip().split(','))))
        mu = np.array(mu, dtype=np.float64).reshape(dim)
        alpha = np.array(alpha, dtype=np.float64)
    return mu, alpha


def generate_sequences(hawkes, sequences_file, N, mu, alpha, T, min_length,
                       max_length):
    length = []
    with open(sequences_file, 'w') as fout:
        idx = 0
        while True:
            sequence = hawkes.simulation(mu, alpha, T, max_length)
            # print(len(sequence))
            if len(sequence) > min_length:
                fout.write(','.join(sequence) + '\n')
                length.append(length)
                idx += 1

                if idx >= N:
                    break


def train(hawkes, train_sequences_file, true_alpha=None):
    sequences = []
    T = []
    with open(train_sequences_file, 'r') as fin:
        for line in fin.readlines():
            sequence = line.strip().split(',')
            T.append(float(sequence[-1].split()[0]) + 1)
            sequence = [[float(item.split()[0]),
                         int(item.split()[1])] for item in sequence]
            sequences.append(sequence)
    print('sequence length: ', len(sequences))

    T = np.array(T)
    # p_mu, p_alpha = hawkes.mle_full(sequences, T, true_alpha=true_alpha)
    p_mu, p_alpha = hawkes.mle_lowrank_sparse(sequences,
                                              T,
                                              true_alpha=true_alpha)
    print('p_mu: ', p_mu)
    print('p_alpha: ', p_alpha)
    return p_mu, p_alpha


def predict(hawkes, test_sequences_file, mu, alpha, predict_length=1):
    cnt = 0
    acc_event = 0
    time_err = 0
    with open(test_sequences_file, 'r') as fin:
        for line in fin.readlines():
            sequence = line.strip().split(',')
            sequence = [[float(item.split()[0]),
                         int(item.split()[1])] for item in sequence]
            result = hawkes.predict(mu, alpha, sequence[:-predict_length],
                                    predict_length)
            cnt += predict_length
            for idx in range(1, predict_length + 1):
                if result[-idx][1] == sequence[-idx][1]:
                    acc_event += 1
                time_err += abs(result[-idx][0] - sequence[-idx][0])
    print('number of test data: ', cnt)
    print('acc of event prediction: ', acc_event / cnt)
    print('time error of time prediction: ', time_err / cnt)


if __name__ == '__main__':
    dim = 20
    beta = 0.01
    parameters_file = 'mu_alpha.txt'
    mu, alpha = set_parameters(dim)
    print(mu.shape)
    print(alpha.shape)
    print('mu: ', mu)
    print('alpha: ', alpha)

    eigenvalue, featurevector = np.linalg.eig(alpha)
    print('spectral radius: ', np.max(np.abs(eigenvalue)))

    train_N = 4000
    test_N = 1000
    hawkes = MDHawkesProcess(dim, beta)
    train_sequences_file = 'train_hawkes.txt'
    test_sequences_file = 'test_hawkes.txt'
    train_T = 10
    test_T = 10
    min_length = 30
    max_length = 200
    predict_length = 1

    # sample
    # generate_sequences(hawkes, train_sequences_file, train_N, mu, alpha, train_T, min_length, max_length)
    # generate_sequences(hawkes, test_sequences_file, test_N, mu, alpha, test_T, min_length, max_length)

    # train
    p_mu, p_alpha = train(hawkes, train_sequences_file, alpha)
    print('relErr: ', hawkes.relErr(alpha, p_alpha))

    # predict
    predict(hawkes, test_sequences_file, p_mu, p_alpha, predict_length)

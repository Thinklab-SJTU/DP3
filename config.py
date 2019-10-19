import logging
from configparser import ConfigParser
import os


class BaseConfigurator(object):
    def __init__(self, config):
        self._config = config
        if not os.path.isdir(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)

    @property
    def DATASET_NAME(self):
        return self._config.get('Data', 'dataset_name')

    @property
    def DATASET_DIR(self):
        return self._config.get('Data', 'dataset_dir')

    @property
    def DOMAIN_DICT(self):
        return self._config.get('Data', 'domain_dict')

    @property
    def CSV_FILE(self):
        return self._config.get('Data', 'csv_file')

    @property
    def STATISTIC_FILE(self):
        return self._config.get('Data', 'statistic_file')

    @property
    def EVENT_INTERVAL_STATISTIC_PICTURE(self):
        return self._config.get('Data', 'event_interval_statistic_picture')

    @property
    def EVENT_INDEX_FILE(self):
        return self._config.get('Data', 'event_index_file')

    @property
    def MIN_EVENT_INTERVAL(self):
        return self._config.getfloat('Data', 'min_event_interval')

    @property
    def MIN_LENGTH(self):
        return self._config.getint('Data', 'min_length')

    @property
    def MAX_LENGTH(self):
        return self._config.getint('Data', 'max_length')

    @property
    def TRAIN_RATE(self):
        return self._config.getfloat('Data', 'train_rate')

    @property
    def TIME_FILE(self):
        return self._config.get('Data', 'time_file')

    @property
    def EVENT_FILE(self):
        return self._config.get('Data', 'event_file')

    @property
    def TRAIN_TIME_FILE(self):
        return self._config.get('Data', 'train_time_file')

    @property
    def TRAIN_EVENT_FILE(self):
        return self._config.get('Data', 'train_event_file')

    @property
    def DEV_TIME_FILE(self):
        return self._config.get('Data', 'dev_time_file')

    @property
    def DEV_EVENT_FILE(self):
        return self._config.get('Data', 'dev_event_file')

    @property
    def TEST_TIME_FILE(self):
        return self._config.get('Data', 'test_time_file')

    @property
    def TEST_EVENT_FILE(self):
        return self._config.get('Data', 'test_event_file')

    @property
    def MARK(self):
        return self._config.getboolean('Data', 'mark')

    @property
    def DIFF(self):
        return self._config.getboolean('Data', 'diff')

    @property
    def SAVE_LAST_TIME(self):
        return self._config.getboolean('Data', 'save_last_time')

    @property
    def SAVE_DIR(self):
        return self._config.get('Save', 'save_dir')

    @property
    def PRED_DEV(self):
        return self._config.get('Save', 'pred_dev')

    @property
    def PRED_TEST(self):
        return self._config.get('Save', 'pred_test')

    @property
    def BEST_MODEL(self):
        return self._config.get('Save', 'best_model')

    @property
    def LAST_MODEL(self):
        return self._config.get('Save', 'last_model')

    @property
    def LOG_FILE(self):
        return self._config.get('Save', 'log_file')

    @property
    def MODEL_FILE(self):
        return self._config.get('Load', 'model_file')

    @property
    def EPOCHS(self):
        return self._config.getint('Run', 'epochs')

    @property
    def TRAIN_BATCH_SIZE(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def TRAIN_PRINT_FREQ(self):
        return self._config.getint('Run', 'train_print_freq')

    @property
    def DEV_BATCH_SIZE(self):
        return self._config.getint('Run', 'dev_batch_size')

    @property
    def TEST_BATCH_SIZE(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def EPOCH_THRESHOLD(self):
        return self._config.getint('Run', 'epoch_threshold')

    @property
    def PYTORCH_SEED(self):
        return self._config.getint('Pytorch', 'pytorch_seed')


class ERPPConfigurator(BaseConfigurator):
    def __init__(self, config_file, extra_args=None):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(ERPPConfigurator, self).__init__(self._config)
        config.write(open(config_file, 'w'))
        for section in config.sections():
            print('[' + section + ']')
            for k, v in config.items(section):
                print(k + ' ' + v)
        print('load config file successfully.')

    @property
    def EVENT_CLASSES(self):
        return self._config.getint('Network', 'event_classes')

    @property
    def EMB_DROPOUT(self):
        return self._config.getfloat('Network', 'emb_dropout')

    @property
    def EMB_DIM(self):
        return self._config.getint('Network', 'emb_dim')

    @property
    def RNN_HIDDEN_DIM(self):
        return self._config.getint('Network', 'rnn_hidden_dim')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def MLP_DIM(self):
        return self._config.getint('Network', 'mlp_dim')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def EPS(self):
        return self._config.getfloat('Optimizer', 'eps')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')

    @property
    def WEIGHT_DECAY(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def LOSS_ALPHA(self):
        return self._config.getfloat('Loss', 'loss_alpha')


class AERPPConfigurator(BaseConfigurator):
    def __init__(self, config_file, extra_args=None):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(AERPPConfigurator, self).__init__(self._config)
        config.write(open(config_file, 'w'))
        for section in config.sections():
            print('[' + section + ']')
            for k, v in config.items(section):
                print(k + ' ' + v)
        print('load config file successfully.')

    @property
    def EVENT_CLASSES(self):
        return self._config.getint('Network', 'event_classes')

    @property
    def EMB_DROPOUT(self):
        return self._config.getfloat('Network', 'emb_dropout')

    @property
    def EMB_DIM(self):
        return self._config.getint('Network', 'emb_dim')

    @property
    def RNN_HIDDEN_DIM(self):
        return self._config.getint('Network', 'rnn_hidden_dim')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def THRESHOLD(self):
        return self._config.getfloat('Network', 'threshold')

    @property
    def MLP_DIM(self):
        return self._config.getint('Network', 'mlp_dim')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def EPS(self):
        return self._config.getfloat('Optimizer', 'eps')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')

    @property
    def WEIGHT_DECAY(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def LOSS_ALPHA(self):
        return self._config.getfloat('Loss', 'loss_alpha')


class RMTPPConfigurator(BaseConfigurator):
    def __init__(self, config_file, extra_args=None):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(RMTPPConfigurator, self).__init__(self._config)
        config.write(open(config_file, 'w'))
        for section in config.sections():
            print('[' + section + ']')
            for k, v in config.items(section):
                print(k + ' ' + v)
        print('load config file successfully.')

    @property
    def EVENT_CLASSES(self):
        return self._config.getint('Network', 'event_classes')

    @property
    def EMB_DROPOUT(self):
        return self._config.getfloat('Network', 'emb_dropout')

    @property
    def EMB_DIM(self):
        return self._config.getint('Network', 'emb_dim')

    @property
    def RNN_HIDDEN_DIM(self):
        return self._config.getint('Network', 'rnn_hidden_dim')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def MLP_DIM(self):
        return self._config.getint('Network', 'mlp_dim')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def EPS(self):
        return self._config.getfloat('Optimizer', 'eps')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')

    @property
    def WEIGHT_DECAY(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def LOSS_ALPHA(self):
        return self._config.getfloat('Loss', 'loss_alpha')


class ARMTPPConfigurator(BaseConfigurator):
    def __init__(self, config_file, extra_args=None):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(ARMTPPConfigurator, self).__init__(self._config)
        config.write(open(config_file, 'w'))
        for section in config.sections():
            print('[' + section + ']')
            for k, v in config.items(section):
                print(k + ' ' + v)
        print('load config file successfully.')

    @property
    def EVENT_CLASSES(self):
        return self._config.getint('Network', 'event_classes')

    @property
    def EMB_DROPOUT(self):
        return self._config.getfloat('Network', 'emb_dropout')

    @property
    def EMB_DIM(self):
        return self._config.getint('Network', 'emb_dim')

    @property
    def RNN_HIDDEN_DIM(self):
        return self._config.getint('Network', 'rnn_hidden_dim')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def THRESHOLD(self):
        return self._config.getfloat('Network', 'threshold')

    @property
    def MLP_DIM(self):
        return self._config.getint('Network', 'mlp_dim')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def EPS(self):
        return self._config.getfloat('Optimizer', 'eps')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')

    @property
    def WEIGHT_DECAY(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def LOSS_ALPHA(self):
        return self._config.getfloat('Loss', 'loss_alpha')

import logging
import argparse
import ast
from config import ERPPConfigurator, AERPPConfigurator, RMTPPConfigurator, ARMTPPConfigurator
import numpy as np
from scipy import integrate
import torch
from utils.statistic import dataset_statistic
from utils.sequence_generator import load_sequences, generate_sequence
from utils.batch_iterator import PaddedBatchIterator
import models


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', help='config file path')
    argparser.add_argument('--continue_training', action='store_true', help='load models continue training')
    argparser.add_argument('--model', help='ERPP, RMTPP, AERPP, ARMTPP')
    argparser.add_argument('--gpu', default='-1', help='gpu id (-1 to cpu)')
    argparser.add_argument('--mode', default='2', help='mode id(0: statistic, 1:preprocessing, 2: trianing, 3:testing)')
    args, extra_args = argparser.parse_known_args()

    cfg = None
    if args.model == 'ERPP':
        cfg = ERPPConfigurator(args.config_file, extra_args)
    elif args.model == 'AERPP':
        cfg = AERPPConfigurator(args.config_file, extra_args)
    elif args.model == 'RMTPP':
        cfg = RMTPPConfigurator(args.config_file, extra_args)
    elif args.model == 'ARMTPP':
        cfg = ARMTPPConfigurator(args.config_file, extra_args)

    # TODO:loging module
    #  设置log_file后记录所有训练输出
    #  初步是log_file由modelname+datasetname+timestap构成
    #  保证输出到日志同时打印到console
    # logger = logging.getLogger(args.model)
    # logger.setLevel(level=print)
    # handler = logging.FileHandler(cfg.LOG_FILE)
    # handler.setLevel(print)
    # LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    # formatter = logging.Formatter(LOG_FORMAT)
    # handler.setFormatter(formatter)
    # console = logging.StreamHandler()
    # console.setLevel(print)
    # 
    # logger.addHandler(handler)
    # logger.addHandler(console)

    if args.mode == '0':
        statistic(cfg, args, extra_args)
    elif args.mode == '1':
        preprocessing(cfg, args, extra_args)
    elif args.mode == '2':
        training(cfg, args, extra_args)
    elif args.mode == '3':
        testing(cfg, args, extra_args)


def statistic(cfg, args, extra_args):
    print('dataset statistic starting...')
    domain_dict = ast.literal_eval(cfg.DOMAIN_DICT)
    dataset_statistic(cfg.CSV_FILE, domain_dict, cfg.DATASET_NAME, cfg.DATASET_DIR)
    print('dataset statistic finished.')


def preprocessing(cfg, args, extra_args):
    print('preprocessing starting...')
    domain_dict = ast.literal_eval(cfg.DOMAIN_DICT)
    # generate_time_sequence(cfg.CSV_FILE, domain_dict, cfg.TIME_FILE,
    #                        cfg.TRAIN_TIME_FILE, cfg.DEV_TIME_FILE,
    #                        train_rate=cfg.TRAIN_RATE, min_length=cfg.MIN_LENGTH, max_length=cfg.MAX_LENGTH)
    # generate_event_sequence(cfg.CSV_FILE, domain_dict, cfg.EVENT_FILE, cfg.TRAIN_EVENT_FILE, cfg.DEV_EVENT_FILE,
    #                         cfg.EVENT_INDEX_FILE, train_rate=cfg.TRAIN_RATE,
    #                         min_length=cfg.MIN_LENGTH, max_length=cfg.MAX_LENGTH)
    generate_sequence(cfg.CSV_FILE, domain_dict,
                      cfg.TIME_FILE, cfg.TRAIN_TIME_FILE, cfg.DEV_TIME_FILE,
                      cfg.EVENT_FILE, cfg.TRAIN_EVENT_FILE, cfg.DEV_EVENT_FILE,
                      cfg.EVENT_INDEX_FILE, train_rate=cfg.TRAIN_RATE,
                      min_length=cfg.MIN_LENGTH, max_length=cfg.MAX_LENGTH, min_event_interval=cfg.MIN_EVENT_INTERVAL)
    print('preprocessing finished.')


def training(cfg, args, extra_args):
    print('training starting...')
    # pytorch setting
    if torch.cuda.is_available() and args.gpu != '-1':
        args.device = torch.device('cuda:' + str(args.gpu))
    else:
        args.device = torch.device('cpu')
    torch.manual_seed(cfg.PYTORCH_SEED)

    # load dataset
    train_sequences = load_sequences(cfg.TRAIN_TIME_FILE, cfg.TRAIN_EVENT_FILE)
    train_batch_iterator = PaddedBatchIterator(train_sequences, cfg.MARK, cfg.DIFF, cfg.SAVE_LAST_TIME)
    dev_sequences = load_sequences(cfg.DEV_TIME_FILE, cfg.DEV_EVENT_FILE)
    dev_batch_iterator = PaddedBatchIterator(dev_sequences, cfg.MARK, cfg.DIFF, cfg.SAVE_LAST_TIME)

    # load model
    # 1.直接重新初始化一个新的模型
    # 2.continue training,载入上次训练的模型来继续训练
    # model
    model = getattr(models, args.model)(cfg, args)
    if args.continue_training:
        checkpoint = torch.load(cfg.MODEL_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.to(args.device)

    # citeration
    citeration = getattr(models, args.model+'Loss')(cfg, args)

    # optimizer
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2), weight_decay=cfg.WEIGHT_DECAY)
    # citeration_optimizer = torch.optim.Adam(citeration.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2), weight_decay=cfg.WEIGHT_DECAY)
    model_optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.LEARNING_RATE, eps=cfg.EPS)
    model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=30, gamma=0.1)
    if cfg.SAVE_LAST_TIME:
        citeration_optimizer = torch.optim.RMSprop(citeration.parameters(), lr=cfg.LEARNING_RATE, eps=cfg.EPS)
        citeration_scheduler = torch.optim.lr_scheduler.StepLR(citeration_optimizer, step_size=30, gamma=0.1)

    # meters
    loss_meter = []
    max_event_f1 = None
    max_event_precision = None
    max_event_recall = None
    max_event_acc = None
    min_time_loss = None
    epoch_cnt = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        model_scheduler.step()
        if cfg.SAVE_LAST_TIME: citeration_scheduler.step()
        train_batch_iterator.shuffle()
        batch_id = 1
        while True:
            end, input, target, last_time, length = train_batch_iterator.next_batch(cfg.TRAIN_BATCH_SIZE)
            model_optimizer.zero_grad()
            if cfg.SAVE_LAST_TIME: citeration_optimizer.zero_grad()
            output = model.forward(input, length)
            batch_loss = citeration(output, target)
            loss_meter.append(batch_loss[2].item())
            batch_loss[2].backward()
            model_optimizer.step()
            if cfg.SAVE_LAST_TIME: citeration_optimizer.step()
            if batch_id % cfg.TRAIN_PRINT_FREQ == 0:
                print("epoch: %d\tbatch_id:%d\tloss:%f\ttime loss: %f\tevent loss: %f" %
                             (epoch, batch_id, np.array(loss_meter).mean(), batch_loss[0].item(), batch_loss[1].item()))
                loss_meter = []
            batch_id += 1
            if end: break

        model.eval()
        with torch.no_grad():
            event_total = 0
            all_cnt = np.zeros(cfg.EVENT_CLASSES)
            acc_cnt = np.zeros(cfg.EVENT_CLASSES)
            pre_cnt = np.zeros(cfg.EVENT_CLASSES)
            time_error = 0
            dev_batch_iterator.shuffle()
            while True:
                end, input, target, last_time, length = dev_batch_iterator.next_batch(cfg.DEV_BATCH_SIZE)
                output = model.forward(input, length)
                event_total += length.shape[0]

                event_output = output[1].cpu().numpy()
                event_output = np.argmax(event_output, axis=1).astype(int)
                event_target = target[:, 1].astype(int)
                for idx in range(event_target.shape[0]):
                    all_cnt[event_target[idx]] += 1
                    pre_cnt[event_output[idx]] += 1
                    if event_output[idx] == event_target[idx]:
                        acc_cnt[event_output[idx]] += 1

                if last_time is None:
                    time_output = output[0].squeeze().cpu().numpy()
                    time_target = target[:, 0]
                    time_error += np.abs(time_output - time_target).sum()
                else:
                    time_target = target[:,  0]
                    history_event = output[0].squeeze().cpu().numpy()
                    intensity_w = citeration.intensity_w.cpu().data.numpy()
                    intensity_b = citeration.intensity_b.cpu().data.numpy()
                    next_time = np.array([integrate.quad(lambda t: (t + last_time[idx]) * np.exp(history_event[idx] +
                                intensity_w * t + intensity_b + (np.exp(history_event[idx] + intensity_b) -
                                np.exp(history_event[idx] + intensity_w * t + intensity_b)) / intensity_w), 0, np.inf)[0]
                                 for idx in range(history_event.shape[0])])
                    time_error += np.abs(next_time - last_time - time_target).sum()
                if end: break
            print(acc_cnt, acc_cnt.sum())
            print(pre_cnt, pre_cnt.sum())
            print(all_cnt, all_cnt.sum())
            cnt = 0
            score = 0.0
            for idx in range(cfg.EVENT_CLASSES):
                if all_cnt[idx] != 0:
                    cnt += 1
                    score += acc_cnt[idx] / all_cnt[idx]
            event_recall = score / cnt
            cnt = 0
            score = 0.0
            for idx in range(cfg.EVENT_CLASSES):
                if pre_cnt[idx] != 0:
                    cnt += 1
                    score += acc_cnt[idx] / pre_cnt[idx]
            event_precision = score / cnt
            cnt = 0
            score = 0.0
            for idx in range(cfg.EVENT_CLASSES):
                if all_cnt[idx] != 0 and pre_cnt[idx] != 0:
                    cnt += 1
                    precision = acc_cnt[idx] / pre_cnt[idx]
                    recall = acc_cnt[idx] / all_cnt[idx]
                    score += ((2 * precision * recall) / (precision + recall))
            event_f1 = score / cnt
            event_acc = acc_cnt.sum() / all_cnt.sum()
            print("epoch: %d\tevent_recall: %f\tevent_precision: %f\tevent_f1: %f\ttime_error: %f" %
                  (epoch, event_recall, event_precision, event_f1, time_error / event_total))
            print("-------------------------------------------------------------------------------------")

            if max_event_f1 is None or event_f1 > max_event_f1:
                max_event_f1 = event_f1
                best_model = {'epoch': epoch, 'precision': event_precision, 'recall': event_recall, 'f1': event_f1}
                epoch_cnt = 0
                # torch.save({'epoch': epoch,
                #             'model_state_dict': model.state_dict(),
                #             'model_optimizer_state_dict': model_optimizer.state_dict(),
                #             # 'citeration_optimizer_state_dict': citeration_optimizer.state_dict(),
                #             'citeration': citeration.state_dict()}, cfg.BEST_MODEL)
            else:
                epoch_cnt += 1

            if max_event_precision is None or event_precision > max_event_precision:
                max_event_precision = event_precision

            if max_event_recall is None or event_recall > max_event_recall:
                max_event_recall = event_recall

            if min_time_loss is None or time_error / event_total < min_time_loss:
                min_time_loss = time_error / event_total

            if max_event_acc is None or event_acc > max_event_acc:
                max_event_acc = event_acc

            if epoch_cnt > cfg.EPOCH_THRESHOLD:
                break

    print('best model:', best_model)
    print("max_event_precision: %f\tmax_event_recall: %f\tmax_event_acc: %f\tmin_time_loss: %f" %
          (max_event_precision, max_event_recall, max_event_acc, min_time_loss))
    print('training finished.')


def testing(cfg, args, extra_args):
    print('testing starting...')
    print('testing finished.')


if __name__ == '__main__':
    main()
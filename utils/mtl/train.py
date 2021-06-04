import numpy as np
import pandas as pd

from collections import defaultdict
import tqdm

import torch
from torch.utils.data import DataLoader


class Trainer:
    '''
    Класс для обучения НН
    '''
    def __init__(self, model, criterion, device, learning_rate):

        '''
        model - архитектура модели
        criterion - используемая функция ошибки
        device - тип устройства для обучения модели
                 строка, может принимать значения: сpu, cuda, cuda:n,
                 где n - номер использумой видеокарты
        learning_rate - темп обучения
        '''

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            learning_rate)
        self.device = device

    def train_step(self, dataloader, verbose=False):

        '''
        Метод, выполняющий шаг обучения модели
        '''

        self.model.train()

        total_loss = 0.
        num = 0

        for X_batch, y_batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            pred = self.model(X_batch)

            loss = self.criterion(pred, y_batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item() * len(X_batch)

            X_batch, y_batch = X_batch.cpu(), y_batch.cpu()
            if verbose:
                if (num % 100) == 0:
                    print('loss: {:.4f}'.format(loss.item()))
            num += 1

        total_loss = total_loss / len(dataloader.dataset)

        return total_loss

    def eval_step(self, dataloader):

        '''
        Метод, выполняющий оценку модели
        '''

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item() * len(X_batch)

        total_loss = total_loss / len(dataloader.dataset)
        return total_loss

    def train(self, train_dataloader, test_dataloader,
              n_epochs=10, gap=3, verbose=False):

        '''
        Метод, выполняющий обучение модели
        train_dataloader - обучающая выборка
        test_dataloader - валидационная выборка
        n_epochs - количество эпох обучения
        gap - критерий ранней остановки: если ошибка модели в течение gap эпох
            не улучшается на тестовой выборке, то обучение прекращается
        verbose - вывод
        '''

        losses = defaultdict(list)

        min_score = np.inf

        for epoch in range(n_epochs):
            loss_train = self.train_step(train_dataloader)
            loss_test = self.eval_step(test_dataloader)

            losses['train'].append(loss_train)
            losses['test'].append(loss_test)

            print('epoch: {}| train loss: {:.4f}, test loss: {:.4f}'.format(
                epoch, loss_train, loss_test))

            if min_score > loss_test:
                min_score = loss_test
                patience = 0
            else:
                patience += 1
                if patience == gap:
                    self.model.cpu()
                    break

        self.model.cpu()

        return losses

    def predict(self, test_dl):

        '''
        Метод, возвращающий предсказания модели по переданной выборке
        '''

        sigmoid = torch.nn.Sigmoid()
        preds = np.empty(test_dl.dataset.targets.shape[1])
        preds = preds[None, :]

        with torch.no_grad():
            self.model.eval()
            for X_test, _ in test_dl:
                pred_batch = self.model(X_test)
                pred_batch = [sigmoid(pred).detach().numpy()
                              for pred in pred_batch]
                preds = np.vstack((preds, np.array(pred_batch).transpose()))

            self.model.cpu()

        return preds[1:]

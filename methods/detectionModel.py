from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from torch import nn
import torch
from .standardModel import EarlyStopping
from minisom import MiniSom

def Evaluating(pred, actual):
    accuracy = accuracy_score(pred, actual)
    precision, recall, f1, _ = prf(pred, actual, average='binary')
    auc = roc_auc_score(actual, pred)
    print("Real Anomaly Count:", np.count_nonzero(actual))
    print("Pred Anomaly Count:", np.count_nonzero(pred))
    print(f'Test Accuracy: {accuracy}')
    print(f'Test Precision: {precision}')
    print(f'Test Recall: {recall}')
    print(f'Test F1 Score: {f1}')
    print(f'AUC Score: {auc}')


def IsolationForestEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test,y_off):
    # 构建分类器
    print("IsolationForest Formal Data")
    clf1 = IsolationForest(n_estimators=500, random_state=30)
    clf1.fit(x1_train)
    pred1 = clf1.predict(x1_test)
    pred1 = np.where(pred1 == 1, 0, 1)
    Evaluating(pred1, y_test)
    print("\n")

    print("IsolationForest CRG Data")
    clf2 = IsolationForest(n_estimators=500, random_state=30)
    clf2.fit(x2_train)
    pred2 = clf2.predict(x2_test)
    pred2 = np.where(pred2 == 1, 0, 1)
    for i in range(len(pred2)):
        if y_off[i]==1:
            pred2[i]=0
        if y_off[i]==-1:
            pred2[i]=1
    Evaluating(pred2, y_test)


def RandomForestClassifierEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test,y_off):
    # 构建分类器
    print("RandomForest Formal Data")
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf1.fit(x1_train, y_train)
    pred1 = clf1.predict(x1_test)
    Evaluating(pred1, y_test)
    print("\n")

    print("RandomForest CRG Data")
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2.fit(x2_train, y_train)
    pred2 = clf2.predict(x2_test)
    for i in range(len(pred2)):
        if y_off[i]==1:
            pred2[i]=0
        if y_off[i]==-1:
            pred2[i]=1
    Evaluating(pred2, y_test)


def OneClassSVMEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test):
    # 构建分类器
    print("OneClassSVM Formal Data")
    clf1 = OneClassSVM(gamma='auto')
    clf1.fit(x1_train)
    pred1 = clf1.predict(x1_test)
    Evaluating(pred1, y_test)
    print("\n")

    print("OneClassSVM CRG Data")
    clf2 = OneClassSVM(gamma='auto')
    clf2.fit(x2_train)
    pred2 = clf2.predict(x2_test)
    Evaluating(pred2, y_test)


def GaussianMixtureEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test):
    # 构建分类器
    print("GaussianMixture Formal Data")
    clf1 = GaussianMixture(n_components=2, random_state=40)
    clf1.fit(x1_train)
    pred1 = clf1.predict(x1_test)
    Evaluating(pred1, y_test)
    print("\n")

    print("GaussianMixture CRG Data")
    clf2 = GaussianMixture(n_components=2, random_state=40)
    clf2.fit(x2_train)
    pred2 = clf2.predict(x2_test)
    Evaluating(pred2, y_test)


def KNeighborsClassifierEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test,y_off):
    # 构建分类器
    print("KNeighborsClassifier Formal Data")
    clf1 = KNeighborsClassifier(n_neighbors=9,weights='distance',algorithm='ball_tree',metric="chebyshev")
    clf1.fit(x1_train, y_train)
    pred1 = clf1.predict(x1_test)
    Evaluating(pred1, y_test)
    print("\n")

    print("KNeighborsClassifier CRG Data")
    clf2 = KNeighborsClassifier(n_neighbors=9,weights='distance',algorithm='ball_tree',metric="chebyshev")
    clf2.fit(x2_train, y_train)
    pred2 = clf2.predict(x2_test)
    for i in range(len(pred2)):
        if y_off[i]==1:
            pred2[i]=0
        if y_off[i]==-1:
            pred2[i]=1
    Evaluating(pred2, y_test)


class DNN(nn.Module):
    def __init__(self, dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim * 3)
        self.fc3 = nn.Linear(dim * 3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def DNNClassifierEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test,y_off):
    # 构建分类器
    print("DNN Formal Data")
    clf1 = DNN(x1_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(clf1.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=20)
    for epoch in range(1000):
        inputs = torch.tensor(x1_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        optimizer.zero_grad()
        outputs = clf1(inputs)
        loss = criterion(outputs, labels)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            loss.backward()
            optimizer.step()
        # print("epoch:",epoch,"loss:", loss.item())
    inputs = torch.tensor(x1_test, dtype=torch.float32)
    pred1 = clf1(inputs).detach().numpy().ravel()
    pred1 = np.where(pred1 > 0.5, 1, 0)
    Evaluating(pred1, y_test)
    print("\n")

    print("DNN CRG Data")
    clf2 = DNN(x2_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(clf2.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=20)
    for epoch in range(1000):
        inputs = torch.tensor(x2_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        optimizer.zero_grad()
        outputs = clf2(inputs)
        loss = criterion(outputs, labels)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            loss.backward()
            optimizer.step()
        # print("epoch:",epoch,"loss:", loss.item())
    inputs = torch.tensor(x2_test, dtype=torch.float32)
    pred2 = clf2(inputs).detach().numpy().ravel()
    pred2 = np.where(pred2 > 0.5, 1, 0)
    for i in range(len(pred2)):
        if y_off[i]==1:
            pred2[i]=0
        if y_off[i]==-1:
            pred2[i]=1
    Evaluating(pred2, y_test)
    print("\n")


def SOMClassifierEval(x1_train, x1_test, x2_train, x2_test, y_train, y_test):
    # 构建分类器
    print("SOM Formal Data")
    clf1 = MiniSom(20, 20, x1_train.shape[1], sigma=0.05, learning_rate=0.05)
    clf1.train(x1_train, 1000)
    pred1 = np.zeros(x1_test.shape[0])
    for i in range(x1_test.shape[0]):
        pred1[i] = clf1.winner(x1_test[i])[0]
    pred1 = np.where(pred1 > 0.5, 1, 0)
    Evaluating(pred1, y_test)
    print("\n")

    print("SOM CRG Data")
    clf2 = MiniSom(10, 10, x2_train.shape[1], sigma=1.0, learning_rate=0.5)
    clf2.train(x2_train, 100)
    pred2 = np.zeros(x2_test.shape[0])
    for i in range(x2_test.shape[0]):
        pred2[i] = clf2.winner(x2_test[i])[0]
    pred2 = np.where(pred2 > 0.5, 1, 0)
    Evaluating(pred2, y_test)
    print("\n")






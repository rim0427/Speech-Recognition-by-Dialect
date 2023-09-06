import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# import os
# import torch
# import torch.nn as nn
#
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
# #os.environ["CUDA_VISIBLE_DEVICES"]= "1"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print('Device:', device)  # 출력결과: cuda
# print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
# print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)
#
# _net = ResNet50().cuda()
# net = nn.DataParallel(_net).to(device)

import math
import time
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import scale
import random
from unicodedata import normalize
from keras.layers import Dense
from keras import Input
from keras import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU,  Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D
import pandas as pd
from scipy.io import wavfile
import librosa
from xgboost import XGBClassifier
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#
# def split_wav(data, sample_rate, start, end):
#   start *= sample_rate
#   end *= sample_rate
#   return data[start:end]
#
# pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
# pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
#
# DATA_DIR_1 = "/media/sungshin/새 볼륨/"
# DATA_DIR_2="/media/sungshin/새 볼륨/"
# # wav, sr = librosa.load('/home/sungshin/PycharmProjects/mfcc/sr_mfcc/trainset_1/DCDG20000116.wav', sr=16000)
# # wav = split_wav(wav, sr, 470, 480)
#
# # Data set list, include (raw data, mfcc data, y data)
# trainset = []
# testset = []
#
# # split each set into raw data, mfcc data, and y data
# # STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
# train_X = []
# train_mfccs = []
#
# test_X = []
# test_mfccs = []
#
# frame_length = 0.025
# frame_stride = 0.0010
#
# # train data를 넣는다.
# for filename in os.listdir(DATA_DIR_1 + "trainset2/"):
#     filename = normalize('NFC', filename)
#     try:
#         # wav 포맷 데이터만 사용
#         if '.wav' not in filename in filename:
#             continue
#
#         wav, sr = librosa.load(DATA_DIR_1+ "trainset2/" + filename, sr=16000)
#         wav=split_wav(wav, sr, 450, 462) #12sec
#         mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40, n_fft=400, hop_length=160)
#         padded_mfcc = pad2d(mfcc, 1200)
#         print(filename)
#         # print(mfcc.shape)
#         #print(padded_mfcc.shape)
#         # 추임새 별로 dataset에 추가
#         if filename[1] == 'C': #충청도
#              trainset.append((padded_mfcc, 0))
#         elif filename[1] == 'J': #전라도
#             trainset.append((padded_mfcc, 1))
#         elif filename[1] == 'K': #경상도
#             trainset.append((padded_mfcc, 2))
#         elif filename[1] == 'G':  #강원도
#             trainset.append((padded_mfcc, 3))
#         elif filename[1] == 'Z':  # 제주도
#             trainset.append((padded_mfcc, 4))
#     except Exception as e:
#         print(filename, e)
#         raise
#
# # test data를 넣는다.
# for filename in os.listdir(DATA_DIR_2 + "testset2/"):
#     filename = normalize('NFC', filename)
#     try:
#         # wav 포맷 데이터만 사용
#         if '.wav' not in filename in filename:
#             continue
#
#         wav, sr = librosa.load(DATA_DIR_2 + "testset2/" + filename, sr=16000)
#         wav=split_wav(wav, sr, 450, 462) #12sec
#         mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40, n_fft=400, hop_length=160)
#         padded_mfcc = pad2d(mfcc, 1200)
#         print(filename)
#         # 추임새 별로 test dataset에 추가
#         if filename[1] == 'C':
#             testset.append((padded_mfcc, 0))
#         elif filename[1] == 'J':
#             testset.append((padded_mfcc, 1))
#         elif filename[1] == 'K':
#             testset.append((padded_mfcc, 2))
#         elif filename[1] == 'G':
#             testset.append((padded_mfcc, 3))
#         elif filename[1] == 'Z':  # 제주도
#             testset.append((padded_mfcc, 4))
#     except Exception as e:
#         print(filename, e)
#         raise
# #
# #
# train_x = [a for (a,b) in trainset]
# train_y = [b for (a,b) in trainset]
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# np.save('train_x_test.npy',train_x)
# np.save('train_y_test.npy',train_y)
#
# test_x = [a for (a,b) in testset]
# test_y = [b for (a,b) in testset]
# test_x = np.array(test_x)
# test_y=np.array(test_y)
# np.save('test_y_test.npy',test_y)
# np.save('test_x_test.npy',test_x)
#
# train_y_cat = to_categorical(np.array(train_y)) #(0,1,1,0) -> ((1,0), (0,1), (0,1), (1,0))
# test_y_cat = to_categorical(np.array(test_y))
# np.save('train_y_cat_test.npy',train_y_cat)
# np.save('test_y_cat_test.npy',test_y_cat)

train_x=np.load('train_x_test.npy')
train_y=np.load('train_y_test.npy')
test_x=np.load('test_x_test.npy')
test_y=np.load('test_y_test.npy')
test_y_cat=np.load('test_y_cat_test.npy')
train_y_cat=np.load('train_y_cat_test.npy')

X_train_flat = np.array([features_2d.flatten() for features_2d in train_x])
X_test_flat = np.array([features_2d.flatten() for features_2d in test_x])


x_all = np.concatenate([X_train_flat,X_test_flat])
y_all = np.concatenate([train_y,test_y])

# train_X_ex = np.expand_dims(train_x, -1)
# test_X_ex = np.expand_dims(test_x, -1)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# #
# #로지스틱 회귀
print("logistic regression")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(verbose = 0)
start = time.time()
lr.fit(X_train_flat, train_y)
end = time.time()
y_pre = lr.predict(X_test_flat)
print(lr.score(X_train_flat, train_y))
print(lr.score(X_test_flat, test_y))
print("Acc: ", accuracy_score(test_y, y_pre)) #정합도(정답-정답, 오답-오답)
print("Pre: ", precision_score(test_y, y_pre, average='macro')) #정밀도(tp/(tp+fp))
print("recall: ", recall_score(test_y, y_pre, average='macro')) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
print("f1_sco: ", f1_score(test_y, y_pre, average='macro')) #정밀도,재현율 역수 평균의 역수
print(f"logistic time: {end - start:.5f} sec")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
# 예측값과 실제 값
predicted_labels = y_pre # 모델의 예측값
true_labels = test_y   # 실제 레이블
print("예측:",predicted_labels)
print("실제:",true_labels)
# 예측이 틀린 경우를 추적하기 위한 리스트 초기화
wrong_predictions = []

# 예측값과 실제 값 비교
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append((true_labels[i], predicted_labels[i]))

# 틀린 예측을 표로 나타내기 위한 딕셔너리 초기화
error_counts = {}

# 틀린 예측 개수 계산
for true_label, predicted_label in wrong_predictions:
    if (true_label, predicted_label) in error_counts:
        error_counts[(true_label, predicted_label)] += 1
    else:
        error_counts[(true_label, predicted_label)] = 1

# 결과 표 출력
print("실제 값 -> 예측 값 : 틀린 개수")
for (true_label, predicted_label), count in error_counts.items():
    print(f"{true_label} -> {predicted_label} : {count} 개")


confusion = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',vmin=0,vmax=300,
            xticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'],
            yticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('result_logistic.png')
#https://perconsi.tistory.com/83 다중분류

#이진분류 score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# print("Acc: ", accuracy_score(y_test, y_pre)) #정합도(정답-정답, 오답-오답)
# print("Pre: ", precision_score(y_test, y_pre)) #정밀도(tp/(tp+fp))
# print("recall: ", recall_score(y_test, y_pre)) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
# print("f1_sco: ", f1_score(y_test, y_pre)) #정밀도,재현율 역수 평균의 역수
# end = time.time()
# print(f"logistic time: {end - start:.5f} sec")
# start = time.time()

#XGB
print("xgb 시작")
xgb_2 = XGBClassifier(tree_method='gpu_hist', objective = "multi:softmax", gpu_id = 0)
start = time.time()
xgb_2.fit(X_train_flat,train_y)
end = time.time()

y_pre_xgb = xgb_2.predict(X_test_flat)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
print(xgb_2.score(X_train_flat,train_y))
print(xgb_2.score(X_test_flat,test_y))
print("Acc: ", accuracy_score(test_y, y_pre_xgb)) #정합도(정답-정답, 오답-오답)
print("Pre: ", precision_score(test_y, y_pre_xgb, average='macro')) #정밀도(tp/(tp+fp))
print("recall: ", recall_score(test_y, y_pre_xgb,average='macro')) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
print("f1_sco: ", f1_score(test_y, y_pre_xgb,average='macro')) #정밀도,재현율 역수 평균의 역수
print(f"xgb time: {end - start:.5f} sec")
# 예측값과 실제 값
predicted_labels = y_pre_xgb # 모델의 예측값
true_labels = test_y   # 실제 레이블
print("예측:",predicted_labels)
print("실제:",true_labels)
# 예측이 틀린 경우를 추적하기 위한 리스트 초기화
wrong_predictions = []

# 예측값과 실제 값 비교
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append((true_labels[i], predicted_labels[i]))

# 틀린 예측을 표로 나타내기 위한 딕셔너리 초기화
error_counts = {}

# 틀린 예측 개수 계산
for true_label, predicted_label in wrong_predictions:
    if (true_label, predicted_label) in error_counts:
        error_counts[(true_label, predicted_label)] += 1
    else:
        error_counts[(true_label, predicted_label)] = 1

# 결과 표 출력
print("실제 값 -> 예측 값 : 틀린 개수")
for (true_label, predicted_label), count in error_counts.items():
    print(f"{true_label} -> {predicted_label} : {count} 개")

confusion = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',vmin=0,vmax=300,
            xticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'],
            yticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('result_xgb.png')

# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 정확도 함수
print("RF")
clf = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0, verbose = 1)
start = time.time()
clf.fit(X_train_flat,train_y)
end = time.time()
y_pre_rf = clf.predict(X_test_flat)
print(clf.score(X_train_flat,train_y))
print(clf.score(X_test_flat,test_y))
print("Acc: ", accuracy_score(test_y, y_pre_rf)) #정합도(정답-정답, 오답-오답)
print("Pre: ", precision_score(test_y, y_pre_rf, average='macro')) #정밀도(tp/(tp+fp))
print("recall: ", recall_score(test_y, y_pre_rf, average='macro')) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
print("f1_sco: ", f1_score(test_y, y_pre_rf, average='macro')) #정밀도,재현율 역수 평균의 역수
print(f"RF time: {end - start:.5f} sec")
import numpy as np
#
# 예측값과 실제 값
predicted_labels = y_pre_rf # 모델의 예측값
true_labels = test_y   # 실제 레이블

# 예측이 틀린 경우를 추적하기 위한 리스트 초기화
wrong_predictions = []

# 예측값과 실제 값 비교
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append((true_labels[i], predicted_labels[i]))

# 틀린 예측을 표로 나타내기 위한 딕셔너리 초기화
error_counts = {}

# 틀린 예측 개수 계산
for true_label, predicted_label in wrong_predictions:
    if (true_label, predicted_label) in error_counts:
        error_counts[(true_label, predicted_label)] += 1
    else:
        error_counts[(true_label, predicted_label)] = 1

# 결과 표 출력
print("실제 값 -> 예측 값 : 틀린 개수")
for (true_label, predicted_label), count in error_counts.items():
    print(f"{true_label} -> {predicted_label} : {count} 개")

confusion = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',vmin=0,vmax=300,
            xticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'],
            yticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('result_RF.png')

# print("sequn")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
keras_model = Sequential()
# keras_model.add(Dense(3, activation = 'softmax', input_shape = (X_train_flat.shape[1],)))
# keras_model.compile(optimizer=Adam(learning_rate = 1e-1), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# keras_model.fit(np.array(X_train_flat), np.array(train_y), epochs = 20, verbose = 1, batch_size = 100, validation_split = 0.2)
# keras_result = keras_model.evaluate(np.array(X_test_flat), np.array(test_y))
# print(keras_result)

# SVM
print("SVM")
from sklearn import svm

svm_model = svm.SVC()
start = time.time()
svm_model.fit(X_train_flat, train_y)
end = time.time()

y_pre_svm = svm_model.predict(X_test_flat)
print(svm_model.score(X_train_flat, train_y))
print(svm_model.score(X_test_flat, test_y))
print("Acc: ", accuracy_score(test_y, y_pre_svm))  # 정합도(정답-정답, 오답-오답)
print("Pre: ", precision_score(test_y, y_pre_svm, average='macro'))  # 정밀도(tp/(tp+fp))
print("recall: ", recall_score(test_y, y_pre_svm, average='macro'))  # 재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
print("f1_sco: ", f1_score(test_y, y_pre_svm, average='macro'))  # 정밀도,재현율 역수 평균의 역수
print(f"SVM time: {end - start:.5f} sec")
# 예측값과 실제 값
predicted_labels = y_pre_svm  # 모델의 예측값
true_labels = test_y  # 실제 레이블

# 예측이 틀린 경우를 추적하기 위한 리스트 초기화
wrong_predictions = []

# 예측값과 실제 값 비교
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append((true_labels[i], predicted_labels[i]))

# 틀린 예측을 표로 나타내기 위한 딕셔너리 초기화
error_counts = {}

# 틀린 예측 개수 계산
for true_label, predicted_label in wrong_predictions:
    if (true_label, predicted_label) in error_counts:
        error_counts[(true_label, predicted_label)] += 1
    else:
        error_counts[(true_label, predicted_label)] = 1

# 결과 표 출력
print("실제 값 -> 예측 값 : 틀린 개수")
for (true_label, predicted_label), count in error_counts.items():
    print(f"{true_label} -> {predicted_label} : {count} 개")

confusion = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=300,
            xticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'],
            yticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('result_svm.png')

#CNN
train_X_ex = np.expand_dims(train_x, -1)
test_X_ex = np.expand_dims(test_x, -1)

ip = Input(shape=train_X_ex[0].shape)
m = Conv2D(128, kernel_size=(3,3), activation='relu')(ip)
m = MaxPooling2D(pool_size=(2,2))(m)
m = Dropout(0.1)(m) #dropout 추가

m = Conv2D(64, kernel_size=(3,3), activation='relu')(m)
m = MaxPooling2D(pool_size=(2,2))(m)
m = Dropout(0.1)(m) #dropout 추가

m = Conv2D(32, kernel_size=(3,3), activation='relu')(m)
m = MaxPooling2D(pool_size=(2,2))(m)
m = Dropout(0.1)(m) #dropout 추가

# m = Conv2D(32, kernel_size=(3,3), activation='relu')(m)
# m = MaxPooling2D(pool_size=(2,2))(m)
# m = Dropout(0.1)(m) #dropout 추가

m = Flatten()(m)

m = Dense(64, activation='relu')(m)

m = Dense(32, activation='relu')(m)

op = Dense(5, activation='softmax')(m)

model = Model(ip, op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

start = time.time()
history = model.fit(np.array(train_X_ex),np.array(train_y_cat),epochs=100, batch_size=64,verbose=1,validation_data=(np.array(test_X_ex), np.array(test_y_cat)))
end = time.time()
_,acc = model.evaluate(test_X_ex,test_y_cat,batch_size=64,verbose=1)

print('loss:',_,'acc:',acc)
# print(X_train_flat.shape)
# print(X_train_flat.[1])
#훈련세트 정답

_,acc = model.evaluate(train_X_ex,train_y_cat,batch_size=64,verbose=1)
print('loss:',_,'acc_train:',acc)
# y_pre_cnn = clf.predict(test_X_ex)
# print("Acc: ", accuracy_score(test_y_cat, y_pre_cnn)) #정합도(정답-정답, 오답-오답)
# print("Pre: ", precision_score(test_y_cat, y_pre_cnn), average='macro') #정밀도(tp/(tp+fp))
# print("recall: ", recall_score(test_y_cat, y_pre_cnn, average='macro')) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
# print("f1_sco: ", f1_score(test_y_cat, y_pre_cnn, average='macro')) #정밀도,재현율 역수 평균의 역수

# Make predictions
test_pred = model.predict(test_X_ex)
test_pred_classes = np.argmax(test_pred, axis=1)

# Compute precision, recall, and F1-score
precision = precision_score(np.argmax(test_y_cat, axis=1), test_pred_classes, average='weighted')
recall = recall_score(np.argmax(test_y_cat, axis=1), test_pred_classes, average='weighted')
f1 = f1_score(np.argmax(test_y_cat, axis=1), test_pred_classes, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print(f"CNN time: {end - start:.5f} sec")

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

import numpy as np

# 예측 확률 분포와 실제 확률 분포 (예시)
predicted_probabilities = test_pred

true_probabilities = test_y_cat
print("예측:",predicted_probabilities)
print("실제:",true_probabilities)
# 틀린 예측을 추적하기 위한 리스트 초기화
wrong_predictions = []

# 예측 확률 중 가장 높은 클래스 선택
predicted_labels = np.argmax(predicted_probabilities, axis=1)
true_labels = np.argmax(true_probabilities, axis=1)
print("예측:",predicted_labels)
print("실제:",true_labels)
# 예측값과 실제 값 비교
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append((true_labels[i], predicted_labels[i]))

# 틀린 예측을 표로 나타내기 위한 딕셔너리 초기화
error_counts = {}

# 틀린 예측 개수 계산
for true_label, predicted_label in wrong_predictions:
    if (true_label, predicted_label) in error_counts:
        error_counts[(true_label, predicted_label)] += 1
    else:
        error_counts[(true_label, predicted_label)] = 1

# 결과 표 출력
print("실제 확률 분포 -> 예측 확률 분포 : 틀린 개수")
for (true_label, predicted_label), count in error_counts.items():
    print(f"{true_label} -> {predicted_label} : {count} 개")



# 모델의 예측값(predicted_labels)과 실제값(true_labels) 준비 (예시 데이터)
# predicted_labels = [2, 4, 3, 2, 7, 1, 4, 2, 6, 0]  # 모델의 예측값
# true_labels = [2, 4, 3, 5, 7, 1, 9, 2, 6, 0]      # 실제 레이블

# 혼동 행렬 생성
confusion = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',vmin=0,vmax=300,
            xticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'],
            yticklabels=['Chungcheong', 'Jeolla', 'Gyeongsang', 'Gangwon', 'Jeju'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('result_cnn.png')

#LSTM
# from keras.layers import Dense, LSTM
#
# model = Sequential()
# train_X_ex = np.expand_dims(train_x, -1)
# test_X_ex = np.expand_dims(test_x, -1)
#
# model = Sequential()
# model.add(LSTM(512,input_shape=(8,100, 4000),return_sequences=True)) #input_shape은 x의 라벨값 6개 시퀀스 출력은 True 512차원 출력
# model.add(Dropout(0.3)) #과적합 방지를 위한 드랍아웃 비율은 0.3
# model.add(LSTM(256, return_sequences=True)) #LSTM 층  256차원출력
# model.add(Dropout(0.3)) #드랍아웃 층
# model.add(LSTM(128)) #LSTM층 128차원 출력
# model.add(Dense(128)) #은닉층
# model.add(Dropout(0.3)) #드랍아웃 층
# model.add(Dense(9)) #은닉층
# model.add(Dense(9, activation='softmax')) #활성화 함수 소프트맥스 사용 출력은 0~8 사이인 9개의 차원출력
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop' ,metrics=['accuracy'])
# model.summary()
#
# model.fit(train_x, train_y_cat, epochs=100,
#           batch_size=8, verbose=1)
#
# # X_test= test_x.reshape(test_x.shape[0], test_x.shape[1], 1) # LSTM층 에 맞게 형변환
# y_score = model.predict(test_x)
#
# y_pred = np.argmax(y_score, axis=1)
# print(y_pred)

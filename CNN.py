#gpu 사용 코드
# print("sequn")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
keras_model = Sequential()

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
#훈련세트 정답
_,acc = model.evaluate(train_X_ex,train_y_cat,batch_size=64,verbose=1)
print('loss:',_,'acc_train:',acc)

y_pre_cnn = clf.predict(test_X_ex)
print("Acc: ", accuracy_score(test_y_cat, y_pre_cnn)) #정합도(정답-정답, 오답-오답)
print("Pre: ", precision_score(test_y_cat, y_pre_cnn), average='macro') #정밀도(tp/(tp+fp))
print("recall: ", recall_score(test_y_cat, y_pre_cnn, average='macro')) #재현율(tp/(tp+fn)), 실제 정답을 얼마나 많이 선택?
print("f1_sco: ", f1_score(test_y_cat, y_pre_cnn, average='macro')) #정밀도,재현율 역수 평균의 역수

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

# 모델의 예측값(predicted_labels)과 실제값(true_labels) 준비
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

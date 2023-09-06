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

# Final-project

1. 데이터 증강
이미지당 5번 증강하여 로컬 디렉토리에 저장.
증강된 데이터를 별도로 저장하고 관리 가능.

2. 기본 CNN 모델
2개의 합성곱 레이어, 1개의 풀링 레이어, 1개의 드롭아웃 레이어 포함.
Fully Connected 레이어를 통해 CIFAR-10의 10개 클래스를 예측.

3.모델 훈련 및 평가
Adam 옵티마이저와 CrossEntropyLoss 사용.

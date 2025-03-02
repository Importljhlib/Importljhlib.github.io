---
layout: post
title: 모델을 overfitting하게 만들어보자
date: 2025-03-03 01:53:00 + 0900
category: experiment
---
오버피팅이 일어나는 이유로는 크게 다음과 같다.

1. 데이터 수의 부족
2. 모델이 너무 깊거나 복잡함
3. 과도한 학습, 에폭 수
4. 정규화 부족

초기 세팅:

- AlexNet architecture
- 학습 데이터셋 수 : 전체의 80%
- mini batch : 128
- Conv 5개, FC 3개
- lr = 0.01, momentum = 0.9, weight_deacy = 0.0005
- epoch = 10
- LRN, Dropout 사용.

## 1. 학습 데이터 수 줄이기

세팅 : 학습 데이터 수를 전체의 20%로 감소 시켰다.

*학습 데이터가 적으면 모델이 특정 패턴에 더 강하게 학습하여 오버피팅이 될 것이라고 생각하였다.* 

![](/img/ovimage.png)

결과는 다음과 같다.

![](/img/ovimage1.png)

오버피팅이 발생하지 않았다.

학습을 하는 횟수가 너무 짧아서 오버피팅이 아직 발생하지 않은 것 같다.

그래서 다음 학습에서는 업데이트 횟수를 증가시키기 위해서 배치의 크기를 줄였다.

### 1-2. 배치 크기 줄이기

세팅 : 배치 사이즈를 128 → 32로 줄였다.

*배치 사이즈를 줄임으로써 모델이 적은 양의 데이터로 여러 번 업데이트하게 되면서 오버피팅이 발생할 것으로 생각했다.* 

![](/img/ovimage2.png)

### 1-3. epoch 늘리기

세팅 : epoch를 10 → 30으로 늘렸다. 또한 1-2의 모델을 초기화 시키지 않고 이어서 학습을 시켰다.

*아직까지 오버피팅이 발생 할 신호가 보이지 않아서 epoch을 더 늘리고 지켜보기로 하였다.*

![](/img/ovimage3.png)

오버피팅이 발생하였다.

val_loss는 6~7 epoch 까지는 감소하다가 그 이후로는 증가하는 경향성을 보였다.

Train_acc는 95%가 넘어가는 반면에 Validation_acc는 70%가 겨우 넘고, 크게 증가하지 않는 경향을 보이고 있다.

![](/img/ovimage4.png)

Test_acc는 72.02%를 기록하였다.

## 2. 정규화 기법 제거

세팅 : 초기 세팅에서 LRN, Dropout, L2 제거, epoch 20으로 설정.

*정규화 기법을 제거함으로써 오버피팅이 일어날 것이라고 생각하였다.*

결과는 다음과 같다.

![](/img/ovimage5.png)

오버피팅이 발생하였다.

Train loss는 계속해서 감소하는 반면에 Validation loss는 epoch 7.5 부터 상승하는 경향을 보이고 있다.

Train acc는 계속해서 증가하여서 99에 가까워진 반면에 Validation acc는 7.5 epoch 부터 증가하지 않고 계속 유지되는 것에 더해서 80%보다도 낮은 정확도를 보이고 있다. 

![](/img/ovimage6.png)

Test set에 대한 정확도는 77.10%를 기록하였다.

## 3. 네트워크 복잡도 증가

세팅 : 초기 세팅에서 Conv 레이어 2개를 추가하였다.

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 추가된 레이어
            ###################################################################
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            ###################################################################
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6 * 6 * 512, 4096),  # 입력 크기 256 -> 512 조정
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
```

![](/img/ovimage7.png)

모델이 깊어져서 학습을 하는데에 굉장히 오래 걸리는 것 같다.

### 3-1. 출력층 직전의 노드 수 증가

세팅 : 초기 세팅에서 출력층 직전의 노드 수를 증가 시켰다.

블로그를 찾아보니, 오버피팅을 줄이려면 출력층 직전의 파라미터 수를 줄이는 것이 좋다고 한다.

설명에 따르면 출력층 직전 은닉층 노드 수는 셜명변수의 수 라고 한다. 따라서 의미있는 설명변수를 생성하기 위해 출력 직전의 노드 수를 줄이는 것이라고 한다.

그래서 나는 그 반대로 출력층 직전의 노드 수를 증가 시켰다.

```python
    self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(6 * 6 * 256, 8192), #  출력 크기 4096 -> 8192로 증가
        nn.ReLU(inplace = True),

        nn.Dropout(0.5),
        nn.Linear(8192, 8192), # 4096 -> 8192로 노드 수 증가
        nn.ReLU(inplace = True),
        nn.Linear(8192, num_classes)
    )
```

![](/img/ovimage8.png)

이어서 학습 시켰다.

![](/img/ovimage9.png)

오버피팅이 발생하였다.

validation loss는 감소하다가 증가하는 구간이 생긴 것을 볼 수 있다.

acc 곡선에서는 Train acc는 계속해서 증가하는 반면에 validation acc는 Train acc와 점점 격차가 커지는 것을 관찰할 수 있다.

## 정리

epoch가 커야 러닝커브의 경향성을 더 정확하게 볼 수 있는 것 같다.

그리고 한 가지 간과한 점은 모델을 학습 시킨 후, Test loss를 보지 않았고, Test acc도 전부 확인하지는 못했다. 

epoch을 늘릴 수록 모델의 acc가 증가하는 것은 사실이다. 하지만 점점 증가할 수록 오버피팅이 일어나는 시점이 생길 것이고, 그 지점을 잘 찾아내는 것이 중요한 것 같다고 생각하였다.

또한 Train acc와 Test acc가 크게 차이나는 모델, 즉 오버핏이 된 모델은 제대로 일반화 되었다고 보기 어렵기 때문에 좋은 모델이라고 볼 수 없는 것 같다.

## 추가 (익스트림한 세팅)

## 학습 이미지 1개

세팅 : train_set의 사이즈를 1로 설정하였다.

*학습할 수 있는 데이터가 1개이기 때문에 Train_acc는 0 또는 100이 나올 것이라고 생각했고, val_acc와 test_acc는 클래스의 갯수가 10개이기 때문에 10%로 나올 것이라고 생각했다.*

결과는 다음과 같다.

![](/img/ovimage10.png)

오직 하나의 학습 데이터에 대해서 오버핏이 된 결과를 볼 수 있다.

![](/img/ovimage11.png)

클래스가 10개이기 때문에 Test_acc도 10%가 나오게 된다.

## 배치 사이즈 1

세팅 : 배치 사이즈를 1로 설정하였다.

결과는 다음과 같다.

![](/img/ovimage12.png)

데이터 하나 하나를 학습하기 때문에 수렴하지 않고 굉장히 불안정하게 학습된 것을 볼 수 있다.

또한 배치 사이즈가 극도로 작기 때문에 반복되는 시간이 많아져 학습 시간이 매우 오래 걸렸었다.

## 높은 학습률

세팅 : 학습률을 0.5로 설정하였다.

![](/img/ovimage13.png)

loss가 Nan이 되어서 0.1로 변경하고 다시 학습하였다.

![](/img/ovimage14.png)

학습도중에 loss가 nan이 되어서 학습을 중단시켰다.

이번에도 마찬가지로 수렴하지 않고 불안정하게 학습되고 있었던 것을 관찰할 수 있다.
---
layout: post
title: ImageNet Classification with Deep Convolutional Neural Networks
date: 2025-02-16 01:53:00 + 0900
category: paper
---

# AlexNet: ImageNet Classification with Deep Convolutional Neural Networks

## Abstract

**AlexNet**은 ILSVRC-2012에서 top-5 test error rate를 15.3% 달성하며 우승한 CNN 아키텍쳐이다. 

6천만개의 파라미터와 650,000개의 뉴런을 가졌으며, 5개의 convolutional layer과 3개의 FC-layer, 마지막엔 1000-way softmax로 구성되어 있다. 

또한 ReLU, 효과적인 GPU 사용, dropout 등 다양한 방법들이 사용되었다.

## Introduction

원래는 객체를 탐지하려면 머신러닝 방식을 필수적으로 사용하였다. 그리고 이의 성능을 향상하기 위해서 더 큰 데이터셋과 더 강력한 모델, 그리고 과적합을 막기 위해서 더 나은 기술들을 사용해왔다. 

간단한 recognition과 같은 작업에서는 작은 데이터셋에서 잘 작동하였지만, 현실의 데이터는 매우 다양하기 때문에 그것들을 인식하기 위해서는 매우 큰 데이터셋이 필요하다. 또한 이러한 많은 양의 데이터를 학습 시키기 위한 모델이 필요했다.

CNN은 모델의 깊이와 너비를 변경할 수 있어서 기존의 신경망보다 더 적은 연결과 파라미터로 훈련이 가능했고, 또한 당시에 강력한 GPU가 등장하면서 2D 합성곱 연산에 최적화 되어 CNN을 훈련시키기에 충분하였다.

AlexNet은 GTX580 3GB GPU 2개를 사용하여 5~6일간 훈련을 진행하였다.

## The Dataset

![](/img/alex1.png)

**ImageNet**은 1500만개 이상의 높은 해상도의 이미지와 약 22,000개의 카테고리를 가지고 있는 데이터셋이다.

ILSVRC에서는 ImageNet의 subset으로 1,000개의 카테고리, 120만개의 훈련 이미지, 5만개의 검증 이미지 그리고 15만개의 test 이미지를 사용한다.

또한 모델의 input에 일정한 차원으로 들어가야 하기 때문에 256*256 만큼 다운 샘플링하여 모델의 입력으로 사용하게 된다.

## The Architecture

![](/img/alex2.webp)

AlexNet은 5개의 conv layer과 3개의 fc layer로 총 8개의 layer로 구성 되어있다. 

### ReLU Nonlinearity

보통 모델에서 뉴런의 입력 $x$에 대한 출력으로 $f$는 $f(x) = tanh(x) \; or \; f(x) = (1 + e^{-x})^{-1}$
 이 기본적인 방법이다. 하지만 이러한 방법들은 saturation이 발생하여 학습의 속도를 저하 시킨다.

따라서 이 논문에서는 non-saturating nonlinearity로 ReLU를 사용한다. 

  

$$
f(x) = max(0, x)
$$

ReLU 함수는 위와 같이 정의된다.

아래의 Figure는 ReLU (solid line)과 tanh (dashed line)을 비교한 것인데, ReLU를 사용했을 때 수렴 속도가 무려 6배나 빨라진 것을 확인할 수 있다.

![](/img/alex3.png)

### Training on Multiple GPUs

하나의 GTX 580 GPU는 3GB의 메모리 밖에 가지고 있지 않아서, 120만개의 training 샘플을 학습 시키기에는 힘들기 때문에 2개의 GPU를 사용하였다. 

layer3의 모든 입력은 layer2의 모든 커널 맵으로 부터 받지만, layer4의 입력은 layer3 중 같은 GPU에 있는 것만 받게 된다.  

### Local Response Normalization

![](/img/alex4.png)

$a^{i}_{x, y}$ : (x, y)에 존재하는 픽셀에 대해 i번째 커널을 적용하여 얻은 결과에 ReLU를 적용한 값.

$N$ : 레이어에 있는 전체 커널 수

$n$  : 정규화에 사용할 채널 개수

$α,β$ : 하이퍼파라미터 (논문에선 $α = 10^{-4},β = 0.75$)

LPN은 특정 채널의 뉴런이 활성화될 경우, 주변 채널의 활성화를 억제하여 상대적인 차이를 강조한다.

AlexNet에서는 첫 번째와 두 번째 conv layer 뒤에 LRN을 적용하였다.

### Overlapping Pooling

![](/img/alex5.png)

![](/img/alex6.png)

기존의 pooling과 다르게 연산을 겹쳐서 진행한다.

### Overall Architecture

![](/img/alex7.png)

AlexNet의 전체적인 구조이다.

앞에서 설명했듯이 5개의 conv layer, 3개의 fc layer로 이루어져 있다. 

1. Conv1 (Conv + ReLU)
    1. 입력 크기 : 224 * 224 * 3
    2. 필터 크기 : 11 * 11
    3. 필터 수 : 96
    4. 스트라이드 : 4
    5. 출력 크기 : 55 * 55 * 96 ( 224 - 11 / 4 + 1)
    6. 96개의 특징 맵이 생기게 된다.
    7. pooling 크기가 3 * 3이고 stride가 2인 Max pooling을 적용한 후 LPN을 적용하여 27 * 27 * 96 크기의 output을 출력한다.
2. Conv2 (Conv + ReLU)
    1. 입력 크기 : 27 * 27 * 96
    2. 필터 크기 : 5 * 5
    3. 필터 수 : 256
    4. 스트라이드 : 1
    5. 출력 크기 : 27 * 27 * 256 (패딩 사용)
    6. pooling 크기가 3 * 3이고 stride가 2인 Max pooling을 적용한 후 LPN을 적용하여 13 * 13 * 256크기의 output을 출력한다.
3. Conv3 (Conv + ReLU)
    1. 입력 크기 : 13 * 13 * 256
    2. 필터 크기 : 3 * 3
    3. 필터 수 : 384
    4. 스트라이드 : 1
    5. 출력 크기 : 13 * 13 * 384 (패딩 사용)
4. Conv4 (Conv + ReLU)
    1. 필터 크기 : 3 * 3
    2. 필터 수 : 384
    3. 스트라이드 : 1
    4. 출력 크기 13 * 13 * 384 (패딩 사용)
5. Conv5 (Conv + ReLU)
    1. 필터 크기 : 3 * 3
    2. 필터 수 : 256
    3. 스트라이드 : 1
    4. 출력 크기 : 13 * 13 * 256
    5. pooling 크기가 3 * 3이고 stride가 2인 Max pooling을 적용하여 6 * 6 * 256 크기의 output을 출력한다.
6. FC6 (FC + ReLU + Dropout)
    1. 출력 크기 : 4096
7. FC7 (FC + ReLU + Dropout)
    1. 출력 크기 : 4096
8. FC8 (Softmax)
    1. 출력 크기 : 1000 (ImageNet Classes)

## Reducing Overfitting

### Data Augmentation

Data Augmentation은 데이터를 여러장으로 증가 시키는 방법이다.

첫 번째 방법은 단순히 이미지를 **수평반전** 하는 것이다.

이미지를 좌우반전을 한 후에 랜덤으로 224 * 224 크기의 패치를 얻어서 데이터로 사용하였다.

두 번째 방법은 RGB 채널의 강도를 변환하는 것이다.

원본 이미지의 RGB 픽셀 값들에 대해서 PCA (주성분 분석)을 수행하여 주성분 벡터를 구한 뒤에, 이를 이용해 원본 이미지에 작은 노이즈를 추가하여 색상 강도를 변형한다.

### Dropout

dropout 기법은 임의로 지정한 확률(AlexNet에서는 0.5)로 각각의 hidden neuron의 출력을 0으로 만드는 것이다. 그래서 우리는 매번 인풋을 넣을 때 마다 다른 아키텍쳐에 학습을 하는 효과를 얻을 수 있다. 하지만 모든 이 아키텍쳐의 가중치는 공유된다. 

 

## Details of learning

SGD를 사용하였다. 배치 사이즈는 128, 0.9 momentum 그리고 weight decay를 0.0005로 설정하였다.

![가중치 w에 대한 업데이트](/img/alex8.png)

가중치 w에 대한 업데이트

i는 반복 횟수, v는 모멘텀 변수, ϵ는 학습률을 의미한다. 그리고 $\left\langle \frac{\partial L}{\partial w} \Big|_{w_i} \right\rangle_{D_i}$
는 i번째 배치의 평균인 D_i를 의미한다.

각 layer의 weight는 평균이 0이고 표준편차가 0.01인 가우시간 분포로 초기화 하였다.

2, 4, 5번째 Conv layer와 모든 fc layer의 bias는 ReLU에 양수의 입력을 넣어주어 학습을 가속시키기 위해 1로 초기화 하였고, 나머지는 0으로 초기화 하였다.

## Results

![ILSVRC-2012 대회의 validation과 test set에 대한 error rate를 나타낸 table이다.](/img/alex9.png)

ILSVRC-2012 대회의 validation과 test set에 대한 error rate를 나타낸 table이다.

### Qualitative Evaluations

![](/img/alex10.png)

첫번째 conv layer에서 11 * 11 * 3의 사이즈인 커널 96개가 224 * 224 * 3의 입력 이미지를 학습할 때 위의 3줄은 첫번째 GPU에서, 아래의 3줄은 두번째 GPU에서 병렬적으로 학습한다고 한다.

GPU 1에서는 color 정보를 고려하지 않고 학습한 반면에, GPU 2는 color의 특징을 학습했다고 한다. 

![](/img/alex11.png)

왼쪽 8개의 그림은 ILSVRC-2010에서 테스트 이미지에 대해 모델이 예측한 상위 5개의 라벨이다. 

오른쪽 그림은 첫번째 열에 있는 테스트 이미지에 대한 마지막 hidden layer에서 만들어낸 특징 벡터로부터의 최소 유클리드 거리를 이용하여 얻은 훈련 이미지이다

개와 코끼리의 사진들을 보다싶이 이들은 다양한 포즈로 나타나는데, 이는 픽셀 수준에서 생각하면 L2에서 이미지가 가깝지 않다는 것을 알 수 있다.

두 개의 4096차원에서 실수값 벡터의 유클리드 거리의 유사도를 계산하는 것보다 오토 인코더를 이용하여 짧은 이진코드로 벡터를 압축하여 훈련하는 것이 더 효율적이다.

이미지 라벨을 사용하지 않기 때문에 유사한 edge 패턴을 가진 이미지를 검색하는 경향이 있는 raw 픽셀에 오토 인코더를 적용하는 것보다 훨씬 좋은 이미지 검색 방법이다.
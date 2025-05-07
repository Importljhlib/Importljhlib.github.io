---
layout: post
title: "AND : Adversarial Neural Degradation for Learning Blind Image Super-Resolution (NeurIPS 2023)"
date: 2025-05-08 01:53:00 + 0900
category: paper
---
## Introduction

Image Super-Resolution (SR) 분야에서 딥러닝 모델이 학습 시 가정한 degradation 모델과 실제 추론 단계에서 degradation 소스 간의 불일치로 인해 성능이 저하되는 문제가 있었다. 실제 이미지 degradation은 복접하고 비선형적이며, 다양한 원인으로 인해 발생하기 때문이다. 

이 논문은 이러한 한계를 극복하기 위해 새로운 적대적 신경망 열화(Adversarial Neural Degradation, AND) 모델을 제안하였다. AND 모델은 심층 복원 신경망과 함께 minmax 기준 하에 적대적으로 학습되어, 명시적인 지도(supervision) 없이도 광범위하고 복잡한 비선형적 열화 효과를 생성할 수 있다. 이 모델은 학습 시 보지 못한 다양한 열화 변형에 대해 훨씬 더 잘 일반화되어 실제 이미지에서 향상된 복원 성능을 제공하는 독특한 장점을 가진다.

## Adversarial Neural Degradation for Blind Super-Resolution

![](/img/AND/image.png)

1. 대부분의 이미지 품질 저하는 기본적인 CNN에서 대응되는 연산으로 찾을 수 있다.
2. 대부분의 이미지 품질 저하는 identity transformation에서 발생하는 작은 편차로 간주할 수 있다.

![](/img/AND/image%201.png)

먼저, 전체적인 NN을 3가지 부분으로 나눌 수 있다:

- Degradation network : CNN 기반 열화 생성
- Restoration network : ESRGAN 기반 SR 생성
- Discriminator network

그리고 네트워크 optimization step은 4가지의 세부 단계로 나눌 수 있다:

- 먼저, Degradation network를 identity transformation으로 초기화 한다.
- 초기화된 Degradation network에서 restoration network(generator)의 성능을 최악으로 만들 수 있는 작은 편차를 적대적으로 탐색한다.
- 찾아낸 Degradation case를 통해서 restoration network를 최적화한다.
- 마지막으로 결과물을 discriminator에 넣어 복원 이미지와 실제 이미지를 구별하도록 업데이트한다.

이 과정을 학습동안 반복하게 되며, 학습이 완료되면 restoration network만 추론에 사용하게 된다.

### Degradation Network Architecture

Degradation Network 아키텍쳐는 다양한 이미지 degradation을 포괄하기 위해 다음과 같이 표준 CNN을 사용하게 된다.

Convolution layer → filter related degradations를 표현 (blur, ringing 등)

Non-linear activation layer (LeakyReLU) → color changes을 표현 (색상 페이딩 등)

noise injection layer → block artifacts 표현

Pooling layer → downsampling을 표현

### Identity Degradation Network Initialization

Degradation 네트워크를 항등 변환을 수행하도록 빠르게 초기화하는 기법을 제안하였다.

이 초기화 방법을 통해서 단순한 랜덤 초기화나 gradient descent 학습 없이도 빠르고 정확하게 항등 변환을 구현할 수 있도록 고안되었다.

과정은 다음과 같다.

- LeakyReLU → slope = 1로 설정 (선형 함수)
- Noise Layer → 노이즈 값을 0으로 설정, 즉 입력 그대로 전달
- 3x3 conv layer에서 중앙 (1x1) 픽셀만 유효하게 만들고, 나머지는 0으로 설정 → 1x1 conv 효과
- Xavier Initialization을 사용하여 center weight 초기화, 마지막 conv lyaer의 center weight는 역행렬 계산을 통해 보정 → 여러 conv layer를 합쳐서도 항등이 보장
- Pooling은 downsampling이므로 strict한 항등이 불가, 따라서 anti-aliased average pooling을 사용해 시각적 항등으로 간주

이러한 과정을 거쳐서 최종적으로 초기화된 네트워크는 입력 이미지와 출력 이미지가 pixel-wise로 거의 동일할 것이다.

하지만 파라미터는 random한 초기화 방법보다 균형 잡힌 다양한 형태를 갖는다. 이후에 네트워크를 작게 perturbation하면서 다양한 degradation 케이스를 생성하게 됨.

### Adversarial Degradation Perturbation, Super-Resolution Model Training

앞서 초기화한 Identity Degradation Network를 의도적으로 악화 시켜, SR 모델이 가장 취약한 degradation 조건에서도 잘 작동하는 훈련 과정이다.

최악의 degradation 상황을 생성하여 이것까지 SR 모델이 복원할 수 있도록 하게 함으로써 SR 모델의 하한을 끌어올리는 것. 즉, 현실 환경에서의 강인성 향상이 이 과정의 목표이다.

논문에서는 ESRGAN을 채택하여 train을 진행하였다.

$$
\theta_F^* = \arg\max_{\|\theta_F - \theta_{\text{id}}\|_2 < \epsilon_{}} \mathcal{L}\left(G(F_{\theta_F}(I_{\text{HR}})), I_{\text{HR}}\right)

$$

네트워크에 대해 작은 변화 δ를 줘서 SR 모델 G의 성능으르 의도적으로 악화시키는 파라미터를 찾는 최적화이다. 

최종 최적화 문제는 다음과 같이 정의된다.

$$
\begin{align*}
\min_{\theta_G} \Big\{ \mathbb{E}_{I^{HR}} \Big[ \max_{\theta_F \in \mathcal{S}} L_{\text{cont}}(I^{HR}; \theta_G, \theta_F) \Big] 
+ \lambda \max_{\theta_D} \mathbb{E}_{I^{HR}} \Big[ \max_{\theta_F \in \mathcal{S}} L_{\text{GAN}}(I^{HR}; \theta_G, \theta_D, \theta_F) \Big] \Big\}
\end{align*}

$$

$$
\begin{align*}
L_{\text{cont}}(I^{HR}; \theta_G, \theta_F) &= \| I^{HR} - G_{\theta_G}(F_{\theta_F}(I^{HR})) \|_1 \\
&\quad + \sum_j c_j \| \phi_j(I^{HR}) - \phi_j(G_{\theta_G}(F_{\theta_F}(I^{HR}))) \|_2^2
\end{align*}

$$

$$
\begin{align*}
L_{\text{GAN}}(I^{HR}; \theta_G, \theta_D, \theta_F) = \log D_{\theta_D}(I^{HR}) - \log D_{\theta_D}(G_{\theta_G}(F_{\theta_F}(I^{HR})))
\end{align*}

$$

## Experiments

![](/img/AND/image%202.png)

![](/img/AND/image%203.png)

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
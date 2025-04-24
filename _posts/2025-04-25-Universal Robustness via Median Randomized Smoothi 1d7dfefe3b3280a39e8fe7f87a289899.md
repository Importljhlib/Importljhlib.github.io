---
layout: post
title: "Universal Robustness via Median Randomized Smoothing for Real-World Super-Resolution (CVPR 2024)"
date: 2025-04-25 01:53:00 + 0900
category: paper
---
## Introduction

이 논문은 Super-Resolution, SR 기술 중에서도 특히 real-world 이미지에 대한 Robust Super-Resolution 방법에 대해서 다루고 있다.

SR 분야에서 real-world image는 노이즈가 많기 때문에 기존 SOTA SR model들에게도 취약하였다. 

따라서 이 논문에서는 **Median Randomized Smoothing (MRS)**를 활용하여 다양한 적대적 공격과 real-world noise에 강건한 SR model을 개발하는 것을 목표로 하였다. adversarial attack을 SR에 적용하고 MRS를 활용한 새로운 RSR 모델인 **CertSR**을 제안하였다.

### Main contributions

- SR task에 대한 새로운 adversarial attack 제시
- RSR 모델인 CertSR 제안
- MRS의 보편적 Robust 입증

## Adversarial attacks and training on SR

### Adversarial attacks

![](/img/MRSGANSR/image.png)

**Fast Gradient Sign Method (FGSM)**

adversarial LR image를 만드는 데 주로 쓰이는 방법이다.

loss의 gradient를 사용해서 가장 효율적인 픽셀 변경 강도를 찾는다.

 

$$

x_{adv} = x + \epsilon \cdot sign(\nabla_x L(f_\theta(x), y)
$$

Loss는 L_percep과 L_1으로 구성된다. x는 LR 이미지, y는 HR GT, epsilon은 허용된 섭동의 step size(사람 눈엔 거의 안 보이는 노이즈의 정도)를 의미한다.

**Basic Iterations Method (BIM)**

$$
x_t=x_{t-1}+\alpha sign(\nabla_{x_{t-1}}L(f_\theta(x_{t-1}), y))
$$

BIM은 FGSM을 반복적으로 적용하는 것이다. 더 작은 크기의 alpha 만큼 여러 번 이동하게 된다. alpha = epsilon / T , T는 iterations 횟수를 뜻한다.

**Projected Gradient Descent (PGD)**

BIM을 일반화한 것이다. Uniform distribution U(-ϵ, ϵ)을 따르는 섭동된 LR 이미지로 초기화 한다.

$$
x_t=clip_{x, \epsilon }(x_{t-1}+\alpha sign(\nabla_{x_{t-1}}L(f_\theta(x_{t-1}), y)))
$$

clip 함수를 통해서, 전체 입력이 허용된 섭동 범위를 뜻하는 ϵ-ball 밖으로 벗어나지 않도록 한다.

**Carlini and Wagner attack (CW)**

CW 공격은 이미지에 대한 L2 norm을 최소화하는 최적의 적대적 perturbation을 찾는 것을 목표로 한다. perturbation의 크기를 제한하는 대신, loss를 최대화하는 방향으로 이미지를 공격한다. 수식은 다음과 같다.

$$
\min_{\delta} (\|\delta\|_2 - c \cdot \mathcal{L}(f_\theta(x), y)), \text{ such that } x + \delta \in [0, 1]^n
$$

delta는 adversarial perturbation을 의미한다. 적대적 이미지를 생성하는 변화량을 뜻한다.

그리고 L2 norm을 사용하여 섭동의 크기를 측정하며 이 크기를 최소화하여고 하는 것이다. c는 loss와 perturbation 크기 사이의 균형을 조절하는 하이퍼파라미터이다. delta는 다음과 같이 계산된다.

$$

\delta = \frac{1}{2} (\tanh(w) + 1) - x
$$

### Adversarial training

적대적 훈련은 MRS fine-tuning을 하기 전에 모델이 예제에 대해서 더 강건해지도록 학습시키는 과정이다. 적대적 훈련은 다음과 같이 이루어 진다. 

$$
\theta_{adv}^{*} = \operatorname{argmin}_{\theta \in \Theta} \frac{1}{N} \sum_{(x^{(i)},y^{(i)}) \in \mathcal{D}} \max_{\|\delta\|_2 \leq \epsilon} \mathcal{L}(f_\theta(x^{(i)} + \delta), y^{(i)}),
$$

학습을 통해서 최적의 모델 파라미터를 얻고자 하는 것이다.  max 연산을 통해서 L2 norm이 epsilon 이하인 perturbation delta에 대해서 loss를 최대화하는 perturbation을 찾는 것이다. Loss는 모델이 perturbation이 더해진 x에 대해 예측한 결과와 실제 정답 y 사이의 손실을 계산한다. D는 LR 이미지와 HR 이미지의 배치를 의미한다. argmin은 세타에 대해 손실을 최소화 하는 값을 찾는 연산이다. 

즉, adversarial example delta를 찾아서 모델의 예측과 실제 정답 간의 손실을 최대화하되, 동시에 모델 파라미터를 조정하여 이러한 adversarial example에 덜 민감하게 만드는 것을 목표로 한다. 

## The Main Method

![](/img/MRSGANSR/image%201.png)

### Median Randomized Smoothing (MRS)

MRS의 주요 기법은 다음과 같다.

가장 먼저 LR image에 특정 표준 편차를 가지는 Gaussian noise를 추가하여 이미지 샘플을 얻는다. 그 후에 이미지 픽셀 단위로 중앙값을 얻게 된다. 

DNN $g_{\theta}:R^n->R$ 의 percentile smoothing은 다음과 같이 정의된다.

$$
\begin{aligned}
\bar{q}_{p}(x) &=\inf \left\{y \in \mathbb{R} \mid \mathbb{P}\left(g_{\theta}(x+G) \leq y\right) \geq p\right\}, \\
\underline{q}_{p}(x) &=\sup \left\{y \in \mathbb{R} \mid \mathbb{P}\left(g_{\theta}(x+G) \leq y\right) \leq p\right\}.
\end{aligned}

$$

먼저 q_p(x)는 smoothed 함수를 나타낸다. p는 분위수 값을 나타내는데, 함수는 입력 x에 대해서 특정 분위수 p에서의 smoothed된 값을 출력하게 된다.

- inf : x에 가우시안 노이즈 G를 더한 값을 모델에 통과시킨 결과가 y보다 작거나 같을 확률이 p이상인 y들의 집합에서 가장 작은 값을 찾게 된다.
- sup : 마찬가지로 y보다 작거나 같은 값들 중 p 이하인 y들의 집합에서 가장 큰 값을 찾게 된다.

p=0.5일 때는 median 값과 같다.

### Median Randomized Smoothing for SR

CertSR (Certified Super-Resolution) 이라고 불리는 RSR model을 만들기 위해서 다음과 같은 과정을 거친다.

처음으로 깨끗한 LR 이미지에 대해 pre trained 된 GAN (Generative Adversarial Network)를 기반으로 초기 SR 모델을 구현하게 된다.

다음으로는 MRS fine-tuning 과정이다. LR 이미지에 각 가우시안 노이즈 샘플을 더하게 된다. 이 이미지들을 Generator에 넣어서 여러 개의 초해상도 이미지를 생성하게 되는데, 이들을 픽셀별 중앙값을 계산하여 하나의 통합된 이미지를 얻게 된다. 그 후에는 Discriminator을 통해서 계산된 중앙값과 HR 이미지 간의 차이를 학습하게 된다.

Loss 계산은 다음과 같은 함수들로 이루어진다.

- L1 Loss : 픽셀 단위의 차이 측정
- Perceptual Loss : feature map을 기준으로 예측 이미지와 정답 이미지의 차이를 계산
- Adversarial Loss : Generator가 Discriminator를 속이기 위한 손실

마지막으로 MRS_Inference라고 불리는 단계를 진행한다. LR 이미지에 가우시안 노이즈를 추가, 초해상도 이미지 생성, MRS를 통해서 하나의 최종 HR 이미지를 얻는다. 

![](/img/MRSGANSR/image%202.png)

![](/img/MRSGANSR/image%203.png)

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
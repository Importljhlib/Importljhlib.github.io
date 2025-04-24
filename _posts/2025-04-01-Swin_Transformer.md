---
layout: post
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021)"
date: 2025-04-01 01:53:00 + 0900
category: paper
---

## Introduction

기존의 Transformer 모델을 vision task에 적용 시켰을 때의 문제점은 다음과 같다.

- 이미지 데이터는 해상도가 계속해서 변화하는데, 기존의 transformer 모델은 토큰의 크기가 고정되어 있다.
- CV에서는 고해상도 이미지를 처리해야 하기 때문에 연산량이 기본적으로 많고, computational complexity가 이미지 사이즈에 대해서 2차 함수로 증가하기 때문에 이미지가 커지면 발산하게 된다.

따라서 이 논문에서는 위의 문제를 극복하기 위해 Hierarchical Feature Map을 구성하고 선형 계산 복잡도를 가지는 Swin Transformer를 제안하였다.

![](/img/swin/image.png)

기존의 ViT는 이미지를 작은 patch 단위로 쪼개는 방식을 사용했다면, Swin Transformer에서는 처음에 더 작은 patch 단위로 시작해서 점점 patch들을 merge하는 방식으로 흘러가면서 계층적인 구조를 가지게 된다.

![](/img/swin/image%201.png)

또한 Swin Transformer는 Shifteed Window 기법을 사용하여 연산을 하게 된다.

먼저 왼쪽과 같이 단순히 window를 나눌 수 있다. 하지만 이런 방법은 window 내부의 patch끼리만 self-attention을 계산하기 때문에 경계 근처의 patch끼리의 연결성이 떨어지게 된다. 

이러한 문제점을 극복하고자 오른쪽 그림과 같이 shifted window partitioning을 통해서 window 간의 연결성을 반영할 수 있다.

## Method

### Overall Architecture

![](/img/swin/image%202.png)

- HxWx3 크기의 이미지가 input으로 들어가게 된다.
- 이 이미지를 패치 단위로 나누게 된다. 각 패치의 크기는 4x4이므로 4x4x3(RGB 채널) = 48의 차원을 가지게 된다.
- Linear Embedding 레이어를 통과하게 되는데, 이 토큰들의 feature들이 디멘션 C로 project된다. (48 → C) 그리고 아웃풋은 H/4 x W/4 x C가 된다.
- Patch merging을 진행하게 된다. [(H/4 x W/4) → (H/8 x W/8)], 2 x 2만큼의 주변 패치의 feature를 concat한 후에 Linear layer를 거치게 된다. (C → 4C → 2C) 즉, 토큰 수는 4배로 줄어들고 채널의 수는 2배만큼 늘어나게 된다.
- 이를 반복하고 마지막엔 MLP에 들어가게 된다.

### Shifted Window based Self-Attention

**Self-attention in non-overlapped windows**

![](/img/swin/image%203.png)

일반적인 MSA는 이미지 크기에 대해서 (hw)^2로 2차이다. 

하지만 W-MSA는 M의 크기가 고정되어 이미지 크기에 따라서 선형적으로 증가한다.

**efficient batch computation for shifted configuration**

![](/img/swin/%EC%A0%9C%EB%AA%A9_%EC%97%86%EC%9D%8C.png)

위에서 언급했다시피 윈도우가 고정되어 윈도우 내에서만 self-attention을 계산하기 때문에 윈도우 간의 연결성이 부족하다는 단점이 있었다. 

또한 일반적인 W-MSA는 2x2 window 계산이 3x3으로 증가하게 된다.

그래서 이러한 문제를 해결하기 위해서 **Shifted Window MSA** 기법을 제안하였다. 

먼저 M//2 개의 Window를 우측 하단으로 회전 이동 시킨다. **(Cyclic Shift)**

그 후 중복 attention 연산을 제한하기 위해 각 sub-window에 masked self-attention을 진행하게 된다.

그 후 원래대로 reverse cyclic shift로 원위치 시킨다.

이렇게 하면 원래 3x3 계산이 2x2로 줄어들게 되고, 윈도우 간의 연결성을 유지할 수 있다.

## Experiments

ImageNet-1k classification, COCO object detection, ADE20K semantic segmentation에 대해서 실험을 진행하였다.

ImageNet

![](/img/swin/image%204.png)

COCO

![](/img/swin/image%205.png)

ADE20K

![](/img/swin/image%206.png)

## Conclusion

Swin Transformer는 COCO object detection과 ADE20K semantic segmentation에서 SOTA를 달성할 수 있었다.

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
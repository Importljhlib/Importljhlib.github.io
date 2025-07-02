---
layout: post
title: "Motion Hologram: Joint opimized hologram generation and motion planning for photorealistic 3D displays via reinforcement learning (Science Advances 2025)"
date: 2025-07-03 01:53:00 + 0900
category: paper
---
![](/img/motion_hologram/image.png)

## Introduction

본 논문은 photorealistic하고 speckle-free한 3D 디스플레이를 구현하기 위해서 **Motion Hologram** 기술을 제안한다. 

CGH (Computer-Generated Holography)의 장점은 픽셀 단위의 깊이 제어, 광학 왜곡 보정, AR/VR/교육 등 다양한 분야에서 활용 가능성에 있다. 하지만 다음과 같은 기존의 문제점들이 존재한다.

- SLM의 공간-주파수 대역폭(Space Bandwidth Product) 한계
    - defocus
    - speckle noise
    - 비자연스러운 재현 품질 발생
- 하드웨어-소프트웨어 통합의 부재
    - 기존은 하드웨어 설계와 알고리즘이 따로 최적화
    - 새시도로 end-to-end differntiable framework로 시스템 전체를 최적화하였지만, **모든 하드웨어가 미분 가능하지 않다는 점**과 **단일 하드웨어 구성**만 고려한다는 부분이 문제점이었다.

따라서 본 논문에서는 위와 같은 문제점을 해결하기 위해서 다음과 같은 기법을 제안하였다.

- 강화학습(RL)의 도입
    - 미분이 불가능한 시스템에도 적용 가능
    - 보상 기반으로 복잡한 설계 공간을 탐색

→ 이를 통해 시스템 설계와 알고리즘을 공동 최적화

- Motion Hologram
    - 기존에는 motion이 노이즈의 원인 → **motion을 오히려 노이즈 제거에 활용**
        - hologram을 조금씩 움직이면서 다중 이미지 생성 → 평균화
        - 이 과정을 통해 스펙클을 줄이고 고품질 3D 영상 생성
    - RL로 jointly optimize
        - phase-only hologram과 motion trajectory를 동시에 최적화

이를 다시 정리하면, 의도적인 움직임이 광원의 spatial coherence를 와해키셔 홀로그램 재구성 시 발생하는 speckle을 줄일 수 있다. 이를 위해 RL을 사용하여 phase-only hologram 생성과 시스템의 motion trajectory를 공동으로 최적화하는 것이다.

## Results

### Motion Hologram

Motion Hologram의 핵심 아이디어는 Motion을 활용해 speckle을 없애고 하나의 phase만으로 고품질 3D를 재현하는 것이다.

기존의 방법처럼 여러 hologram을 쓰는 것이 아닌, 하나의 hologram을 spatial하게 이동시켜 다수의 이미지를 생성하여 합성하면 speckle이 줄어드는 것이다.

**[Joint Optimization Stage]**

![](/img/motion_hologram/image%201.png)

Joint Optimization Stage는 강화학습 기반 에이전트가 SLM의 이동 경로와 함께 홀로그램까지 jointly 최적화하는 과정이다. 이 과정이 끝나면 이후 새로운 장면에 대해서는 motion 경로는 고정하고 hologram만 개별 최적화하게 된다.

 

심층 에이전트($\pi_\theta$)가 Motion Holography 환경과 상호작용하며 일련의 action(*a*), state(*s*), reward(*r*)를 통해 motion planning과 hologram 생성을 공동으로 최적화한다.

**state 초기화**

초기의 state는 SLM의 초기 위치와 step count가 0임을 뜻한다.

에이전트는 이 상태를 보고 다음 action을 결정하게 된다.

또한 초기 홀로그램 $\phi_{motion}$이 설정된다. (보통 랜덤 위상)

**Actor Network → action 생성**

actor는 현재 상태 $s_{k-1}$을 받아서 방향 d(up, down, left, right)와 이동 거리 p(1, 2, 3)을 조합한 **action $a_k$ = (d, p)**를 선택하게 된다.

**Environment Update - Motion Planning**

선택된 $a_k$에 따라 시스템의 모션 계획, 즉 SLM의 예상 움직임 궤적(Trajectory)이 업데이트 된다. 이전 스텝들에서 결정된 움직임에 새로운 움직임 $a_k$가 추가되어 현재까지의 누적 궤적이 된다.

**Environment Update - Hologram Optimization**

새롭게 업데이트된 전체 모션 궤적 ( { a1, . . . ak} )를 기반으로 해당 궤적을 따라 움직였을 때 가장 좋은 결과를 내는 phase-only hologram $\phi_{motion}$을 다시 최적화한다.

최적화하는 과정은 다음과 같다.

$$
|u^j_{motion}| = \frac{1}{K}\sum_{k}|f_{model} \{ Trajectory(\phi_{motion}), z^j\}|
$$

먼저 위 수식과 같이 깊이 z^j에서 최종적으로 재구성된 이미지의 강도를 구한다.

$$
\phi_{motion} = argmin_{\phi} \mathcal{L}(|u^j_{motion}|, \sqrt{I^j_{target}})
$$

그 후에 target 이미지와 u 사이의 손실을 계산하여 가장 최적의 phi를 찾게 되는 것이다.

**Environment Feedback - Reward and Next State**

최적화된 홀로그램과 현재까지의 모션 궤적을 사용하여 3D 장면을 재구성하고 평가한다.

이 결과로부터 즉각적인 보상 $r_k$가 다음과 같이 계산된다.

$$
r_k = r^{all}_k + W \times r^{in-focus}_k
$$

보상은 현재 스텝에서 얻은 전체 장면 및 초점영역의 PSNR 개선 정도를 바탕으로 결정된다.

다음 스텝을 위한 새로운 state s_k가 생성된다.

**Agent Learning**

계산된 보상 r_k와 새로운 상태 s_k는 에이전트로 다시 전달된다.

에이전트의 Critic network는 받은 보상을 바탕으로 엑터 네트워크가 선택한 행동의 가치를 평가한다.

에이전트는 강화 학습 알고리즘을 사용하여 이러한 경험으로부터 정책을 업데이트하게 된다. 정책 업데이트의 목표는 최종적으로 누적 보상 R을 최대화하는 것. 즉 모션 궤적을 학습하는 것이다.

**[Optimization of hologram with fixed motion trajectory]**

![](/img/motion_hologram/image%202.png)

이 단계는 앞에서 강화학습을 통해 최적의 모션 궤적이 학습된 후, 그 고정된 궤적을 사용하여 새로운 3D 장면에 대한 최적의 홀로그램을 생성하는 과정이다.

최적화하려는 phase-only 홀로그램을 고정된 경로에 따라 디지털 방식으로 이동시키게 된다. 만약 8개의 스텝이 있다면 8개의 조금씩 다른 위치에 동일한 홀로그램의 복사본이 생성되는 것이다. 그리고 각각을 3D 이미지를 형성하는 과정을 시뮬레이션한다. 이 때 각 이미지들은 동일한 motion speckle을 갖지만 위치가 약간씩 다를 것이다.

생성된 이미지를 하나로 합쳐 평균을 내면 스페클 노이즈가 효과적으로 상쇄되고 화질 향상이 일어난다.

GT와 비교하여 loss를 계산, SGD로 위상값을 업데이트한다.

## Simulation results

![](/img/motion_hologram/image%203.png)

순서대로 GT, Traditional holography, Proposed holography, TM holography이다.

![](/img/motion_hologram/image%204.png)

정량적으로 봤을 때도 좋은 성능을 보여준다.

![](/img/motion_hologram/image%205.png)

최적화된 홀로그램을 무작위 경로로 움직이거나, 기존 홀로그램을 최적화된 경로로 움직였을 때 모두 최종 결과물보다 품질이 떨어졌다. 즉 홀로그램과 움직임 경로를 함께 최적화하는 것이 좋은 성능을 보여준다.

![](/img/motion_hologram/image%206.png)

실제 하드웨어 기반 실험 결과

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
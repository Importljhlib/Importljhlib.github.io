---
layout: post
title: "Full-colour 3D holographic augmented-reality displays with metasurface waveguides"
date: 2025-06-01 01:53:00 + 0900
category: paper
---
![](/img/colour3dholo/image.png)

## Introduction

증강현실(Augmented Reality AR)은 교육, 엔터테이먼트, 원격 협업 시각 연구 등 다양한 분야에서 현실 세계에 디지털 정보를 겹쳐 보여주는 방식으로 인간-기계 인터페이스의 미래를 선도하고 있다. 하지만 현재의 AR 기기의 문제점들 중 가장 큰 원인은 디스플레이 장치의 **광학계가 크고 무겁다는 것**. 그리고 **3차원 입체감을 제대로 표현하지 못한다는 점**이다.

기존의 AR 안경들은 마이크로 디스플레이(OLED, MicroLED 등)와 투사 렌즈(optics)를 사용하는 구조인데, 이는 렌즈의 초점 거리만큼의 공간이 필요해 전체 시스템이 두껍고 무거워질 수밖에 없다. 또한 이러한 구조는 디지털 이미지를 고정된 거리에서만 출력할 수 있어 사용자에게 현실적인 깊이 정보를 제공하지 못하고 Vergence-Accommodation Conflict를 일으켜 장시간 사용 시 눈의 피로를 유발하게 된다. 

이에 본 논문에서는 inverse-designed meatasurface waveguide, 렌즈 없는 holographic light engine, AI 기반 wave propagation model, 이 세 가지 요소들을 결합하여 새로운 AR 시스템을 제안한다.

**waveguide 기반 홀로그램 AR의 기본 구조**

기존 AR 디스플레이는 주로 OLED/LED 등의 amplitude modulator를 기반으로 하며 이는 단순한 2D 콘텐츠를 고정된 거리에서 표시하는 데 그친다. 하지만 이 논문에서는 phase-only SLM (Spatial Light Modulator)를 사용하여 각 화소에서 빛의 위상만 조절하고 그 위상 정보를 기반으로 홀로그램을 생성한다.

이를 통해 렌즈 없이도 빛을 조절할 수 있으며 입체적인 이미지 생성이 가능하다. 여기서 중요한 점은 phase-only SLM이 **입사 광선과 매우 가까운 위치에 설치될 수 있어 시스템을 매우 얇게 설계**할 수 있다는 점이다.

이때 waveguide를 통해 빛을 유도하며 유리와 같은 투명한 매질을 통해 빛이 내부에서 전반사되며 진행하도록 설계된다. 이 waveguide는 사용자의 눈으로 빛을 전달하는 경로가 된다.

이 waveguide는 metasurface grating coupler에 의해 입사/출사하는데, 이 grating의 주기와 구조를 통해 **RGB 각각의 파장을 동일한 지점으로 맞추기 위해** 복잡한 설계가 필요한 것이다.

## Inverse-designed metasurface waveguide

본 논문에서 제안하는 metasurface waveguide는 기존 waveguide 방식보다 훨씬 정밀하고 효과적인 3D 콘텐츠 전달을 가능하게 한다. 핵심은 waveguide 내에서 RGB 빛을 동일한 위치에서 정확하게 결상되도록 제어하는 데 있으며, 이를 위해 inverse design 기법을 통해 metasurface grating 구조를 최적화하였다.

먼저 빛이 waveguide 내에서 유지되기 위해서는 전반사 조건(Total Internal Reflection)을 만족해야 한다. 이를 위해서는 빛의 파장  λ, 굴절률 n에 따라서 임계각 θ_c가 결정되고, 

$$
\theta_c(\lambda)=sin^{-1}(\frac{1}{n(\lambda)})
$$

이를 기반으로 RGB 파장을 모두 전반사 시켜 하나의 grating coupler로 통과시키려면 **n ≥ 1.8** 이상인 유리가 필요하며, 본 논문에서는 고굴절률 유리(SF6)를 사용하여 이를 충족한다.

![](/img/colour3dholo/image%201.png)

각 RGB 파장은 waveguide 내에서 서로 다른 반사 횟수를 가지므로 최종적으로 동일 지점에 결상되게 하기 위해 최소공배수 조건을 설계한다. 이때 waveguide 내 pupil의 위치 이동량은 다음과 같이 정의된다.

$$
l(\lambda)=\frac{2d_{wg} \cdot \tan(\theta)}{\Lambda}
$$

$$
\exists \ d_{wg}, \Lambda : LCM(l(\lambda_{R}), l(\lambda_G). l(\lambda_{B})) < L_{wg}
$$

l(λ)는 각 파장이 이동하는 lateral 거리이며, LCM 조건은 파장 간 일치를 위한 최소공배수 조건이다.

이 조건을 만족시키면 모든 파장이 동시에 동일 위치에서 출사될 수 있다.

![](/img/colour3dholo/image%202.png)

## Waveguide propagation model

![](/img/colour3dholo/image%203.png)

SLM을 통과한 광파는 in-coupler에서 waveguide로 들어가고 , 다시 out-coupler를 통해 출사된다. 이를 수학적으로 모델링하면 다음과 같다.

**입사파 수식**

$$
u_{IC}(e^{i\phi})=e^{-i \frac{2\pi}{\lambda} \sqrt{x^2+y^2+f^2_{illum}}}e^{i\phi}a_{IC}
$$

이 수식은 SLM에 적용된 위상 패턴 (e^i*phi)에 따라 waveguide 안으로 결합되는 빛의 복소 진폭 분포인 wavesurface를 나타낸다.

**waveguide 내 전파**

$$
u_{OC}(e^{i\phi})=a_{OC} \int \int \mathcal{F}(u_{IC}(e^{i \phi})H_{WG}e^{i2\pi(f_xx+f_yy)}df_xdf_y)
$$

이 수식은 out-coupler에서 방출되는 빛의 wavefront(파면)이다. 입력 SLM 패턴 (e^i*phi)에 의해 결정된 wavefront가 waveguide를 통과한 후의 결과이다.

이 수식의 물리적 과정은 다음과 같다.

- 먼저 SLM 패턴에 의해 생성된 빛의 wavefront가 in-coupler를 통해 waveguide 안으로 결합된다. → u_IC(e^i*phi)
- 이 wavefront는 푸리에 변환을 통해 주파수 영역으로 변환된다.
- 주파수 영역에서 waveguide의 물리적 특성을 나타내는 전달 함수 H_WG가 이 wavefront에 적용된다. 이는 빛이 waveguide 내부를 전파하면서 겪는 변화를 모델링하는 것이다.
- 변환된 주파수 영역의 결과는 Inverse 푸리에 변환을 통해 다시 공간 영역의 wavefront로 변환된다.
- 마지막으로 out-coupler의 aperture 함수 a_OC가 이 wavefront에 곱해져 최종적으로 사용자 눈으로 향하는 out-coupler field가 얻어지는 것이다.

**자유공간 전파 (사용자 눈 앞 이미지)**

AR 디스플레이 시스템에서 사용자가 특정 깊이에서 보게 될 최종 이미지를 계산하는 법을 나타낸다.

$$
f_{WG}(e^{i\phi},d_{target}) = 
f_{free}(u_{OC}(e^{i\phi}), d_{target})
$$

하지만 waveguide 내에서 빛의 파동 전파를 물리적으로 정확하게 모델링하는 것은 현실적으로 매우 어렵다. 왜냐하면 시뮬레이션 모델과 실제 광학 시스템 간에 나노미터 수준의 미세한 차이가 존재하기 때문이다. 이러한 미세한 차이는 결과적으로 관찰되는 홀로그램 이미지의 품질을 크게 저하시킨다. 

따라서 이러한 시뮬레이션 모델과 실제 광학계 간의 불일치를 설명하기 위해 모델에 학습 가능한 구성 요소를 추가하였다. 

먼저 모델은 카메라 피드백을 사용하여 자동으로 보정되며, 이 과정에서 다음을 학습하게 된다.

- a_IC와 a_OC 하이퍼파라미터 : in-coupler 및 out-coupler의 조리개 함수를 복소수 필드로 학습한다. 이는 빛이 waveguide 안팎으로 커플링될 때 발생하는 실제 효율 및 공간적 변동을 반영한다.
- 공간적으로 변화하는 회절 효율 : metasurface 격자의 회절 효율이 위치에 따라서 다를 수 있는 부분을 학습한다.
- CNN : in-coupler 및 목표 평면에 CNN을 적용하여 시뮬레이션 모델이 포착하지 못하는 물리 광학계의 잔여 수차나 미세한 효과를 학습하고 보정한다.

$$
u_{IC}(e^{i\phi})=CNN_{IC}(e^{-i \frac{2\pi}{\lambda} \sqrt{x^2+y^2+f^2_{illum}}}e^{i\phi}a_{IC})
\\
f_{WG}(e^{i\phi}, d_{target}) = CNN_{target}(f_{free}(u_{OC}(e^{i\phi}), d_{target}))
$$

## Experimental results

먼저 사용된 장비들은 다음과 같다.

- Phase-only SLM : HOLOEYE LETO-3, 1080x1920 해상도
- 광원 : FISBA READYBeam (RGB Laser)
- 카메라 : FLIR Grasshopper3 + Canon 35mm 렌즈

![](/img/colour3dholo/image%204.png)

CGH 생성 모델에 따른 2D 이미지 품질 비교이다. AI 보정 모델이 naive 모델이나 물리 기반 모델보다 더 높은 성능을 보여준다는 것을 알 수 있다.

![](/img/colour3dholo/image%205.png)

이 결과는 초점 거리에 차이에 따른 3D 입체감 재현 능력을 평가한 것이다.

![](/img/colour3dholo/image%206.png)

마지막으로 현실 장면과 디지털 객체가 함께 등장하는 AR 장면을 촬영한 이미지이다.

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
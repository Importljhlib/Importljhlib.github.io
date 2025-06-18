---
layout: post
title: "Color Contrast Enhanced Rendering for Optical See-through Head-mounted Displays (IEEE Transactions 2021)"
date: 2025-06-19 01:53:00 + 0900
category: paper
---
![](/img/Color_Contrast_Enhanced/image.png)

이 논문은 **OST-HMD (Optical See-through Head-mounted Displays)**에서 발생하는 혼합 문제를 해결하기 위한 실시간 렌더링 기법을 제안한다. OST-HMD는 물리적 배경과 가상 객체를 광학 결합기 (optical combiner)를 통해 사용자에게 동시에 보여주는데 이 과정에서 가상 객체의 렌더링된 색상이 배경 색상과 혼합되어 가시성이 저하되는 문제가 발생한다. 기존의 소프트웨어 기반 색상 보정 방법은 주로 배경 색상을 빼는 방식(subtraction compensation)을 사용하지만, 이는 가상 콘텐츠의 밝기를 감소시켜 구별성을 떨어뜨릴 수 있다.

![](/img/Color_Contrast_Enhanced/image%201.png)

이 논문에서는 인간 시각 시스템(HVS)의 색상 지각 특성. 특히 simultaneous color induction 현상에 주목한다. 이 현상은 한 영역의 색상이 인접 영역의 색상 지각에 complementary color (보색) 방향으로 영향을 미치는 것을 의미한다. 논문은 이러한 원리에 기반하여 가상 객체와 실제 배경 간의 색상 대비를 높이는 접근 방식을 채택한다. 핵심 아이디어는 가상 객체의 디스플레이 색상을 배경 색상에 대해 인지적으로 더 대비되도록 최적화하되, 원래 의도된 색상과의 일관성을 유지하는 것이다.

![](/img/Color_Contrast_Enhanced/image%202.png)

수학적으로 OST-HMD에서 사용자가 인지하는 색상 c는 HVS의 작용 H에 의해 결정되며 이는 디스플레이 빛 l_d와 배경 빛 l_bg이 혼합된 빛 l_bl로 부터 온다. 즉 c = H(I_bl). 

OST-HMD 시스템 DMR의 작용은 I_bl = DMR( I_d, I_bg )로 표현되나 I_bl ≒ I_d + I_b로 근사한다. 여기서 I_b는 디스플레이에 투과되어 보이는 배경 빛이다. 인지된 색상 차이를 극대화하기 위해 CIELab 색 공간(LAB space)를 사용하며, 색상 차이를 다음과 같이 정의한다.

$$
∆E∗ab(x, y) = ||x − y||=
√
(L∗
x − L∗
y )2 + (a∗x − a∗
y )2 + (b∗
x − b∗
y )
$$

목표는 배경 색상 l_b와 최적의 디스플레이 색상 l_opt 간의 색상 차이인 delta E*_ab( l_opt, l_b)를 제약 조건 하에서 최대화하는 최적의 l_opt를 찾는 것이다.

$$
l_{opt} = \arg \max_{l_{opt}} \Delta E^*_{ab}(l_{opt}, l_b) \quad \text{subject to constraints}.
$$

배경 색상 l_b에 대한 가장 큰 색상 차이는 l_b의 보색 방향에 존재하지만 이 방향으로 무한정 이동하면 원래 색상이 크게 왜곡된다. 따라서 다음과 같은 네 가지의 제약 조건이 도입된다.

- **Color Difference Constraint** : 최적 색상 l_opt와 원래 디스플레이 색상 l_d간의 색상 차이가 임계값 lambda_E 이어야 한다. ∆E∗_ab( l_opt, l_d ) ≤ lambda_E
- **Chroma Constraint** : 최적 색상 l_opt의 채도 ch_opt가 원래 색상 l_d의 채도 ch_d보다 작아지지 않아야 한다. (ch_opt - ch_d ≥ 0)
- **Luminance Constraint** : 최적 색상 l_opt와 원래 색상 l_d 간의 휘도 차이  ∆L*가 임계값 lambda_L이하여야 한다. (∆L*(l_opt, l_d) ≤ lambda_L)
- **Just Noticeable Difference Constraint** : 최적 색상 l_opt와 배경 색상 l_b 간의 색상 차이가 최소 식별 차이 (JND, Just Noticeable Difference) 임계값 lambda_JND 이상이어야 한다. (∆E∗_ab( l_opt, l_d ) ≥ lambda_JND)

## Algorithm

![](/img/Color_Contrast_Enhanced/image%203.png)

제안된 실시간 알고리즘은 크게 세 단계로 구성된다.

### Preprocessing

preprocessing 과정에서는 Gaussian blur와 FoV calibration을 수행하게 된다.

**Gaussian blur**

![](/img/Color_Contrast_Enhanced/image%204.png)

가우시안 블러는 HVS의 특정 특성을 모방하고 배경 영상의 노이즈를 줄이기 위해 수행한다. HVS는 공간 주파수 채널을 통해 이미지를 처리하는데, 색상 인지는 저주파 특성을 가진다. 또한 non-focal field의 경우 흐릿하게 인식되는 영향이 있다. 가우시안 블러의 이러한 HVS의 저주파 특성 및 비초점 영역의 흐림을 시뮬레이션하여 배경의 공간 색상 정보를 추출하고 세부 정보를 걸러낸다.

따라서 가우시안 블러를 통해 인접한 배경 색상의 평균을 계산하여 해당 영역의 배경색을 나타내도록 한다. 이는 최적화된 색상에서 발생하는 flicker 현상을 줄이는 데 도움이 된다.

**FoV calibration**

![](/img/Color_Contrast_Enhanced/image%205.png)

FoV calibration 과정은 OST-HMD의 배경 카메라 시야각(Field of View, FoV)과 디스플레이 시야각 간의 차이를 보정하는 것이다. 배경 카메라는 일반적으로 HMD 디스플레이보다 더 넓은 FoV를 가질 수 있다. 따라서 카메라로 캡쳐된 배경 영상의 픽셀과 HMD에 렌더링되는 가상 객체의 픽셀을 정확하게 일치시키기 위한 보정이 필요하다. 

배경 영상을 렌더링 시스템의 frame buffer에 매핑하기 위한 2D 화면 공간 좌표 매핑을 사용하게 된다. $u = s_ui + b_u \ ,  v = s_vj + b_v$인 선형 변환으로 근사된다. 여기서  (u, v)는 프레임 버퍼의 2D 텍스처 좌표이고, (i, j)는 배경 영상의 좌표이다. (s_u, s_v)는 수평 및 수직 방향의 스케일 인자이고, (b_u, b_v)는 해당하는 오프셋이다.

이 캘리브레이션을 통해 배경 영상의 저주파 정보가 OST-HMD 디스플레이를 통해 보이는 실제 장면의 저주파 정보와 가능한 한 유사하게 정렬된다. 이는 이후의 색상 최적화 단계에서 가상 객체 픽셀에 대응하는 배경 색상을 정확하게 파악하는 데 중요하다.

### Conversion

디스플레이 색상과 배경 색상을 RGB 색 공간에서 LAB 색 공간으로 변환한다. LAB space는 인지적으로 균등하며 휘도와 색도가 분리되어 있어 최적화에 용이하다. 최적화를 한 후에는 다시 LAB에서 RGB로 변환하여 출력한다. LAB space는 [-1, 1] 범위로 단순화하게 스케일링된다.

### Optimization

![](/img/Color_Contrast_Enhanced/image%206.png)

스케일링된 LAB space에서 blurred된 배경 색상 I_b와 원래 디스플레이 색상 I_d로부터 최적 색상인 I_opt를 찾게 된다.

이상적인 최적 색상 I는 B에서 가장 먼 점으로 계산된다.

$$
I = -norm(B)
$$

Color Difference Constraint를 적용하여 D에서 I방향으로 이동하되, 거리가 λ’_E로 제한된 점 E를 찾는다.

$$
\overset{\rightarrow}{DE} = \min(dist(D, I), \lambda'_E) \cdot norm(\overset{\rightarrow}{DI})

$$

다음으로는 Chroma Contraint를 적용한다. DE의 a*b* 평면 투영인 DE’를 분해하여 채도 변화 성분 DE’_ch와 색조 변화 성분 DE’_h를 얻게 된다. OD’과 DE’ 사이의 각도 θ_ch에 따라 채도 감소 성분은 적응적으로 제거된다.

$$
\overset{\rightarrow}{DC} = t_{ch} \cdot \overset{\rightarrow}{DE'_{ch}} + \overset{\rightarrow}{DE'_h}
$$

Luminance Constraint를 적용한다. DE의 L축 성분 DE_L은 L축과의 각도 θ_l에 따라 채도 감소 성분은 적응적으로 감쇠된다.

$$
\overset{\rightarrow}{DL} = (1-|\cos{\theta_l}| \cdot \overset{\rightarrow}{DE_L})
$$

최종 최적 색상 P는 D에 제약 조건을 거친 변화 벡터들을 더하여 얻게 된다. (P = D + DC + DL)

Just Noticeable Difference Constraint는 예외적으로 dist(P, B) <  λ’_JND인 경우에만 적용되며, P를 B로부터  λ’_JND만큼 떨어진 점으로 조정하게 된다.

## Results

본 논문은 Unity 엔진을 사용하여 구현되었으며, 시뮬레이션 환경과 Microsoft HoloLens 상에서 평가되었다.

![](/img/Color_Contrast_Enhanced/image%207.png)

실험 결과, 가상 객체와 배경 간의 인지적 대비를 효과적으로 향상시켜 가시성을 높였으며 이와 동시에 원래 색상과의 일관성도 잘 유지함을 보였다. 특히 λ’_E값 조절을 통해 대비와 일관성 사이의 trade-off를 관리할 수 있음을 보여주었다. 

![](/img/Color_Contrast_Enhanced/image%208.png)

기존 방법들과의 비교에서, subtraction compensation보다 우수한 구별성과 자연성을 보여주었다. 

![](/img/Color_Contrast_Enhanced/image%209.png)

visibility-enhanced blending보다는 더 나은 객체 내부 대비와 자연성을 제공했다. 본 알고리즘은 real-time 성능을 보이며, 하드웨어 변경 없이 적용 가능하다.

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
---
layout: post
title: Tranformer - Attention Is All You Need
date: 2025-03-06 01:53:00 + 0900
category: paper
---
# Tranformer : Attention Is All You Need

## Model Architecture

![](/img/attn1.png)

### Encoder and Decoder Stacks

Endcoder는 총 6개의 동일한 레이어로 구성되어 있고, 2개의 sub-layer로 이루어져 있다.

Multi-Head Attention → add and norm → feed forward → add and norm

residual connection을 한 다음에 layer norm을 하게 된다.

Decoder도 역시 6개의 동일한 레이어로 구성되어 있고, 2개의 sub-layer 이외에 또 다른 layer 하나를 더 가진다.

Masked Multi-Head Attention → add and norm → Multi-Head Attention → add and norm → feed forward

마찬가지로 Residual connection이 존재하고, decoder가 출력을 생성할 때 다음 출력에서 정보를 얻는 것을 방지하기 위해 masking을 사용한다. 이는 i번째 원소를 생성할 때 i 보다 작은 원소만 참조할 수 있게 해준다.

### Attention

**Scaled Dot-Product Attention**

![](/img/attn2.png)

Scaled Dot-Product Attention은 말 그대로 Attention을 하는데에 내적을 쓴 뒤, scaling하는 것이다. 

input으로 각각 d_k와 d_v 차원인 query와 value로 구성된다. 

물어보는 주체인 query가 들어오고, 각각의 attention을 수행할 key가 들어오게 된다. 그리고 이 값들을 scale, softmax를 해준 뒤에 value와 곱해주면 최종적으로 attention value를 얻게 된다. 

![](/img/attn3.png)

query와 모든 key를 내적한 뒤 루트 d_k로 나누어 준다. 그리고 V(value)에 softmax 가중치를 곱해주어 최종 출력을 얻게 된다.

**Multi-Head Attention**

![](/img/attn4.png)

Multi-Head Attention은 각각 h개의 V, K, Q를 병렬적으로 attention을 적용하는 것이다.

입력값과 출력값의 차원이 같아야 하기 때문에 마지막에 concat을 해주고 FC를 거쳐서 최종적으로 출력하게 된다.

수식은 다음과 같다.

![](/img/attn5.png)

## 동작 원리

![](/img/attn6.png)

“I love you”라는 문장이 있고, 임베딩 차원이 4차원이라고 가정해면 3 by 4짜리 matrix가 생기게 된다. 여기에 이제 각각의 가중치를 곱해줌으로써 Q, K, V를 얻게 된다.

![](/img/attn7.png)

그렇게 구한 Q와 K를 곱해주게 되면 Attention Energy가 만들어지게 되는데, 이는 각각의 단어가 서로에게 어떤한 연관성을 가지는지 나타내게 된다.

마지막으로 Attention은 softmax를 취해준 확률과 value를 곱하게 됨으로써 최종적인 값이 나오게 된다.

추가로 mask matrix에 음수의 무한값을 넣어준 뒤 attention energy인 $QK^T$에 씌워줌으로써 softmax 함수의 출력이 0에 가까워지도록 할 수 있다.

즉, mask matrix를 사용해서 특정 단어를 무시할 수 있는 것이다.

![](/img/attn8.png)

### Applications of Attention in our Model

- **Encoder-Decoder Attention** : 디코더에서 인코더의 출력을 참고할 때 사용된다. query는 디코더의 입력, key와 value는 인코더의 출력에서 가져온다. 따라서 디코더가 query와 인코더의 정보인 key, value를 비교하여 적절한 정보를 추출한다.
- **Encoder Self-Attention** : Encoder에서 사용되는 Self-Attention이다. 각 단어가 입력 시퀀스 내의 모든 단어를 참고할 수 있다. Query, Key Value 모두를 인코더에서 가져온다.
- **Masked Decoder Self-Attention** : Decoder에서 사용되는 Self-Attention이다. Query, Key, Value가 같은 디코더 입력에서 생성되지만, 현재 토큰 이후의 단어를 보지 못하도록 마스킹한다. 미래 토큰에 해당하는 Attention score들은 -무한으로 설정하여 Softmax 이후 0이 되도록 한다.

### Positional Encoding

RNN을 사용하지 않으려면 위치 정보를 포함하고 있는 임베딩을 사용해야 한다. 따라서 transformer에서는 Positional Encoding을 사용한다.

이 논문에서는 간단하게 sin곡선의 상대적인 위치에 대한 정보를 임베딩과 동일한 차원으로 가져와 더해주는 것이다. 즉, d_model 차원과 동일한 sin그래프를 가져와서 더해주는 것.

![](/img/attn9.png)

![](/img/attn10.png)
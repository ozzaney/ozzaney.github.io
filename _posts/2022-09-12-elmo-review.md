---
layout: post
title:  "What is ELMo?"
categories: [ paper_review, AI, NLP ]
image: assets/images/elmo.jpg
---

# ELMo: Deep contextualized word representations 

## What is ELMo?

언어에서 맥락은 매우 중요합니다. 
그 이유는 맥락 속에서만 “의미”가 명확히 드러나기 때문입니다.
( 물론 맥락에는 상황적, 문화적, 언어적 맥락 다양한 종류가 있고 “의미”는 이 모든 맥락들에 의해 영향을 받습니다. 여기서 맥락은 언어적 맥락에 국한합니다.)
ELMo란 Embeddings from Language Models의 줄임말로, 요약하자면 문장 내에서의 언어적 “맥락"을 고려한 워드 임베딩을 가능하게 하는 모델입니다.
문장 내에서의 맥락을 고려한 임베딩, ELMo에 대해 더 자세히 알아봅시다!

## before ELMo

ELMo가 나오기 전에는 자연어처리 분야에서 Word2vec이나 Glove와 같은 pre-trained word embeddings를 많이 사용했습니다.
pretrained word embeddings를 이용해 NLP task를 수행하면 수많은 데이터로 이미 학습이 되어 있기에 문장에서 의미적, 구문적 정보를 캡쳐해내는 성능이 뛰어나고, 학습을 시킬 때도 시간이 얼마 걸리지 않습니다. 그 이유는 이미 학습이 되어 있기에 시작부터 거의 완벽한 상태의 임베딩 레이어를 가지기 때문입니다. 이 때문에 많은 SOTA 모델의 구조는 pre-trained 모델의 형태를 가지고 있습니다.

하지만 pre-trained word embeddings에도 단점이 존재합니다.
바로 각 단어는 하나의 임베딩만 가진다는 것입니다. 즉 한 단어가 vector space에서 한개의 점으로 밖에 표현이 안된다는 것인데요.
이것이 문제가 되는 이유는 하나의 철자를 가진 단어라도 여러가지 뜻을 가질 수 있기 때문입니다.
예를 들어 영어 단어 bank는 bank account에서는 "은행"의 의미를 가지고, river bank에서는 "둑"의 의미를 가집니다.
이렇게 서로 다른 뜻을 가졌지만 하나의 임베딩으로 표현된다면 nlp 문제를 해결하는데 있어 왜곡이 발생할 수 있기에 문제가 됩니다.

이러한 전통적인 방식의 단점을 극복하기 위해 subword information을 활용하는 방식 또는 각각의 의미마다 다른 임베딩을 활용하는 방법들이 개발 되어 왔습니다.  그러나 각각의 의미미다 다른 임베딩을 활용하게 되는 방법의 경우 문제가 있을 수 있는데요.
바로 사전에서의 하나하나의 의미를 구분짓는 경계가 모호하며, 사전마다 각각의 의미를 구분하는 방식이 다를 수 있기 때문입니다.
 이러한 상황에서 완전히 새로운 임베딩 방법론 ELMo가 등장하게 됩니다.

## our savior ELMo comes out!

ELMo는 문장(context)를 입력으로 받는 함수입니다.
그래서 문장(context)에 따라 같은 단어도 다른 임베딩으로 출력이 가능합니다.
예를 들어 문장(context)에서 “bank account”의 입력이 주어지면 bank에 대한 임베딩 벡터는 “은행”의 뜻에 가깝게 임베딩이 되고,
“River bank”가 입력으로 주어지면 bank에 대한 임베딩 벡터는 “둑”의 뜻에 가깝게 임베딩이 될 것입니다.

또 철자가 같지만 시제가 다른 경우에 ELMo가 이에 대해 인식을 할 수 있습니다.

(1) “I read a book now”
(2) “I read a book yesterday”

위 두 문장이 있을 때 첫번째 문장에서 ELMo는 now라는  맥락을 파악해 read를 “리드”로 읽을 수 있고,
두번째 문장에서는 yesterday를 통해 맥락을 파악해 read를 “레드”로 읽을 수 있습니다.


### ELMo is a function providing contextualized embedding

ELMo(input) = output

ELMo(bank account) = [1,3],[,2,4]
ELMo(river bank) = [5,1],[10,7]
 
서로 철자가 같은 bank임에도 [1,3], [10,7]로 다르게 임베딩되었음을 확인할 수 있습니다.
즉 ELMo는 다른 단어이지만 형태가 같은 동철이의어(또는 동음이의어)에 대해 각각 다른 representation을 부여할 수 있습니다.
이것이 가능한 이유는 무엇일까요?


ELMo의 이러한 특별한 성능을 가능하게 한 세 가지 포인트는 
**1. biLM(bidirectional Language Model)구조를 사용하고, 2. 해당 구조 내의 모든 internal state를 사용했으며, 3. Character embedding from CNN을 사용했다는 점** 입니다.

1. biLM(bidirectional Language Model) 구조

LM은 Language Model의 약자로, 자연어처리에서 뒤의 단어들로 앞의 단어를 예측하는 모델을 이르는 말입니다. 
(LM에는 기본적인 Neural Network LM, RNN LM등 다양한 LM이 존재합니다.)
biLM은 두개의 방향을 가진 순방향 LM과 역방향 LM을 합친 LM입니다. 
예를 들어 “나는 사과를 먹었다”라는 문장이 있을 때,
순방향 LM은 “나는”, “사과를”을 이용해 “먹었다”를 예측합니다.
반면 역방향 LM은 “사과를”, “먹었다”를 이용해 “나는”을 예측할 수 있습니다.

순방향 LM

<img width="445" alt="순방향" src="https://user-images.githubusercontent.com/85322951/189663790-12c9c7ee-f893-4251-91e1-39358f0d2616.png">


역방향 LM

<img width="444" alt="역방향" src="https://user-images.githubusercontent.com/85322951/189663942-0e69d2c8-2f37-417e-979b-0a6395039abd.png">


forward and backward direction의 log likelihood를 합한 값을 maximize 합니다.

<img width="452" alt="로그라이클리" src="https://user-images.githubusercontent.com/85322951/189664088-ed154014-e9cc-42f3-a8f7-463b534c52b4.png">

 Deep contextualized word representations에서는 거대한 corpus로 사전 학습된  biLM을 사용합니다.

**주의**
양방향 RNN과  Deep contextualized word representations에서의 biLM은 다릅니다.
양방향 RNN은 순방향 RNN과 역방향 RNN의 은닉상태를 연결(concatenate)하여 다음층의 입력으로 사용하는 반면
biLM은 순방향 언어모델과 역방향 언어보델이라는 두 개의 언어모델을 별개의 모델로 보고 학습을 합니다.

2. 모든 internal state의 사용
internal state을 모두 사용한다는 말의 의미는 모든 레이어의 출력값을 활용해 임베딩을 만든다는 뜻입니다.
이전의 SOTA모델들이 최상위 레이어만 사용한 것과 비교해  매우 대조적 입니다.
Intrinsic evaluation을 해본 결과, 상위 레이어(출력층에 가까운 레이어)일수록 문맥에 의한 벡터를 출력할 수 있었습니다.
(예를 들어 supervised word sense disambiguation 문제를 해결하는데 좋은 performanace를 보였습니다. 
https://bab2min.tistory.com/576)
또한 하위 레이어(입력층에 가까운 레이어)일수록 문법에 가까운 벡터를 출력한다고 합니다. (Pos 태깅과 같은 작업에 사용될 수 있었습니다.)
이러한 결과는 MT encoders에서와 유사하다고 합니다. 

3. Character embedding from CNN 사용

[CNN for NLP](https://ozzaney.github.io/Character-embedding-from-CNN/)에서 자세히 다룹니다.

Character convolution을 통한 subword unit을 사용해 동철이의어(또는 동음이의어)를 잘 처리할 수 있도록 했습니다.
최초 레이어의 임베딩은 문맥의 영향을 받지 않고(왜?), ELMo를 pretrained word embedding과 비교하기 위해서 Glove나 word2vec을 워드 임베딩으로 사용하지 않았습니다. 
첫 레이어가 문맥으로부터 영향을 받지 않도록 설계되었지만 이후의 레이어는 문맥에 영향을 받도록 설계하였습니다.(어떻게?)
또 첫 lstm 출력이 char 임베딩과 residual conncetion을 가지고 있어서 1. 상위 레이어들이 하위 레이어들의 특징을 잃지 않고 활용하도록 돕고 2.gradient vanishing 현상을 극복하도록 도움.

### ELMo representation

ELMo는 이러한 특별한 방식으로 predefined sense class로 명시적으로 학습시키는 과정 없이도 단어의 여러 의미를 임베딩 representation에 반영할 수 있습니다.

ELMo representation은 biLM에서 등장하는 internal state의 값들을 특별하게 합친 것으로, 다음과 같은 식을 가집니다.
<img width="342" alt="elmo_representation" src="https://user-images.githubusercontent.com/85322951/189664147-f7a17c5c-c3e5-41de-8885-d8167806c49c.png">

위 식에서 s는 soft max에 정규화된 가중치이고, gamma는 전체 ELMo 벡터의 크기를 조절하는 역할을 합니다.
아래의 그림은 ELMo가 임베딩 벡터를 만드는 과정을 시각화한 것 입니다.

<img width="546" alt="forward_and_backward" src="https://user-images.githubusercontent.com/85322951/189664417-906aa5f3-e218-4e2a-9f01-188d56648de4.png">

<img width="535" alt="elmo_step" src="https://user-images.githubusercontent.com/85322951/189664313-9c807f17-eabc-42ec-bb84-5bccc50d0f84.png">

출처: https://wikidocs.net/33930


### 모델의 작동 과정

위에서 완성된 ELMo representation을 활용해 다양한 NLP 테스크를 수행하고자 한다면 어떻게 할까요?

<img width="308" alt="elmo_task" src="https://user-images.githubusercontent.com/85322951/189664310-0b516f69-afb2-428b-b0df-4196b8b80273.png">
출처: https://wikidocs.net/33930

위 그림은 ELMo representation을 활용해 기존의 방법론으로 만든 embedding과 concatenate해 NLP task에 적용하는 구조입니다. 이때 pre-trained된 biLM에 대한 가중치를 고정한 채 위에서 사용한 s, gamma는 훈련과정에서 학습됩니다.
이런식으로 pre-trained된 ELMo 모델을 NLP 모델 앞에 연결하여 활용하게 됩니다. (그러나 연구에 따르면 task에 따라 output layer에 elmo표현을 넣거나, input과 output layer 모두에 넣는 방식이 성능을 향상시키는 경우도 있다고 합니다.)

### ELMo의 성능

<img width="664" alt="elmo_performance" src="https://user-images.githubusercontent.com/85322951/189664156-951e983d-3520-4da0-8439-357b2122a75b.png">

단순히 ELMo representaition을 추가하는 것만으로 (textual entailment, question answering and sentiment analysis를 포함한) 6가지의 nlp의 어려운 task들에 대한 sota모델들의 error를 상대적으로 6-20%까지 감소시켰습니다.
또한 ELMo와 CoVe의 분석에 따르면 심층 표현(모든 internal state를 사용하는 것)은 LSTM의 최상위 계층에서 파생된 표현보다 성능이 우수합니다.
또한 ELMo를 통해 sample efficiency를 달성할 수 있습니다. 예를 들어 ELMo를 추가하지 않았을 때 SRL 모델은 486 에폭 이후 최고 development F1에 도달하지만 ELMo를 추가한 뒤 base line의 최대치를 10 에폭만에 도달합니다. 또한 ELMo를 사용할 경우 더 적은 training sets로도 좋은 결과를 낼 수 있습니다.

<img width="307" alt="elmo_performance2" src="https://user-images.githubusercontent.com/85322951/189664163-6f5e93f9-f8f8-4e03-a5ad-7086a0a971d0.png">

위 그림은 baseline과 ELMo를 추가한 모델의 training set의 크기에 따른 비교입니다. 이를 통해 ELMo를 추가하는 것이 더 적은 데이터로도 더 좋은 결과를 낼 수 있음을 확인할 수 있습니다. 

### 참고문헌

#### 논문링크
https://arxiv.org/pdf/1802.05365.pdf
#### 참고링크 (유튜브, 블로그, 사이트)
https://wikidocs.net/33930
https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/20/ELMo-Deep-contextualized-word-representations/
https://velog.io/@skm0626/Deep-contextualized-word-representationsELMo-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
https://www.youtube.com/watch?v=YZerhaFMPTw

### 더 공부해보면 좋을 keyword

1. Residual connection
2. Intrinsic/extrinsic evaluation
3. biLM
4. Perplexity

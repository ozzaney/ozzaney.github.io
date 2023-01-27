---
layout: post
title:  "CNN for NLP"
categories: [ keyword, AI, NLP ]
image: assets/images/cnn_fency.png
use_math: true
---
# how to use CNN in NLP? 

## content

- CNN(합성곱신경망)의 동작과정
- 자연어처리를 위한 1D 합성곱 신경망
- 합성곱 신경망을 이용한 분류문제 해결
- 합성곱 신경망을 통한 문자임베딩

해당 포스트는 [자연어처리 입문 NLP를 위한 합성곱 신경망](https://wikidocs.net/64065)을 참고하였습니다.

## Summary

fully connected layer의 불필요한 연산량, 오버피팅, 공간정보 유실의 문제를 해결하기 위해 등장한 CNN.
CNN은 hidden layer로 convolution layer가 있으며, 이 layer는 "공간정보(spatial structure)를 반영한 패턴을 detection"하는 역할을 수행한다.

## CNN(합성곱신경망)의 동작과정

CNN은 주로 이미지 데이터를 처리하는데 사용되는 신경망입니다.
다층 NN구조가 기존에 있는데도 불구하고 이미지 데이터에는 주로 CNN을 사용하는 이유는 무엇일까요? 
그 이유는 이미지 데이터의 공간정보(spatial structure)를 반영하기 위해서 입니다.
기존의 다층 NN구조를 사용할 경우 이미지 데이터를 1차원 벡터로 변환하게 되면서 각 픽셀 값들이 어떤 픽셀들과 가까이에 있었는지 등의 정보가 유실됩니다. 그러나 CNN을 사용하면 이러한 공간정보의 유실없이 이미지 데이터를 처리할 수 있습니다.
또한 다층 NN구조의 fully connected layer는 불필요한 연산량, 오버피팅의 문제가 나타납니다.
이제 CNN의 주요 용어들을 알아봅시다.

#### 1. 채널(channel)

우리가 다루는 모든 NN은 tensor(숫자)를 input으로 받아들입니다.(기계는 숫자만 이해할 수 있습니다.)
그 중 이미지는 (높이=세로방향 픽셀 수, 너비 = 가로방향 픽셀 수 , 채널 = RGB색성분 )이라는 3차원의 텐서입니다.
3차원 텐서는 직관적으로 상자모양을 떠올리면 이해가 쉬울 것입니다.
흑백이미지는 채널의 수가 1이고 컬러이미지는 채널 수가 3입니다.
또 각 픽셀은 0~255 사이의 값을 가집니다.
* 채널을 때로는 깊이(depth)라고 부를 때도 있습니다.

(예시) 흑백이미지 1장  = (28, 28, 1), 컬러이미지 1장  = (28, 28, 3)

#### 2. kernel = 특정 패턴에 대한 detector 

CNN에서는 이미지의 특징을 추출하기 위해 kernel(또는 filter)를 이용합니다.
여러개의 패턴을 감지하고 싶다면 그만큼 많은 개수의 kernel을 이용하면 됩니다.
kernel은 조그만 행렬조각이라고 이해하면 되는데, input으로 들어온 이미지에 이 kernel이 전체를 훑고 지나가며 해당 pattern을 가지고 있는지 확인하는 작업을 하게 됩니다.
이때 kernel의 값들은 난수로 초기화가 되어 있습니다. kernel은 학습을 통해 update되며 이미지들의 패턴을 학습하게 됩니다.
이 부분을 인간이 이미지를 처리하는 작업에 비유해보자면, 갓 태어나 아직 많은 정보를 보지 못한 아기(난수로 초기화된 kernel)는 특정한 사물이 가진 이미지 패턴을 모르나, 다양한 정보(데이터)를 접하면서 사물들의 패턴(kernel 또는 filter)를 알게 되는 것(학습, update)과 비슷하다고 생각하면 됩니다. 
예를 들어 강아지는 귀를 가지고 있고 인간은 눈코입이 있다 등과 같은 패턴이 있을 것입니다.

#### 3. 합성곱 연산

위에서 설명한 커널이 이미지 전체를 훑고 지나가면서 합성곱 연산을 하게 되고, 이 결과로 나온 값을 feature map(특성맵)이라고 합니다.
이때 한번 이동할 때의 크기를 stride라고 합니다.

#### 4. 패딩

feature map은 kernel이 지나가고 계산된 합성곱의 결과이므로 차원이 input 이미지보다 작습니다.
따라서 convolutional layer를 통과할수록 최종적인 특성맵의 크기가 작아집니다. 이를 방지하기 위해 패딩을 입듯이 feature map의 가장자리에 지정된 개수의 값들로 채우는 padding을 진행하게 되며 보통 이 값을 0으로 하는 제로패딩을 주로 시행합니다.

#### 5. 합성곱 신경망의 가중치와 편향

CNN에서 가중치는 이미 언급한 바와 같이 kernel(filter)들의 원소입니다.

<img width="662" alt="합성공신경망" src="https://user-images.githubusercontent.com/85322951/190055686-319fceb0-fc50-4f23-b9aa-396ca65324f3.png">

출처: https://wikidocs.net/64066

위 그림은 합성곱 연산은 인공신경망 모형으로 나타낸 모습입니다.
특성맵을 얻기 위해서는 위 그림의 w1,w2,w3,w4만을 가지고 이미지 전체를 훓으며 값을 계산하게 됩니다.
또 각 합성곱의 연산마다 이미지의 모든 픽셀을 사용하지 않고 커널과 맵핑되는 픽셀만을 입력으로 사용합니다.
이를 통해 fc layer보다 훨씬 적은 가중치를 사용하면서도 spatial structure를 보존합니다.
convolutional layer를 통과한 후의 차원은 다음과 같습니다.

$floor(\frac{Input-Kernel}{Stride}+1)$

input이 1개의 channel(depth)만 가지는 경우 즉 흑백 이미지인 경우라면 kernel(filter)는 1개의 채널을 가지겠지만
만약 input이 3개의 channel을 가진 경우 즉 컬러 이미지라면 kernel또한 3개의 채널을 가져야 합니다.
**즉 커널의 채널수와 입력의 채널수가 같아야 합니다.**
즉 입력이 면이라면 kernel도 면이고, 입력이 상자모양이라면 kernel도 상자모양입니다.

<img width="447" alt="3차원텐서합성곱" src="https://user-images.githubusercontent.com/85322951/190062170-c99d2b69-6312-43db-9af4-e5814a0c2e7c.png">

위 그림은 3차원의 텐서의 경우 합성곱의 과정을 보여주고 있습니다.
3차원 텐서의 합성곱일 경우 커널도 3차원이므로 3개의 feature map이 도출되고 이 3개의 합으로 하나의 feature map을 구하게 됩니다.
합성곱 연산의 결과로 얻은 특성 맵의 채널 차원은 RGB 채널 등과 같은 컬러의 의미를 담고 있지는 않습니다. => 만약 task가 컬러정보가 필요하다면 어떤 정보를 기반으로 컬러에 대한 정보를 전달하는지가 궁금하네요!

따라서 채널의 수가 1이든 3이든 커널을 통과한 결과 채널이 1인 feature map을 얻게 됩니다.

<img width="375" alt="하나" src="https://user-images.githubusercontent.com/85322951/190064012-f530dbd2-53b9-41f8-a4fc-fbaba678f088.png">

그러나 커널의 개수가 여러 개인 경우 채널이 1인 feature map들이 커널의 개수만큼 생기게 되고 이것이 쌓여서 feauture map은 커널의 개수를 channel의 크기로 가지게 됩니다.

<img width="347" alt="쌓인거" src="https://user-images.githubusercontent.com/85322951/190064045-fc5e83a8-bf33-47b8-ad48-bee293c27b9e.png">

따라서 하나의 가중치 매개변수의 총 수는 "kernel의 width * kernel의 height * kernel의 depth * kernel의 개수" 입니다.

6. 풀링

일반적으로 합성곱층(합성곱 + activation ftn) 다음에는 풀링층을 추가하며 max pooling 또는 average pooling이 사용됩니다.
합성곱 연산과 마찬가지로 풀링 이후 특성 맵의 크기가 줄어들며  stride도 조정가능합니다.
다른 점은 학습해야할 가중치는 없고 연산 후에 채널의 수가 변하지 않습니다. 

## 자연어처리를 위한 1D 합성곱 신경망

하나의 문장이 있다고 합시다.
이 문장이 토큰화, 패딩, 임베딩 층을 지나 행렬이 된다면 n(문장의 길이)* k(임베딩벡터의 차원수) 행렬이 될 것입니다.

만약 이 행렬이 합성곱 신경망에 입력으로 주어진다면 어떻게 될까요?

이럴 경우 앞서 이야기한 kernel의 너비는 k로 고정됩니다.
그래서 문장 행렬을 kernel이 훓고 지나가려고 할 때 가로방향으로는 더이상 움직일 수 없으므로 세로방향으로만 움직이게 됩니다.
따라서 2방향이 아니라 1방향으로만 움직이기에 1D CNN라고 하는 것입니다.

<img width="708" alt="1dcnn" src="https://user-images.githubusercontent.com/85322951/190066962-3bfb929f-5cfa-4579-b990-adc1659f1819.png">

위 그림은 너비가 k이고 높이는 2인 kernel이 문장행렬을 지나가는 모습입니다.
높이가 2라는 것은 참고하는 단어 묶음의 수가 2, 즉 bigram을 뜻합니다.
kernel의 높이를 수록 더 많은 수의 단어를 함께 묶어 참고하게 되는 것이죠.

<img width="618" alt="maxpool" src="https://user-images.githubusercontent.com/85322951/190067871-22132394-87b7-411a-8670-a4abff0a5aa3.png">

이후 앞서 도출된 feature map에서 가장 큰 수를 추출해내는 max pooling 과정 (average pooling을 해도 됩니다.)을 거치게 됩니다.

이제 이러한 1D CNN을 이용해 신경망을 설계한 것을 살펴봅시다.

<img width="571" alt="신경망설계" src="https://user-images.githubusercontent.com/85322951/190068019-2c3fd2f8-bbd0-4744-b113-83c298b2d1e3.png">

<img width="571" alt="신경망설계" src="https://user-images.githubusercontent.com/85322951/190068019-2c3fd2f8-bbd0-4744-b113-83c298b2d1e3.png">

크기가 4,3,2로 다양한 kernel을 이용해 합성곱을 통한 feature map을 구하고 max pool을 통해 하나의 값을 구하는 과정을 kernel의 총 개수인 6번하고 이를 concatenate한 뒤 이를 뉴런이 2개인 층에 fc layer로 연결한 모델임을 알 수 있습니다.



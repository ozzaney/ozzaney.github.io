---
layout: post
title:  "Steal the author's style"
categories: [ keyword, AI, NLP ]
image: assets/images/authors.png
use_math: true
---
# Adapting Language Models for Non-Parallel Author-Stylized Rewriting

[논문링크](https://arxiv.org/pdf/1909.09962.pdf)

해당 게시글은 제가 본 논문을 읽고 이해한 바를 정리한 내용입니다.  

틀린 내용이 있거나 보충이 필요한 내용, 또는 질문이 있다면 댓글로 남겨주시면 감사하겠습니다 :)  

## Abstract

최근 [transformer](https://wikidocs.net/31379) 기반의 다양한 language modeling이 발전하고 있고 그 중 **스타일링된 텍스트**를 만드는 것에 대해 많은 관심이 쏠리고 있습니다. **스타일링된 텍스트**란 뜻은 동일하지만 스타일을 다르게 만든 텍스트를 이르는 말로, 말투나 부정/긍정, formal/informal 등의 여러가지 범주를 스타일로 정의해볼 수 있습니다.
이러한 배경에서 본 논문에서는 **[language model](https://wikidocs.net/21668)을 활용해 텍스트를 target 작가의 스타일로 다시 쓰는 모델**을 소개합니다. 본 논문에서는 스타일을 **작가의 문체**로 정의했습니다.
모델에 평범한 일상의 문장을 input으로 넣으면 output으로 셰익스피어라는 target 작가의 문체로 바뀐 문장을 output으로 만들어내는 모델을 만든 것입니다.
본 모델은 사전학습 모델을 [fine-tuning](https://eehoeskrap.tistory.com/186)해 작가의 문체로 변형된 텍스트를 만들어내는 것이 목적인데요,
이를 위해 [Denoising AutoEncoder(DAE)](https://deepinsight.tistory.com/126) loss를 사용해 [cascaded](https://daebaq27.tistory.com/79) encoder-decoder 구조에서 학습하였고 데이터는 각 작가들의 책에서 추출한 corpus로 fine-tuning 하였습니다. DAE loss를 이용해 각 작가의 corpus를 학습함으로써 parallel data없이도 모델이 텍스트의 뉘앙스를 학습할 수 있었습니다.

* 여기서 parallel data는 뜻은 같지만 스타일이 다른 두 문장의 쌍이라고 보면 되는데, (일반문장, 셰익스피어 어투의 문장)을 예로 들 수 있습니다. 이러한 parallel data를 이용하면 supervised learning(지도학습)이 가능하다는 장점이 있지만 현실적으로 parallel data는 없거나 매우 부족한 상황이기에 non-parallel data를 이용한 방법론이 주목을 받고 있습니다.

모델평가에 대해서는 target 작가에 대해 생성된 텍스트의 stylistic alignment(스타일이 텍스트에 잘 입혀진 정도)를 어휘, 통사, 표면적 측면에서 정량화하기 위해 언어적 frame work를 활용했습니다. 정량적, 정성적 평가를 통해 해당 접근방식이 SOTA보다 원본을 잘 보존하면서도 target style을 더 잘 보여준다는 것을 보여줍니다. 

## Introduction

최근 들어 NLP 분야에서 stylized text generation과 style transfer에 관심이 집중되고 있습니다.
stylized text generation과 style transfer 두 개의 테스크들 모두 input text를 target style에 맞게 변형하여 새로운 text를 만들어내는 것이 목표인데, 이에 대한 작업들은 sentiment와 formality, 또는 두개의 결합에 대해 다양한 level 별로 말투를 바꾸는 것이었습니다.

(예시)  
내용은 동일하나 formality만 다른 두 문장
- 형식적이면서 기분좋은 말투: 안녕하십니까 만나게 되어 반갑습니다  
- 비형식적이며 기분 좋은 말투: ㅎㅇㅎㅇ 반갑쓰  

이러한 연구들은 주로 formality와 sentiment에 대해 정반대의 level을 지닌 parallel data를 이용해 이루어졌습니다.
그러나 심리언어학적 측면의 개념이 아닌, author의 글에 드러난 언어적 선택으로서의 style에 맞게 텍스트를 생성하는 것에 대해서는 연구가 부족했습니다.
Jhamtani et al. (2017)는 “Shakespearized text”를 만들려고 하긴 했지만 parallel data를 사용했다는 점에 한계가 있습니다.
parallel data를 이용한다면 author별로 parallel 데이터를 구축해야하는 한계가 있으므로 이에 의존하지 않는 방식이 필요했습니다.
이에 따라 본논문에서는 parallel data에 의존하지 않는 author-stylized rewriting 방법을 고안했습니다.

<img width="503" alt="스크린샷 2022-09-26 오후 4 23 02" src="https://user-images.githubusercontent.com/85322951/192216941-66b8c760-8f3b-40fd-910f-4c056be2259d.png">{: width="700" height="400"}

이 모델에서 주목할 점은 **SOTA 언어모델의 일반화능력을 사용하기 위해 사전학습된 언어모델을 fine tuning해 parallel data없이 target author의 스타일 특성에 맞는 text rewriting**에 성공했다는 점 입니다.
구체적인 학습방법은 다음과 같습니다.

1. author courpus와 wikipedia data 두가지 데이터를 활용해 MLM(Masked Language Modeling)방식으로 pre-train합니다. 
2. encoder와 decoder 둘 다에 앞서 pre-train한 언어모델의 가중치로 initialization합니다.

3. 해당 모델을 target author의 corpus에 대해 denoising autoencoder loss로 fine-tuning합니다.

본 모델은 특정 스타일을 지니거나 아무런 스타일이 없는 text를 input으로 받아 target author's style을 가진 text로 변환합니다.
그러나 author의 스타일은 어휘, 구문, 의미 3가지 언어적 특성에 의해 결정되기에 모델 평가하기는 쉽지 않습니다.
따라서 본 논문은 stylistic alignment의 정도를 수량화하기 위해 새로운 방법을 제안합니다.
뒤에서 더 자세히 알아봅시다.

본 논문의 key contribution은 다음의 3가지 입니다.

- parallel data 없이 SOTA모델을 fine tuning해 author-stylized text를 만드는 방법을 제시하고 평가했다.
- 스타일의 어휘적 측면과 통사적 측면의 alignment을 설명하는 평가 프레임워크를 제안한다. 기존 평가 기법과 달리, 해당 평가방법은 언어적으로 쉽게 해석할 수 있다.
- author-stylized text generation에 대해 baseline보다 질적, 양적으로 발전했다.

## Related Work

### Stylized Text Geneartion(STG)
최근의 연구에서 언어심리학적 측면에서 sentiment나 formality의 다양한 level을 구현하는 방식의 STG가 등장했습니다.
본 논문에서는 non-parallel data를 이용해 SOTA language model을 pre-training하기로 했습니다.
language model을 사용한 가장 큰 이유는 stylistic rewriting이 단순한 text generation에 기반을 두기 때문입니다.
본 논문의 목적과 비슷하게 author의 스타일로 text를 adapt하기 위한 시도가 있어왔습니다.

Tikhonov and Yamshchikov (2018)는 stylistic variables들의 임베딩을 concatenate하고 conditioning을 사용해 
style을 end-to-end로 학습하여 author-stylized poetry를 생성하는 모델을 만들었지만 본논문과 똑같이 rewriting을 시도하지는 않았습니다.
Jhamtani et al. (2017)는 parallel data를 사용해 “Shakespearized” version의 현대영어를 만들어내려고 했습니다.
그러나 본 논문은 parallel data가 아닌 non-parallel data를 활용한다는 점에서 차이가 있습니다.
즉 본논문의 모델은 오직 target author의 corpus만을 필요로 합니다.
뒤에서 나오겠지만 본논문이 제시하는 모델은 non-parallel data를 사용했음에도 content preservation과 style transmission metric에서  Jhamtani et al. (2017)에 비견될만한 성능을 보입니다.

### Language models

Generative pre-training of sentence encoders (Radford et al. 2018; Devlin et al. 2019; Howard and Ruder 2018)는 nlp task에서 큰 발전을 가져왔습니다.
해당 접근방식은 transformer language model을 커다란 unsupervised 코퍼스로 학습시키고 분류문제 또는 추론기반의 NLU 테스크에 fine-tuning해 접목하는 방법입니다. Lample and Conneau(2019)는 이를 토대로 cross-lingual language models을 만들었습니다.
이에 착안하여 본 논문에서는 generative pre-training을 author-stylized rewriting 테스크에 적용합니다.
 GPT-2 (Radfordet al. 2019)에서는 크고 다양한 코퍼스로 pre-trained 되었고 NL generation을 비롯한 다양한 도메인과 데이터셋에서 잘 작동합니다.
 unsupervised pre-training는 이전의 말로 다음말을 예측하는 모델(causal language modeling, CLM)로 다음단어가 나올 확률을 예측합니다. 다음은 text generation에 대한 CLM의 확률식입니다.  
 
$P(y_t|y_{1:t-1}, x)$  
위와 같이 CLM중 에서도 text generation 테스크에서만 특징적으로 input prompt x를 취하여 input의 context를 반영하도록 합니다.
GPT-2에서는 author-specific corpus로 fine-tuned된 경우 target author의 style에 대해 상당히 stylistic alignment가 잘 됩니다. 그러나 stylistic rewriting(우리가 하려는 테스크)와 stylized text generation의 본질적인 차이로 인해 GPT-2는 content를 보존하는데에 좋은 성능을 보이지는 못했습니다.
stylistic rewriting에서는 stylized generation에서 input text에서의 정보(content)를 유지하지만 GPT-2에 의한 스타일 생성은 입력 프롬프트와 관련된 콘텐츠를 **생성**하므로 미세 조정된 GPT-2는 stylistic rewriting을 처리할 수 없습니다.
즉 GPT2를 사용하면 내용을 유지하는 것이 아니라 관련된 내용을 새롭게 생성하기에, 내용을 온전히 유지하면서 타겟 작가의 스타일을 닮은 글을 생성하는 rewriting task에는 적합하지 않습니다.

해당 모델에서는 MLM에 대해 거대 corpus로 학습시킨 뒤, encoder-decoder setup에서 DAE loss를 사용해 author specific corpus로 fine-tuning했습니다.

### Evaluating Stylized Generation

Fu et al. (2018)에서는 style transfer models의 성능을 1. content preservation, 2. transfer strength이라는 두가지의 축으로 평가했습니다.
content preservation의 성능은 input과 만들어진 text간에 얼마나 유사한지를 나타내는 [BLEU](https://wikidocs.net/31695)로 평가하고, transfer strength는 target style에 대해 만들어진 텍스트가 얼마나 align하는가로 평가합니다. unsupervised learning이기에 generated된 text의 style이 잘 생성되었는지를 판단할 정답 데이터(groud-truth)가 없고, 그래서 transfer strength를 평가하기가 쉽지 않습니다. 
따라서 본 논문은 author-stylized text에서 스타일의 여러 어휘 및 통사적 측면의 alignment를 정량화하는 평가법을 제안합니다.  

## Propsed Approach : StyleLM

이제 본격적으로 모델에 대해 설명해보도록 하겠습니다.
해당 모델은 두가지 중요한 양상을 가집니다.
1. transformer based model을 커다란 데이터셋으로 pre training하고 author specific corpus로 DAE를 사용해 fine-tuning한다.
2. 해당 모델은 self-supervised 방식으로 학습되며 parallel data에 의존하지 않는다.

<img width="927" alt="스크린샷 2022-09-27 오후 1 34 56" src="https://user-images.githubusercontent.com/85322951/192433048-8dbc1cb7-13b3-4c23-bef0-b7d88297c8b2.png">{: width="700" height="400"}

위 그림은 해당 모델이 프레임워크를 표현한 것입니다. 

먼저 방대한 corpus(Author들의 전체 corpus, 위키피디아 corpus)로 transformer based language model을 MLM 목적함수를 가지고 훈련합니다.
MLM은 masked된 단어가 앞뒤 맥락(bidirectional context)을 통해 유추되도록 합니다.
x가 주어진 문장일 떄 $x_{\u}$는 x에서 position u가 masked된 상태를 나타낸다고 합시다.
(u위치의 token이 masked될 경우 해당 token이 [MASK]로 대체됩니다. 이를 통해 masked된 문장의 길이가 전과 달라지지 않도록 합니다.)
MLM 목적함수는 $x_{\u}$를 input으로 받아들였을 때 $x_u$를 예측하도록 language model을 훈련시킵니다. 

<img width="370" alt="스크린샷 2022-09-27 오후 1 57 31" src="https://user-images.githubusercontent.com/85322951/192439148-f6c52ec6-6381-4e14-aed7-d93d0ef1e1d0.png">

이때 X는 전체 training corpus입니다.
MLM을 pre-training을 하기 위해  Devlin et al. (2019)의 방법을 따라, 각각의 input sentence마다 전체 중 15%의 token들을 랜덤하게 mask합니다. 그리고 그 중 80%를 [MASK]로 대체하고 10%는 radom token으로 대체하며, 나머지 10%는 바꾸지 않은 채로 둡니다.

이제 author-stylized rewriting을 가능하게 하기 위해, 앞서 만든 pre-trained laguage model 두 개를 직렬으로 이어붙여 encoder-decoder 구조를 만듭니다. 
즉 encoder와 decoder의 학습가능한 파라미터들이 pre-trained LM으로 초기화되도록 합니다.

정리하자면,
1. author courpus와 wikipedia data 두가지 데이터를 활용해 MLM(Masked Language Modeling)방식으로 pre-train합니다. 
2. encoder와 decoder 둘 다에 앞서 pre-train한 언어모델의 가중치로 initialization합니다. 
3. 해당 모델을 target author의 corpus에 대해 denoising autoencoder loss로 fine-tuning합니다. 즉 원본문장에 noise를 더한 것이 input이 되고 정답 label을 원본문장으로 주어 학습하는 방식을 택합니다.
구체적으로 설명을 해보자면, 인코더의 input은 noise가 섞인 embedding 문장이고, 인코더의 output은 denoised된 embedding일 것입니다. 
인코더의 output을 디코더로 넘겨줄 때 모델의 디코더도 인코더의 구조를 가지므로 인코더가 받을 수 있는 input을 전달해야합니다.
따라서 인코더의 해당 output에 softmax를 취한 뒤 argmax값으로 선정된 one-hot vector를 디코더의 output으로 넣어주면, denoising된 문장에 대한 임베딩을 만들어내어 최종 output문장을 도출할 것입니다. 
fine-tuning을 할 때 모델이 학습하는 데이터는 author-specific 문장이기 때문에 모델이 원본문장에 노이즈가 더해진 것에 원본문장을 복원하도록 학습하는 과정에서 문장의 의미는 보존하면서 문장의 스타일만 바뀐 모델을 만들어낼 수 있습니다. 

**Transformer 기반 언어 모델의 구조는 attention 메커니즘이 Transformer 설계에 내재되어 있기 때문에 인코더의 출력과 디코더의 입력을 명시적으로 align하지 않고도 pre-trained된 LM의 두 인스턴스를 직렬로 연결할 수 있습니다.**

이제 연결된 encoder-decoder를 다음의 DAE-loss를 이용해 fine-tuning합니다.

<img width="371" alt="스크린샷 2022-09-27 오후 2 14 47" src="https://user-images.githubusercontent.com/85322951/192439238-644b3aa2-b529-47e3-9245-434a45ea391b.png">

이때 C(x)는 input 문장 x의 noisy한 버전이고 S는 target author corpus의 문장들입니다.
C(x)를 만들기 위해 x안의 모든 각각의 단어들에 대해서 $P_{drop}$으로 버리고, $P_{blank}$으로 [BLANK]로 대체합니다. 이렇게 하는 이유는 모델에게 일부러 알아보기 어려운 문장을 알려줌으로써 noise에 robust한 model을 만들기 위해서입니다. 

![denoising ](https://user-images.githubusercontent.com/85322951/192441299-a8b9ba23-4397-4ca2-955a-751c15fb39e9.png)

pre-trained LM를 각각 두 개로 복사해 각각 enocdoer와 decoder로 삼아 두 개를 연결합니다.
이는 autoencoder 구조로 encoder-decoder 구조의 일종입니다.
이제 해당 구조에 input으로 텍스트의 noisy version을 넣어주면 enocder는 masked words를 유추하여 만들어냅니다.
이렇게 encoder가 만든 output은 decoder의 input으로 들어가게 되고 decoder에서는 noisy input text의 clear한 버전을 reconstruct하게 됩니다.
즉 모델의 디코더가 인코더를 통과한 noisy text를 rewrite할 때 target author의 스타일을 가진 문장을 생성하게 됩니다.

### Implementation detail

<img width="927" alt="스크린샷 2022-09-27 오후 1 34 56" src="https://user-images.githubusercontent.com/85322951/192433048-8dbc1cb7-13b3-4c23-bef0-b7d88297c8b2.png">{: width="700" height="400"}

본논문에서는 MLM을 pre-training할 때 12-layer의 tranformer encoder(Vaswani et al. 2017)를 사용하였고, 해당 transformer encoder에서 GLEU 활성화함수(Hendrycks and Gimpel 2017)를 이용, 하이퍼파리미터로는 hidden unit는 512, 16 heads, dropout 비율은 0.1을 채택했고 positional embedding을 학습했습니다. 또 Adam optimizer를 사용했고 learning rate은 0.0001이었습니다.
학습시 input으로 256개 토큰의 streams를 이용했고 mini-batch size는 32를 채택했습니다.
학습은 validation set에 대해 LM의 [perplexity](https://wikidocs.net/21697)가 더이상 줄어들지 않을 때까지 학습하였습니다.
target author에 대해 fine-tune할 때 사용하는 encoder와 decoder 모두에서 똑같은 pre-trained MLM transformer를 초기 값으로 사용합니다. 이때 하이퍼파라미터는 pre-training을 할 때와 동일한 값을 사용합니다.(Lample and Conneau (2019)를 참고한 방법론)
$P_{drop}$, $P_{blank}$는  둘 다 0.1로 정했고 모델이 수렴할 때까지 fine-tune을 했습니다.
또한 거대한 corpus를 다루기 위해 combined training dataset에 Byte Pair Encoding [BPE](https://wikidocs.net/22592)를 적용했고 80k BPE codes를 훈련시켰습니다.

## Evaluation Framework

### Dataset

Gutenberg corpus내에서 142명의 작가가 작성한 2857개의 책을 사용했습니다. 
한번도 본 적이 없는 데이터에 대한 예측을 하는 zero shot learning을 위해서 mark twain의 데이터를 훈련데이터에 포함시키지 않고 따로 떼어습니다. 훈련데이터를 다양화하기 위해 위키피디아 내용도 포함시켰습니다. 결과적으로 4.6M(4600000) passages들이 훈련데이터로 사용되었습니다. validation과 test를 위해서는 각 5000개씩의 passages를 떼어두었습니다.
fine tuning과정에서는 각 작가별로 독립적으로 모델을 만드므로 10명의 작가를 선정하여 튜닝을 진행했습니다.
inference과정에서는 훈련데이터에서 모델이 참고하지 않았던 데이터셋을 가져오기 위해 따로 떼어뒀던 mark twain의 데이터(문학적)와 리뷰데이터(일상적), ai 관련 위키피디아(모델에게 완전히 생소한 내용) 내용을 가져왔습니다. 

모델의 성능을 비교하기 위해 아래의 4개의 baseline과 비교하였습니다.

1. vanilla GPT-2 based generation :  pretrained GPT-2 model으로 문장생성
2. Author fine-tuned GPT-2 : cross-entropy loss로 각각 작가의 corpus로 fine tuning된 모델로 문장생성
3. Denoising-LM(no-author specific fine tuning): 논문에서 소개한 모델과 구조가 같으나(DAE loss) 전체 corpora로 fine tuning하여 문장생성, contents 보존능력을 평가하기 위한 베이스라인
5. Supervised Stylized Rewriting : 셰익스피어의 parallel data로 학습된 모델로 문장생성

## proposed evaluation method

1. 의미 보존 (contents preservation)
 
 BLEU metric, ROUGE(1,2,3)를 사용해 의미가 얼마나 보존됐는지를 평가합니다.

2. 스타일 변환 (stylistic alignment)

스타일 변환의 정도를 측정하는 척도에 대해, sentiment와 formality에 대한 스타일을 변환하는 기존의 연구방법론을 적용하기는 어렵습니다.
왜냐하면 해당 연구에서는 classifier에 기반한 척도를 사용했는데 감정이나 formal한 정도는 classifier로 나타내기가 비교적 용이하지만
문체는 그러한 방식을 적용하기가 어렵기 때문입니다. 따라서 본 논문에서는 multi-level 평가전략을 사용하여 stylistic 표현을 표면, 어휘, 구문 수준에서 평가합니다.

-  lexical elements : 작가들이 선택한 단어들은 주관적/객관적, 형식적/비형식적,  구체적/추상적, 문학적/일상적의 범주에 속할 수 있습니다. 이에 착안하여 각 8개의 범주의 어휘에 대해 seed words의 리스트를 정해두고 이들 단어와의 동시출현빈도를 고려하여 PMI를 계산하고 kNN을 통해 각 단어의 8개의 차원에 대한 점수를 구했습니다. 8개의 차원은 4개의 축에 걸쳐 나타나므로 전체 author-specific corpus에서 4개의 평균을 계산합니다. 각 평균값들은 [0,1]에 존재하여 각 작가의 문체가 주관적/객관적, 형식적/비형식적,  구체적/추상적, 문학적/일상적 중 어떠한 범주에 속하는지를 표현합니다.
-  Syntactic elements: 작가별로 필체가 단순할수도 복잡할 수도 있습니다. 이러한 점에 착안하여 각 어구를  1. simple 2. compound 3. complex 4. compound-complex 5. 나머지의 범주로 나눠볼 수 있습니다. feng, banerjee and choi(2012)에서 발표된 알고리즘을 이용해 각 어구를 5가지 범주로 나눕니다. 각 문장은 5가지 범주중 하나에 속하므로 전체 코퍼스에 있는 문장에 대한 5차원 벡터는 5개 범주에 대한 확률분포로 볼 수 있습니다.
-  Surface elements: 1. 컴마개수, 2. 세미콜론, 3. 각 문장마다의 콜론수, 4. 문단 내 문장 수, 5. 문장의 단어수와 같은 표면적인 특징을 이용해 5차원의 벡터로 표현하였습니다.

위의 각 레벨별 평가방식을 개별 작가의 corpus에 적용해 각 레벨마다 3개의 5차원의 벡터로 작가의 스타일을 숫자로 표현하였습니다.
또한 모델에 의해 만들어진 문장에 대해서도 해당 방식으로 5차원의 벡터를 3개 만들 수 있습니다.
이후 작가 코퍼스에 대한 레벨별 벡터와 만들어진 문장의 레벨별 벡터의 거리를 계산할 수 있습니다.
거리를 구할 때 metric으로는 lexical, surface elements에 대해서는 mse를 계산하고 syntactic에 대해서는 KL divergence를 사용했습니다.

## results and analysis

앞서 구축한 양적 평가지표에 대해 각 모델별 의미보존 score와 스타일 변환 score를 구한 것은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/85322951/205889417-cffff463-dee7-4f7f-a0af-62186ecfaeaa.png)

이를 통해 unsupervised baseline과 비교해봤을 때, 의미보존 score와 스타일 변환 score 모두에서 전반적인 score가 본 논문에서 제안한 styleLM이 가장 높다는 것을 알 수 있습니다.
Supervised baseline과의 비교에서는 몇몇 지표에서는 styleLM이 뒤졌지만 unparallel data를 가지고도 이정도의 비등한 효과를 냈다는 점에서 의의를 가집니다.

## conclusion and future work

본 논문에서는 언어모델의 일반화능력을 사용하고 이를 target author의 corpus를 이용해 fine tuning한다.
이때 DAE LOSS를 사용함으로써 원본을 복원하여 reconstruct하는 과정에서 stylistic adaptation이 발생하도록 유도한다.
또한 본 논문에서는 스타일 변환에 대한 metric에 대해 언어학적인 3가지 층위의 척도를 제시하였다.
또한 styleLM은 unsupervised learning방식 중에서는 가장 좋은 성능을 냈고, supervised learning에도 비견할 만한 성능을 내었다.

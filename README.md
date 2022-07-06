# 딥러닝을 이용한 중국어 번역모델 만들기
  

## 프로젝트 기획 배경  
+ 대학교에서 중국학부를 전공하면서 중국어의 난이도를 체감...
+ 중국어의 한자는 총 약 3만 6천개로, 통용되는 한자는 약 7천개이며, 자주 사용되는 한자만 추려도 약 3천개 이상
+ 특히 한자문화권이 아닌 국가에서는 세계에서 가장 배우기 어려운 언어 중 하나로 꼽힘
  + 아래는 영국 외무부에서 각 나라의 영국 대사관 직원들에게 현지 언어를 배우도록 한 뒤  
  설문조사 한 데이터를 모아 세계 언어의 난이도를 다섯 등급으로 나눈 표인데,  
  아랍어, 광동어, 표준중국어, 일본어, 한국어가 가장 배우기 어려운 등급으로 분류되어 있음
![image](https://user-images.githubusercontent.com/88722429/175809275-88747d13-e93b-41d7-b60b-35856b7ab3b9.png)  
  
+ 아래의 데이터 샘플을 살펴보면, 한국어는 단어별로 띄어쓰기, 품사를 나타내는 조사 등  
  읽거나 번역하는 데에 힌트가 될 요소들이 있지만, 중국어는 그렇지 않기 때문에  
  글이 길어질수록 해석 난이도는 급상승
![image](https://user-images.githubusercontent.com/88722429/175809417-3d9566f9-d21a-424e-b7a4-f9631438752d.png)  
  
  
  
> 인간이 꾸준히 공부하듯이 대량의, 양질의 데이터를 딥러닝모델에 훈련시킨다면?
  


## 데이터
+ [AI HUB 한국어-중국어 번역 말뭉치 데이터(사회과학)](https://aihub.or.kr/aidata/30721)
+ [AI HUB 한국어-중국어 번역 말뭉치 데이터(기술과학)](https://aihub.or.kr/aidata/30722)  
  
+ 데이터 구축분야
    + 사회과학 : 금융/증시, 사회/노동/복지, 교육, 문화재/향토/K-FOOD, 조례,  
    정치/행정, K-POP(한류)/대중문화_공연_콘텐츠
    + 기술과학 : 의료/보건, 특허/기술, 자동차/교통/소재, IT/컴퓨터/모바일
+ 한글 원문 어절 수 : 평균 15어절
+ 수량 : 병렬 말뭉치 130만개씩 => 총 260만개
+ 분야별 세부 구축 수량(단위:만)
    + 사회과학 : 금융/증시(20만), 사회/노동/복지(20만), 교육(10만), 문화재/향토/K-FOOD(15만), 조례(20만), 정치/행정(25만), K-POP(한류)/대중문화_공연_콘텐츠(20만)
    + 기술과학 : 의료/보건(25만), 특허/기술(15만), 자동차/교통/소재(30만), IT/컴퓨터/모바일(60만)
<img src='https://aihub.or.kr/sites/default/files/styles/max_2600x2600/public/2021-05/028.%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8B%E1%85%A5-%E1%84%8C%E1%85%AE%E1%86%BC%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8B%E1%85%A5%20%E1%84%87%E1%85%A5%E1%86%AB%E1%84%8B%E1%85%A7%E1%86%A8%20%E1%84%86%E1%85%A1%E1%86%AF%E1%84%86%E1%85%AE%E1%86%BC%E1%84%8E%E1%85%B5%28%E1%84%89%E1%85%A1%E1%84%92%E1%85%AC%E1%84%80%E1%85%AA%E1%84%92%E1%85%A1%E1%86%A8%29_%E1%84%83%E1%85%A2%E1%84%91%E1%85%AD%E1%84%83%E1%85%A9%E1%84%86%E1%85%A7%E1%86%AB.png?itok=AQ_3oXxE' width='700' height='200'>
=> 위 9가지 정보 중 한국어, 중국어 정보만 추출해 딥러닝에 사용
  


## 모델(Seq2Seq + Attention)
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHixMi%2FbtqDlkuE4XV%2FrtvT9hKMMMkxjVDnyggqj0%2Fimg.png'>
Seq2Seq 
+ RNN(순환신경망, recurrent nearal network) 기반으로, 인코더와 디코더 내부는 RNN으로 구성됨
+ 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용되는 모델  
  => 입력 시퀀스와 출력 시퀀스를 각각 입력 문장과 번역 문장으로 구성하면 번역기를 만들 수 있을 것으로 예상

+ 인코더
  + 요약) 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤, 마지막에 모든 단어 정보들을 압축해 하나의 벡터(context vector)로 만들어 디코더로 전송
  + 1. 입력받은 문장을 토큰화를 통해 단어 단위로 쪼갬
  + 2. 단어 토큰 각각은  임베딩 벡터로 변환 후 RNN 셀의 각 시점의 입력이 됨
  + 3. 인코더의 RNN 셀은 모든 단어를 입력받은 뒤 인코더 RNN 셀의 마지막 시점의 은닉상태(=context vector)를 디코더 RNN 셀로 넘겨줌

+ 디코더
  + 요약) context vector를 입력받아 번역된 단어를 하나씩 순차적으로 출력
  + 1. 초기 입력으로 문장의 시작을 의미하는 <sos>, 문장의 끝을 의미하는 <eos>가 들어감
  + 2. 훈련 과정에서 인코더가 보낸 context vector와 실제 정답을 입력받았을 때, 교사 강요(teacher forcing)을 통해 훈련
    + ex) <sos> I am a student 가 실제 정답일 때, I am a student <eos> 가 나와야 한다고 정답을 알려주면서 훈련
  + 3. 테스트 과정에서 context vector와 <sos>만을 입력받은 후에 다음에 등장할 확률이 높은 단어를 예측, 그 단어를 다음 시점의 RNN 셀의 입력값으로 사용
  + 4. 다음 단어로 <eos>가 예측될 때까지 2번을 반복 
  
+ 출력
  + 출력 단어가 될 확률이 있는 모든 단어들로부터 하나의 단어를 골라 예측하기 위해 softmax 함수 사용
  


> RNN 기반 모델의 문제점
1. 정보손실
+ context vector라는 하나의 고정된 크기의 벡터에 모든 정보를 압축
+ 문장이 길어질수록 모든 의미를 담기 힘들어지고, 앞쪽의 단어 정보는 잃어버릴 가능성이 높음
2. 기울기소실(vanishing gradient)
+ 활성화함수 tanh의 미분값을 역전파 과정에서 반복해서 곱해주기 때문에, recurrent가 반복될수록 기울기 폭발/소실로 인한 장기의존성 문제 발생
+ 즉, 문장이 길어질수록 앞의 정보가 뒤로 충분히 전달되지 못함 (말뭉치 데이터의 문장 최대 길이가 100자가 넘기 때문에 부적합)  
  

> Attention 메커니즘을 추가해 문제점을 극복해보자!
디코더에서 출력단어를 예측하는 매 시점마다 인코더의 전체 입력문장을 참고하는데,
전체 입력문장 중 해당 시점에서 예측해야할 단어와 연관있는 단어 부분을 좀더 집중(attention)해서 본다!
+ seq2seq에서 나아가, 문맥을 더 잘 반영할 수 있는 context vector를 구하여 매 시점마다 하나의 입력으로 사용할 수 있음
+ 1. 인코더의 각 시점마다 생성되는 hidden state vector를 간직해두었다가 모든 단어가 입력되면 한번에 디코더에 넘겨줌
+ 2. 각 vector는 디코더의 각 시점에서 query("나랑 비슷한 애 누구야?!")로 작용해 key들을 불러모음
+ 3. query에 대해서 모든 key와의 유사도를 각각 계산(해당 프로젝트에서는 바다나우(concat) 방법 사용)
+ 4. 각 유사도 값에 softmax 함수를 취해 총합 1의 확률값을 계산
+ 5. 각 확률값과 value(key의 값=단어의 의미)를 곱한 결과를 종합해 context vector 생성  
  + query-key 연관성이 높은 value 벡터 성분이 더 많이 들어가게 되어 문맥을 더 잘 반영하게 됨
+ 6. context vector와 디코더의 hidden state vector를 사용해 출력단어를 결정 
+ 7. 디코더는 인코더에서 넘어온 모든 hidden state vector에 대해 2~6의 계산 수행
  + 각 시점마다 출력할 단어가 어떤 인코더 시점의 어떤 단어 정보와 연관되어 있는지, 즉 어떤 단어에 attention할지 알 수 있음



## 진행사항
<img src='C:\Users\yeonok\Desktop\projects\NLP_Chinese_Translator\1.PNG'>  
(응...?)


## 문제점
1. 훈련데이터의 용량(208만개, 807MB)
+ 10만개만 사용해도 토큰화할 때 runtime error, 훈련할 때 ResourceExhaustedError: Graph execution error 등 GPU를 이용하는데도 컴퓨터가 다운되는 현상 발생...
+ batch size를 64에서 2까지 줄여보았으나 실패, 샘플데이터 개수를 점점 줄여보면서 훈련 시도  
=> 우선 1000개의 문장을 학습시키는 데에만 성공  
=> 훈련되지 않은 토큰(OOV, Out Of Vocabulary)은 예측할 수 없기 때문에 모델의 성능도 자연스럽게 떨어질 수밖에 없음
+ 토큰화할 때 subword tokenizer 활용해 OOV 문제를 해결해봐야겠다.
+ 추후에 GPU를 추가하거나, 모델을 계속해서 사전학습 모델로서 사용하는 방법 찾아봐야겠다.
2. '번역기'의 모호한 평가방법
+ 번역 특성 상 정답이 딱 떨어지지 않기 때문에, 1번의 문제점을 극복하여 그럴싸한 문장을 출력해내더라도 사람이 직접 번역 결과를 확인해보아야 함   
+ 추후 모델 성능개선 후 검증데이터셋 샘플링을 통해 확인해볼 예정
  
  
  
## 추가로 진행할 점
1. Transformer 모델도 생성해보기
+ RNN 없이 Attention 매커니즘을 극대화하는 모델로, 토큰을 순서대로 입력받지 않고 한번에 받아 병렬연산하기 때문에 시간이 적게 들 것으로 예상
+ seq2seq+Attention 모델에 비교했을 때 어떨지 궁금..
+ (GPU가 24GB 이상은 필요하다고 한다...)
2. subword tokenizer 적용하기
+ 현재는 토큰화 과정에서 조사와 단어가 다 붙어있는 상태로, 토큰의 재사용이 힘들고 OOV를 처리할 수 없음



### 참고자료
https://wikidocs.net/22893
https://aimb.tistory.com/181
https://github.com/fxsjy/jieba
https://github.com/YangBin1729/nlp_notes/blob/master/06-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/27-%E5%9F%BA%E4%BA%8EAttention%E7%9A%84%E4%B8%AD%E8%AF%91%E8%8B%B1(TensorFlow).ipynb
https://www.youtube.com/watch?v=WsQLdu2JMgI
https://www.tensorflow.org/text/tutorials/nmt_with_attention  
https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/gpu.ipynb  
https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ko  
https://ainote.tistory.com/15
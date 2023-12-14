# 차량 이미지 내의 스크래치 위치 파악

차량 이미지 내의 스크래치 위치 파악을 위한 Image Segmentation 모델 개발을 주제로 진행한 프로젝트이다. 넥스트랩 기업과 연계하여 개인으로 진행했고, 이미지 라벨링, 데이터 수집, 데이터 전처리, 모델링, 시각화 등의 역할을 수행했다.

[프로젝트 내용 설명 영상](https://drive.google.com/file/d/1RqmRlzD0GBRXMoh25818ANIhmK2HYAjx/view?usp=sharing)


## 프로젝트 개요 및 필요성

- 오늘날에는 쏘카, 그린카와 같은 어플의 등장으로 렌트카 서비스가 활발하게 이루어지고 있다. 대표적인 렌트카 어플 중 하나인 쏘카에서는 하루에 약 10만장의 차량 이미지 사진이 업로드 된다고 하는데 차량의 손상 여부를 확인하기 위해 사람이 이 모든 사진을 눈으로 검토하기에는 많은 시간과 비용이 요구된다. 

- 본 프로젝트는 이러한 문제를 해결하기 위해 차량의 손상 중 하나인 스크래치에 대해 집중적으로 학습하면서 차량 이미지에서 스크래치 위치를 검출해내는 Image Segmentation 모델을 개발했다.

- 모델이 스크래치의 유무와 위치를 분류한다면 차량 이미지에서 파손을 검토하는 인원이 일일이 모든 사진을 확인해야 하는 번거로움이 줄어든다는 것을 해당 프로젝트의 기대효과라고 할 수 있다.

- 목표: 육안으로 봤을 때 명확한 스크래치를 분류하는 모델, IoU Score가 0.20 이상인 모델을 개발하는 것.

## 프로젝트 파이프라인

1. 데이터는 넥스트랩 기업과 AI-Hub에서 제공. 

2. 제공 받은 데이터는 파이썬과 Open CV 라이브러리를 활용해 전처리(256 x 256으로 크기 통일, 색상 조정, 조명과 빛 반사 제거를 위한 광원 제거, 마스킹 이미지 생성 등).

3. 딥러닝 프레임워크 중 하나인 Pytorch를 기반으로 U-Net, U-Net3Plus, DeepLab V3+ 모델을 개발. 

4. 모델의 예측 결과는 파이썬의 matplotlib 라이브러리를 활용하여 시각화.

![image](https://user-images.githubusercontent.com/97672187/183606324-7ea14e26-893a-4528-a0d0-e88f98953549.png)

## 데이터 설명

- 수집 및 모델에 사용한 데이터: 넥스트랩 기업 제공(3,000) + AI-Hub 사이트 제공(13,000) = 약 16,000개

- 학습 데이터: 13,000개

- 검증 및 테스트 데이터: 3,000개(학습할 데이터가 충분하지 않다고 판단하여 검증 데이터와 테스트 데이터를 동일하게 사용)

- 데이터 용량: 약 1GB

- 원본 이미지 / 마스킹(라벨 이미지)

![image](https://user-images.githubusercontent.com/97672187/183614179-e88575d2-2cfb-4f9d-9043-96bb423b71ad.png)

- 원본 / 흑백/ 광원제거

![image](https://user-images.githubusercontent.com/97672187/183614333-50eb308f-2afe-4cdd-9886-5962a74d488b.png)

## 모델링
손실함수는 주로 Dice Loss를 사용했고, Optimizer는 Adam, 평가지표는 IoU Score를 사용했다.

### 1. 사용한 모델

1) U-Net

- Segmentation 모델 중 하나로 2015년에 등장했고, 이전 층의 공간 정보를 최대한 보존하면서 이미지를 픽셀 단위로 분류시키는 모델이다.

- Encoding path의 Downsampling 과정에서 만들어진 피처맵의 일부를 Decoding path의 Upsampling 과정에서 사용함으로써 이전층의 공간 정보 손실을 최소화 시키려는 것이 특징이다.

- Downsampling 과정에서는 합성곱, ReLU 활성화 함수, Max Pooling 연산을 통해 채널의 수는 늘리고, 피처맵의 크기는 줄인다.

- Upsampling 과정에서는 Transpose Convolution과 Concatenate를 활용하여 Downsampling에서 줄어든 피처맵의 크기를 원본과 비슷한 사이즈로 키우고, 채널의 수는 줄인다(채널은 각 class의 갯수. 이진분류면 채널이 1).

2) U-Net3Plus

- U-Net의 상위 모델로 2020년에 등장했고, full scale로 이루어진 skip connection을 가진 모델이다. low-scale, high-scale의 피처맵을 합쳐서 이전의 U-Net 모델들보다 적은 파라미터로도 정확한 결과를 얻을 수 있다.

- 원하는 class의 정확한 위치와 class들간의 경계를 잘 인식하는 맵을 생성하기 위해 Focal loss, Multi-Scale SSIM, IoU loss를 혼합한 hybrid loss를 사용했고, decoder의 stage마다
classification guidance module을 사용하여 정확도를 높였다.

3) DeepLab V3+

- DeepLab V3+ 모델은 2018년 구글에서 발표한 모델이고, DeepLab 시리즈 중 가장 높은 성능을 내는 모델로 알려져있다.

- DeepLab 시리즈에서 사용되는 핵심 개념 중 하나인 Atrous convolution는 필터 내부에 빈 공간을 두고 연산을 함으로써 한 픽셀이 볼 수 있는 영역을 나타내는 field of view를 키우는 연산 방법이다. 즉, 한 픽셀이 커버할 수 있는 영역이 넓을수록 많은 정보를 저장할 수 있기 때문에 DeepLab 시리즈 모델은 이 Atrous convolution 개념을 활용한다. 특히, DeepLab V2 이후의 모델들은 피처맵에 여러 rate의 Atrous convolution를 병렬로 적용하고 결과를 합치는 ASPP(Atrous spatial pyramid pooling)의 개념이 기본적으로 사용되고 있다.

- DeepLab V3+는 field of view를 크게 가져가면서, ASPP를 각 채널마다 독립적으로 수행한 후 결과를 합치는 ASSPP(Atrous Separable Spartial Pyramid Pooling) 개념을 활용하여 파라미터수와 연산량을 줄였다. 이전 시리즈인 DeepLab V3와는 달리 Encoder에도 ASSPP를 적용했고(Xception), Decoder에는 기존의 upsampling 방법을 U-Net style처럼 변경해서 성능을 높였다. 즉, U-Net과 유사하게 intermediate connection을 가지는 encoder-decoder 구조를 적용하여 기존 DeepLab 시리즈보다 정교한 예측을 할 수 있게 한다.

### 2. 성능

1) 1차 모델링(U-Net, DeepLab V3+ 사용)

1차 모델링에서는 주어진 데이터에 AI-Hub에서 제공한 데이터를 추가하여 학습에 사용했다. 그 결과 데이터를 추가하고 Epochs를 100, Batch size 4, Learning rate 0.001, Optimizer는 Adam을 사용했을 때 검증 IoU Score가 **0.1916**으로 향상 되었다. DeepLab V3+ 모델도 개발했지만 U-Net에 비해 성능이 좋지 못해서 추가로 학습시키진 않았다. 또한, 조명과 빛 반사가 있는 부분에 스크래치가 있는 경우 모델이 이를 잘 분류하자 못하는 문제가 있어서 광원을 제거하여 학습시켰지만, 기대와는 달리 성능은 오히려 떨어졌다. 광원을 제거하면서 빛 뿐만 아니라 이미지 자체가 약간 흐려진 것이 원인이 된 것 같다.

2) 2차 모델링

2차 모델링에서는 원본 이미지를 Gray Scale로 바꾸고, Optimizer, 학습률 계획법 등의 하이퍼파라미터를 튜닝했다. 그 결과 Epohcs를 100, Batch size 4, Learning rate 0.0001, Optimizer는 Adam, 학습률 계획법은 Cosine Annealing Warm Restarts를 사용했을 때 기존 U-Net 모델의 성능보다 높은 **0.2107**의 검증 IoU Score를 기록했고, 프로젝트의 목표였던 0.20 이상의 IoU Score 또한 달성했다.

3) 3차 모델링

3차 모델링에서는 U-Net의 상위 모델인 U-Net3Plus 모델을 개발했다. U-Net에 비해 학습시간이 매우 오래걸려서 Epochs를 10으로 하고, 나머지 조건은 모두 동일하게 한 후 기존의 U-Net 모델과 U-Net3Plus 모델을 비교한 결과 U-Net 모델은 검증 IoU Score가 **0.1602**, U-Net3Plus 모델은 **0.1831**을 기록함으로써 U-Net3Plus 모델의 성능이 더 좋았다. U-Net3Plus 모델을 더 깊게 학습시키면 기존의 U-Net 모델보다 더 성능이 높은 모델을 만들 수 있을 것으로 기대한다.

## 결론

- 예측 결과 시각화

![image](https://user-images.githubusercontent.com/97672187/183616510-42e7285b-82e1-44ea-8235-357db2788761.png)

위의 두 사진은 원본 이미지와 라벨 이미지고, 그 옆에 오른쪽에 있는 사진은 손실함수로 Binary Cross Entropy를 사용했을 때의 U-Net 모델의 예측 결과, 밑의 사진은 손실함수로 Dice Loss를 사용했을 때 여러 모델들의 예측 결과이다.

- 초기에는 손실함수로 Binary Cross Entropy를 사용했는데 불균형 데이터의 특성상 모델이 모든 픽셀을 0의 클래스로 즉, 스크래치가 없는 픽셀로 분류를 하더라도 손실이 매우 낮게 나온다는 문제점이 있다. 해당 이미지에서는 0의 클래스의 비율이 약 97퍼센트 이상이기 때문에 모델이 모든 픽셀을 배경으로 예측 해도 매우 높은 정확도를 가지는 것처럼 학습하게 된다는 것이다. 

- 따라서 배경인 부분보다 스크래치가 난 부분에 초점을 둔 Dice Loss를 손실함수로 사용했다. 

- 밑의 사진의 모델들은 모두 다른 형태로 스크래치를 분류했는데 밑에서 4번째 그림인 최고 성능을 낸 U-Net 모델이 가장 비슷한 예측을 한 것을 알 수 있다. 광원을 제거한 데이터를 학습시킨 모델은 다른 U-Net 모델들에 비해 부정확한 예측을 한 것을 볼 수 있는데 광원을 제거하면서 스크래치가 난 부분 또한 흐릿하게 되어버린 것이 원인이 된 것으로 판단된다.

- DeepLab V3+은 해당 이미지 외의 다른 이미지도 U-Net에 비해 예측률이 떨어져서 더 학습시키진 않았다.

- 추가로, 에포크를 10만 주고 학습했던 U-Net3Plus 모델도 비슷한 형태로 스크래치를 분류한 것을 확인할 수 있다.

## 한계점 및 해결방안

1. 스크래치 외의 손상에 대해서는 분류하지 못했다.

    -> 스크래치 외의 손상 데이터를 제공받는다면 찍힘, 벌어짐 등과 같은 다른 손상들도 다중 분류 문제로 해결할 수 있을 것 같다. 

2. 데이터 양이 적어서 과적합을 해결하기 어려웠다.

3. 상위 모델인 U-Net3Plus 모델을 더 깊게 학습시키지 못했다. 

   -> 위 2가지 문제는 더 많은 시간과 좋은 GPU 환경이 주어진다면 훨씬 더 많은 데이터를 사용하고, 성능이 좋은 U-Net3Plus 모델을 더 깊게 학습시킴으로써 지금보다 더 좋은 모델을 만듦으로 해결할 수 잇다.

   -> 또한, UNet 외에도 다양한 모델들을 사용해봄으로써 성능을 비교할 수 있다.

**개인적으로 진행했던 Computer Vision 분야의 첫 프로젝트였고 Segmentation에 대해 공부할 수 있었던 매우 의미있는 시간이었다. 본 프로젝트를 통해 Segmentation 뿐만 아니라 Object Detection task에도 도전하고 싶다는 마음이 생겼고, 앞으로 여러 논문을 읽어가며 Computer Vision 분야에 대해 많은 지식과 경험을 습득하고싶다!!**

## 개발환경

![image](https://user-images.githubusercontent.com/97672187/183615283-dcb227d4-1cdc-4ab9-baf4-e2567c44d5f9.png)


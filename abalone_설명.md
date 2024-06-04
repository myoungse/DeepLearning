# abalone

### 과제 1) 런닝 레이트 조정
---

사용언어 : pycharm

대상코드 : abalone

강의자료 1의 abalone.ipynb 코드를 pycharm에 입력시켰음. (1)과 (2)를 확인하기 위해 디버깅을 해야함. 처음 디버깅이 필요할 것 같은 부분인
※ 편의상 def run_train(x,y):부터 line 1~ 로 부르겠습니다.


```
1> def run_train(x, y):
2>    output, aux_nn = forward_neuralnet(x)
3>    loss, aux_pp = forward_postproc(output, y)
4>    accuracy = eval_accuracy(output, y)
5>
6>    G_loss = 1.0
7>    G_output = backprop_postproc(G_loss, aux_pp)
8>    backprop_neuralnet(G_output, aux_nn)
9>
10>   return loss, accuracy
```  
* 첫 번째 디버깅
1) line 1과 line 4에 Breakpoint를 찍고 디버깅
2) line 1, line 4, line 10에 Breakpoint를 찍고 디버깅
3) line 1, line 10에 Breakpoint를 찍고 디버깅
을 해봤지만 결과가 계속 0이라고만 나와서 cmd 창에 pip install matplotlib를 입력 후 설치.
설치 후, 기존 코드에 print(run_train(1,2)) 추가

```
def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy

print(run_train(1,2))
```
다시 Breakpoint를 찍음.

* 두 번째 디버깅
  
print(run_train(1,2)) 줄만 Breakpoint를 찍고 디버깅을 하니 결과가 나왔지만 캡쳐하는 것을 깜박..

<img width="756" alt="q" src="https://github.com/myoungse/DeepLearning/assets/106461144/31c6375c-a0dc-42a1-8ff5-5e56429a237a">

디버깅을 하지 않고 돌렸을 때의 오류 모습인데, 디버깅을 했을 때와 같은 오류가 뜨는 것을 볼 수 있음.
forward_neuralnet이 정의되지 않았다며 NameError가 뜬다.

//다시 print(run_train(x,y)) 코드를 없앤 후, 원래의 코드로만 실행시켰더니

<img width="767" alt="w" src="https://github.com/myoungse/DeepLearning/assets/106461144/c56501d6-4f7d-4697-9cc0-2d10ce57d1a1">

아까 나왔던 오류가 뜨지 않는다. run_train을 print할 때 forward_neuralnet을 불러오는데에 있어 문제가 생긴듯하다.

---

### 과제 수행

#### 1) 메인 함수 입력

chap1 회귀분석 ppt에 나왔던 부분을 코드의 맨 마지막 줄에 입력시켰다. 

    > ppt p.24 "실험용 메인 함수 abalone_exec() 정의"

abalone_exec()를 입력 후 실행시키니 epoch, loss, accuracy 값이 차례로 나왔다.

#### 2) 하이퍼파라미터 값 조정

ppt에서는 "epoch 추가에 따른 성능 향상 효과 없음 확인" 이라고 나왔지만 그래도 건드려보았다.
    
하이퍼파라미터 값 중, epoch_count (에포크 값), mb_size (미니배치 사이즈), LEARNING_RATE(러닝레이트값)을 바꿔
보았다.

__epoch : 전체 데이터를 n번 사용해서 학습을 거치게 하는 것__
__mb_size : 정해진 수만큼의 data를 학습하고 학습률을 높임.__
__LEARNING_RATE : 너무 크면 최소점 찾지 못하고 overshooting이 생김. 반면 너무 작으면 local minimum에 빠지고 학습 오래 걸림. 일반적으로 0.01로 설정함.__

> (1) 원래 코드를 돌렸을 때.

<img width="422" alt="원본" src="https://github.com/myoungse/DeepLearning/assets/106461144/ee3cc83f-7bf5-48b5-9314-bcf7fa07d5ef">

    최종 성능은 0.736이 나왔다.

> (2) epoch (10 > 450) , mb_size (10), LEARNING_RATE (0.001 > 0.05)

<img width="410" alt="다시" src="https://github.com/myoungse/DeepLearning/assets/106461144/28d6e326-5854-47d8-8d6e-59613d9ecc55">

  최종 성능 0.848

#### 3) "중간 관찰값(1)인 accuracy 결과를 이용해서 backprop_neuralnet 함수(2)에서 weight, bias의 기울손실률 적용 과정에서의 학습이 잘 되도록 작성해보자."

과제란에 표시된 코드에 각각 디버깅을 걸고 실행해보았다.

> accuracy = eval_accuracy(output, y)

<img width="1220" alt="1" src="https://github.com/myoungse/DeepLearning/assets/106461144/83507478-582b-4e31-b35d-3e01691c5e1a">


중간값을 관찰했을 때의 accuracy 값 (output, y) 값이다.

> backprop_neuralnet(G_output, aux_nn, accuracy)

<img width="1225" alt="2" src="https://github.com/myoungse/DeepLearning/assets/106461144/a876dee6-92a6-4564-8ba4-842d9fde5850">

weight, bias의 기울손실률 적용 과정이다. weight와 bias는 앞의 init_model에서 정의되며 이 부분이 뒤의 backprop_neuralnet에서 적용되는 것 같다.

---
#### 특별 조건
    이 실험을 해보고 과제 주제의 실험 설계상 문제점이나 문제가 없는 이유에 관한 내용을 정확하게 작성

문제점은 def run_train 코드 내 backprop_neuralnet(G_output, aux__nn)에 accuracy가 들어가있지않다. 그래서 코드를 작동시킬 때 따로 이 부분을 추가해서 넣어줬다.
반면 전복의 성별에 대한 부분은 원핫벡터를 사용하여 불필요한 선형성을 업생 학습에 유리하다는 점이 있다.

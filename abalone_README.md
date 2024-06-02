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

<img width="758" alt="a" src="https://github.com/myoungse/DeepLearning/assets/106461144/8673973c-3a4b-41f4-bf67-4c6ce9c8f605">

원래의 코드로 되돌리고 디버깅을 하지 않고 실행한 결과이다.

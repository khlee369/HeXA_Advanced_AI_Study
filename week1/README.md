# 개발환경 세팅
1. install anaconda
2. install vscode(optional)

## Anaconda prompt를 실행한후

가상환경 만드는법
```
conda create -n 이름 python=버전
```

가상환경 list
```
conda env list
```

가상환경 activate
```
conda activate 가상환경_이름
```

가상환경 deactivate
```
conda deactivate
```

## jupyter notebook

base 환경에서
```
conda install nb_conda
```

가상환경에서
```
conda install nb_conda
```

base 환경에서
```
(base)$ jupyter notebook
```

## jupyter notebook 사용법
### 셀 추가
1. a(위에 추가)
2. b(밑에 추가)
3. x(셀 삭제)

### 셀 실행
1. shift + enter(실행시키고 아래 셀로 이동)
2. ctrl + enter(현재 셀 실행)

# 다음주까지

가상환경에서
1. Install CUDA version match with your GPUs (https://mickael-k.tistory.com/18)
2. Install cudnn (https://mickael-k.tistory.com/18)
3. Install pytorch-gpu version match with CUDA (https://pytorch.org/)
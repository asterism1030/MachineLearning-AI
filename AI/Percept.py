""" 넘파이는 행렬이나 일반적으로 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원함 """
""" 데이터 구조 외에도 수치 계산을 위해 효율적으로 구현된 기능을 제공 """
import numpy as np

class Perceptron(object):
    """ 퍼셉트론 분류기
    -> 뇌의 뉴런 하나가 작동하는 방식을 흉내 내려는 환원주의 접근 방식 사용
        출력을 내거나/내지않거나 (1 / 0)
        
        요약하면 아래와 같음
        1. 가중치를 0 또는 랜덤한 작은 값으로 초기화
        2. 각 훈련 샘플 x(i) 에서 다음 작업을 함
            a. 훈련값 y^ 계산
            b. 가중치를 업데이트 

    -> 선형적으로 분류 가능한 데이터셋 처리. 분류X 다면 훈련 데이터셋을 반복할 최대 횟수(에포크 epoch)를 지정하고 분류 허용 오차 지정 가능
    
    매개변수
    ---------------------
    eta : float
        학습률 (0.0 과 1.0 사이)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    ---------------------
    w_ : 1d_array
        학습된 가중치
    errors_ : list
        에포크마다 누적된 분류 오류

    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ 훈련 데이터 학습
        
        매개변수
        ---------------------
        x : {array-like}, shape = [n_samples, n_features]
            n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
            타깃 값

        반환값
        ---------------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        """ 다른 메소드를 호출하여 초기화한 속성은 밑줄 사용
        가중치 : 벡터 R^m+1 로 초기화 (m : 데이터셋에 있는 차원-특성 개수, 절편을 위해 +1 즉 self.w_[0] : 절편)
        """
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ 최종 입력 계산 """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ 단위 계단 함수를 사용하여 클래스 레이블을 반환 """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

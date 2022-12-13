from scipy.optimize import minimize
import numpy as np


class rnetsy:
    def __init__(self, u, y):
        self.N = np.shape(u)[0]
        self.u = u
        self.y = y
        self.delta = []
        self.mutable_w = []
        for j in range(11):
            self.mutable_w.append(10)
        for j in range(self.N):
           # print(j)
            self.delta.append(self.u[j]/20)
        self.last_w = self.mutable_w
        self.threshold1 = 1e-8
        self.threshold2 = 1e-8
        self.moving_delta = []
        self.last_delta = self.moving_delta
        self.moving_u = []
        self.moving_y = []


    def step1(self):
        fun1 = lambda solve_w: ((solve_w[0]*self.moving_u[2]+ solve_w[1]*self.moving_u[1] + solve_w[2]*self.moving_u[0])-(solve_w[3]*self.moving_delta[2] + solve_w[4]*self.moving_delta[1] + solve_w[5]*self.moving_delta[0]))**2  + \
        ((solve_w[6]*self.moving_y[2] + solve_w[7]*self.moving_y[1] + solve_w[8]*self.moving_y[0]) - (solve_w[9]*self.moving_delta[1] + solve_w[10]*self.moving_delta[0]))**2 
        #print(last_w)
        temp = minimize(fun1, x0=self.last_w)
        self.mutable_w = temp.x

        #print("temp.x: ", temp.x)
        
        
    def step2(self):
        fun2 = lambda solve_delta:((self.mutable_w[0]*self.moving_u[3] + self.mutable_w[1]*self.moving_u[2] + self.mutable_w[2]*self.moving_u[1]) - (self.mutable_w[3] * self.moving_delta[3] + self.mutable_w[4] * solve_delta[1] + self.mutable_w[5]*solve_delta[0]))**2 \
        +((self.mutable_w[6]*self.moving_y[3] + self.mutable_w[7]*self.moving_y[2] + self.mutable_w[8]*self.moving_y[1]) - (self.mutable_w[9]*solve_delta[1] + self.mutable_w[10]*solve_delta[0]))**2
        temp = minimize(fun2, x0=(self.last_delta[1], self.last_delta[2]))
        self.moving_delta[1] = temp.x[0]
        self.moving_delta[2] = temp.x[1]
        self.last_delta = self.moving_delta
        

    def judgement(self):
        sum = 0
        #print(self.moving_delta)
        #print(self.last_delta)
        #print(self.mutable_w)
        for i in range(np.shape(self.mutable_w)[0]-1):
            sum += (self.mutable_w[i]-self.last_w[0])**2
        abs1 = np.abs(self.moving_delta[1] - self.last_delta[1])
        abs2 = np.abs(self.moving_delta[2] - self.last_delta[2])
        #print("sum: ", sum, " abs1: ", abs1, " abs2: ", abs2)
        if (abs1 > self.threshold1 and abs2 > self.threshold2 and sum > self.threshold1):
            return True
        else:
            return False

    def TwoRecursive(self):
        flag = True
        while(flag):
            print("here")
            #print("mutable_w: ", self.mutable_w)
            self.step1()
            self.step2()
            self.last_w = self.mutable_w
            self.last_delta = self.moving_delta
            if (not self.judgement()):
                flag = False

    def run(self):
        for i in range(self.N-3):
            self.moving_delta = self.delta[i:i+4]
            self.last_delta = self.moving_delta 
            self.moving_u = self.u[i:i+4]
            self.moving_y = self.y[i:i+4]
            self.TwoRecursive()
            self.delta[i+1] = self.moving_delta[1]
            self.delta[i+2] = self.moving_delta[2]
            #print(np.shape(self.delta)[0])
        print("self.mutable_w: ", self.mutable_w)
        #print("delta: ", self.delta)

def test():
    u = []
    for i in range(10000):
        u.append(np.sin(i))
    b0 = 1
    b1 = 1
    b2 = 1
    a0 = 1
    a1 = 1 
    a2 = 1
    c0 = 1
    c1 = 1
    c2 = 1
    d0 = 1
    d1 = 1
    rdelta = []
    y = []
    #print(np.shape(u)[0]-1)
    for i in range(np.shape(u)[0]):
       # print(i)
        if i <= 1:
            rdelta.append(0.0)
        else:
            temp1 = ((b0*u[i]+b1*u[i-1]+b2*u[i-2])-(a1*rdelta[i-1]+a2*rdelta[i-2]))/a0
            rdelta.append(temp1)
    
    for i in range(np.shape(rdelta)[0]):
       # print(i)
        if i <= 1:
            y.append(0.0)
        else:
            temp2 = ((d0*rdelta[i-1] + d1*rdelta[i-2]) - (c1*y[i-1]+c2*y[i-2]))/c0
            y.append(temp2)
    
    print("true u: ", u)
    #print("true delta: ", rdelta)
    #print("true y:", y)
    r1 = rnetsy(u, y)
    r1.run()

        
test()

            
    

    












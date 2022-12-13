import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import csv


# b < a
def Preisach(x, u, y):
    n = 10
    N = int(n * (n+1)/2)
    a0 = 10
    b0 = 10

    #mu = np.zeros([N, 3])
    """
    a1 b1 m1
    a2 b2 m2
    .......
    """

    da = 2*a0 / (n+1)
    db = 2*a0 / (n+1)

    grid = [[-a0+da, -b0+db]]
    #print(grid)
    for i in range(1, n):
        a = grid[0][0]+da*i
        for j in range(0, i+1):
            b = grid[0][1]+db*j
            if (abs(a) < 1e-4):
                a = 0
            elif (abs(b)<1e-4):
                b = 0
            #print(a, b)
            grid.append([a, b])
            
    #print("grid: ", grid)
    #print("grid's shape: ", np.shape(grid))
    mu = [[grid[0][1], grid[0][0]]]
    for i in range(np.shape(grid)[0]-1):
            if (abs(grid[i][1] - grid[i][0]) > 1e-4):
                mu.append([grid[i][1], grid[i][0]])
            else:
                pass

    for i in range(N-np.shape(mu)[0]):
        mu.append([0, 0])
    #print("mu's shape: ", np.shape(mu))
    #print("mu: ", mu)
    
    L = len(u)
    #print(grid) 
    #print(mu)
    #print(np.shape(mu))
    switch = []
    sum = []
    for i in range(np.shape(mu)[0]):
        switch.append(-1)
    
    for i in range(L):
        sum.append(0)
    
    for i in range(0, N):
        if ((switch[i] == -1) and mu[i][1] <= u[1]):
            switch[i] = 1
        sum[1] += switch[i]*x[i]/N

    for k in range(1, L):
        if u[k] > u[k-1]:
            for i in range(0, N):
                if ((switch[i] == -1) and (mu[i][1] <= u[k] or abs(mu[i][1]-u[k]) < 1e-4)):
                    switch[i] = 1
                else:
                    pass
    
        elif u[k] < u[k-1]:
            for i in range(0, N):
                if switch[i] == 1 and mu[i][0] > u[k]:
                    switch[i] = -1 
                else:
                    pass
        for i in range(0, N):
            sum[k] += switch[i]*x[i]/N 

    cost = 0
    for i in range(len(y)):
        cost += (y[i]-sum[i])**2

    return cost



    # plt.figure(1)
    # plt.plot(u, sum)
    # plt.xlabel('Input')
    # plt.ylabel('Output')

    # plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.plot(u)
    # plt.ylabel('Input')
    # plt.subplot(2,1,2)
    # plt.plot(sum)
    # plt.ylabel('Output')
    # plt.show()
def draw_preisach(x, u, y):
    n = 10
    N = int(n * (n+1)/2)
    a0 = 10
    b0 = 10

    #mu = np.zeros([N, 3])
    """
    a1 b1 m1
    a2 b2 m2
    .......
    """

    da = 2*a0 / (n+1)
    db = 2*a0 / (n+1)

    grid = [[-a0+da, -b0+db]]
    #print(grid)
    for i in range(1, n):
        a = grid[0][0]+da*i
        for j in range(0, i+1):
            b = grid[0][1]+db*j
            if (abs(a) < 1e-4):
                a = 0
            elif (abs(b)<1e-4):
                b = 0
            #print(a, b)
            grid.append([a, b])
            

    mu = [[grid[0][1], grid[0][0]]]
    for i in range(np.shape(grid)[0]-1):
            if (abs(grid[i][1] - grid[i][0]) > 1e-4):
                mu.append([grid[i][1], grid[i][0]])
            else:
                pass

    for i in range(N-np.shape(mu)[0]):
        mu.append([0, 0])
    
    L = len(u)
    switch = []
    sum = []
    for i in range(np.shape(mu)[0]):
        switch.append(-1)
    
    for i in range(L):
        sum.append(0)
    
    for i in range(0, N):
        if ((switch[i] == -1) and mu[i][1] <= u[1]):
            switch[i] = 1
        sum[1] += switch[i]*x[i]/N

    for k in range(1, L):
        if u[k] > u[k-1]:
            for i in range(0, N):
                if ((switch[i] == -1) and (mu[i][1] <= u[k] or abs(mu[i][1]-u[k]) < 1e-4)):
                    switch[i] = 1
                else:
                    pass
    
        elif u[k] < u[k-1]:
            for i in range(0, N):
                if switch[i] == 1 and mu[i][0] > u[k]:
                    switch[i] = -1 
                else:
                    pass
        for i in range(0, N):
            sum[k] += switch[i]*x[i]/N 

    without_hys = []
    sum_draw = []
    y_draw = []
    for i in range(len(y)):
        without_hys.append((y[i]-sum[i])*25)
        sum_draw.append(sum[i]*25)
        y_draw.append(y[i]*25)
        
    
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.ylabel('real ouput')
    plt.plot(u)
    plt.plot(y_draw)
    plt.subplot(3,1,2)
    plt.plot(sum_draw)
    plt.ylabel('hysteresis\'s effect')
    plt.subplot(3,1,3)
    plt.plot(u)
    plt.plot(without_hys)
    plt.xlabel('time')
    plt.ylabel('\'improved\' output')

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.scatter(u, y_draw)
    plt.ylabel('real relationship')
    plt.subplot(2,1,2)
    plt.scatter(u, sum_draw)
    plt.ylabel('detected hysteresis')
    plt.show()


filename1 = '../20221110T112637_j7-l4e-LFWSRXSJ5M1F48984_4_125to716_vehicle_control_cmd.csv'
#filename1 = '1110_48984_trailer_amp15T4_vehicle_control_cmd.csv'
#filename2 = '1110_48984_trailer_amp15T4_vehicle_dbw_reports1.csv'
filename2 = '../20221110T112637_j7-l4e-LFWSRXSJ5M1F48984_4_125to716_vehicle_dbw_reports1.csv'

def readfile(name):
    fields = []
    rows = []
    with open(name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        
        for row in csvreader:
            temp = []
            s = ' '.join(str(x) for x in row)
            seq = [float(x) for x in s.split(' ', 5)]
            for x in seq:
                temp.append(x)
            rows.append(temp)

    print(fields)
    return rows

def test():
    #t = np.linspace(0, 20, 100)
    #u = np.sin(t) * 11
    #y = hysteresis(u)
    #plt.figure(1)
    #plt.plot(t, u)
    #plt.plot(t, y)
    #plt.figure(2)
    #plt.scatter(u, y)
    #plt.show()
    control_cmd = readfile(filename1)
    dbw_reports = readfile(filename2)
 
    cmd = []
    for i in range(np.shape(control_cmd)[0]-1):
        cmd.append(control_cmd[i][2])

    time = []
    init_time = control_cmd[0][0]
    for i in range(np.shape(control_cmd)[0]-1):
        control_cmd[i][0] -= init_time
        time.append(control_cmd[i][0])

    yawrate = []
    init_yaw = dbw_reports[0][1]
    for i in range(np.shape(dbw_reports)[0]-1):
        dbw_reports[i][1] -= init_yaw
        yawrate.append(dbw_reports[i][1])


    n = 10
    N_start = 20
    N_stop = 2000 #450
    use_input = cmd[N_start:N_stop]
    use_output = []
    draw_output = []
    for i in range(N_stop-N_start):
        use_output.append(yawrate[i+N_start])
        draw_output.append(yawrate[i+N_start]*25)

    x0 = np.squeeze(np.random.rand(int(n*(n+1)/2), 1))
    res = optimize.least_squares(Preisach, x0, args=(use_input, use_output), verbose=2, ftol=1e-1)

    print(res.x)
    draw_preisach(res.x, use_input, use_output)
    






test()


            
    

        


    

        









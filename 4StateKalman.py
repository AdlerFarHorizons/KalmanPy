import numpy as np
import matplotlib.pyplot as plt

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def demo_kalman_xy():
    x = np.matrix('0. 1200. 0.').T 
    P = np.matrix('''
        1000. 0. 0.;
        0. 1000. 0.;
        0. 0. 0.
    ''') # initial uncertainty

    N = 100
    c = 0
    time = []
    true_x = []
    #true_y = []
    while c <= N:
        true_x.append(1200*c)
        time.append(c)
        #true_y.append()
        c += 1

    result = []
    R = 100**2
    observed_x = true_x + np.random.normal(0.0, np.sqrt(R), N+1)
    #observed_y = true_y + 0.03*np.random.random(N)*true_y
    for i in range(N+1):#for meas in observed_x:#, observed_y):
        if i == 0:
            meas = observed_x[0]
        else:
            meas = observed_x[i-1]
            
        x, P = kalman(x, P, meas, R,
              motion = np.matrix('0. 0. 0.').T,
              Q = np.multiply(np.matrix('''
                1. 1. 0.;
                1. 1. 0.;
                0. 0. 0.
                '''), 10**2),
              F = np.matrix('''
                1. 1. 0.;
                0. 1. 0.;
                0. 0. 0.
                '''),
              H = np.matrix('''
                1. 0. 0.
                '''))
        result.append(x.flat[0])
    kalman_x = result

    #plot the estimate vs the process
    #plt.plot(time, observed_x, 'r-', label="Observed altitude")
    plt.plot(time, true_x, 'y-', label="True altitude")
    plt.plot(time, kalman_x, 'g-', label="Kalman estimate")
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (feet)")
    plt.show()
    
    #plot the errors
    error = np.subtract(kalman_x, true_x)
    print(np.average(error))

    plt.plot(time, error, 'y-', label="Error")
    #legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Error (feet)")
    plt.show()

    

    

demo_kalman_xy()

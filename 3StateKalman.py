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
    
    # LN - The order was changed to PREDICT then UPDATE. The original had this
    # reversed.
    
    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    return x, P

# LN - There is a complete conversion to matrix operations from what was 
# discrete equations.
def demo_kalman_xy():
    P = np.matrix('''
        1000. 0. 0.;
        0. 1000. 0.;
        0. 0. 1000.
    ''') # initial uncertainty

    N = 90
    c = 0
    
    sigma_pv = 10.0
    sigma_pa = 2.0
    time = [0]
    true_F = np.matrix([[1.0, 1.0, 0.5],
                   [0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0]])
    G = true_F
    
    # LN - The simulated process (truth) and result matrices are pre-allocated. 
    # The append approach that expands the arrays on the fly only works with
    # vectors.
    true_X = np.zeros([3,N+1])
    true_noise = np.matrix( [ 0.0,
                              np.random.normal(0.0, sigma_pv),
                              np.random.normal(0.0, sigma_pa) ]).T
    # LN - The very strange Python way to access the 0th column of true_X
    true_X[:,0:1] = np.add( np.matrix( [0.0, 1200.0, 0.0] ).T, true_noise )
    c = 1
    while c <= N:
        # LN - Process simulation: state update equations
        true_noise = np.matrix( [ 0.0,
                                np.random.normal(0.0, sigma_pv),
                                np.random.normal(0.0, sigma_pa) ]).T
        # LN - Again, the Python way for accessing the (c-1)th column of true_x
        temp = np.dot( true_F, true_X[:,c-1:c] )
        # LN - Accessing the cth column even though the (c+1)th column doesn't exist
        # on the last iteration where c=N. Have I mentioned how much I dislike
        # Python?
        true_X[:,c:c+1] = np.add( temp, true_noise )                   
        time.append(c)
        c += 1

    result = np.zeros([3,N+1])
    sigma_mp = 100.0
    sigma_mv = 10.0
    sigma_ma = 1.0
    
    R = np.matrix( [[sigma_mp**2, 0.0, 0.0],
                    [0.0, sigma_mv**2, 0.0],
                    [0.0, 0.0, sigma_ma**2]] )
                            
    Q1 = np.matrix([[1.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0]]) * sigma_pv**2
                   
    Q2 = np.matrix([[0.25, 1.0, 0.5],
                   [1.0, 1.0, 1.0],
                   [0.5, 1.0, 1.0]]) * sigma_pa**2
    Q = Q1 + Q2

    F = np.matrix( [ [ 1.0, 1.0, 0.5 ], [ 0.0, 1.0, 1.0 ], [ 0.0, 0.0, 1.0 ] ] )

    H = np.matrix( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ] )

    motion = np.matrix( [ 0.0, 0.0, 0.0 ] ).T
    
    # LN - Initialize the first estimate. Note that there's no reason we can't
    # provide a very good initial guess for this. In practice, ascent rate
    # would start at zero since we would start this filter before releasing
    # the balloon.
    x = np.matrix('0. 0.0 0.').T
    result[:,0:1] = x
    
    # LN - Now calculate subsequent estimates based on measurements:
    for i in range(1,N+1):
        meas_noise = np.matrix( [ np.random.normal( 0.0, sigma_mp ),
                                  np.random.normal( 0.0, sigma_mv ),
                                  np.random.normal( 0.0, sigma_ma ) ] ).T
        meas = ( true_X[:,i:i+1] + meas_noise ).T
        x, P = kalman(x, P, meas, R,motion, Q, F, H)
        result[:,i:i+1] = x        
    
    #plot the altitude results
    plt.figure(1)
    plt.plot(time, result[0,:], 'y-', label="True altitude")
    plt.plot(time, true_X[0,:], 'g-', label="Kalman estimate")
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (feet)")
    
    #plot the error
    plt.figure(2)
    plt.plot(time, result[0,:] - true_X[0,:], 'y-', label="Altitude Error")
    #legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude Error (feet)")
    plt.show()

demo_kalman_xy()

#                   # NNP     MD      VB       JJ     NN     RB      DT
initial_probs   =  [0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026]                    

transition_probs = [     # NNP     MD      VB       JJ     NN     RB      DT
                        [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025], # NNP
                        [0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041], # MD
                        [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231], # VB
                        [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036], # JJ
                        [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068], # NN
                        [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479], # RB
                        [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]  # DT
                   ]

observation_prob = [
                        # Janet    # will    # back    # the     # bill
                        [0.000032, 0.000000, 0.000000, 0.000048, 0.000000], # NNP
                        [0.000000, 0.308431, 0.000000, 0.000000, 0.000000], # MD
                        [0.000000, 0.000028, 0.000672, 0.000000, 0.000028], # VB
                        [0.000000, 0.000000, 0.000340, 0.000000, 0.000000], # JJ
                        [0.000000, 0.000200, 0.000223, 0.000000, 0.002337], # NN
                        [0.000000, 0.000000, 0.010446, 0.000000, 0.000000], # RB
                        [0.000000, 0.000000, 0.000000, 0.506099, 0.000000]  # DT
                   ]

states = ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']

def viterbi_edu_not_efficient(transition, observation, init_probs):
    _, T = observation.shape
    _, N = transition.shape 

    # creat a path probability matrix viterbi[N,T]
    V = np.zeros((N,T)) # viterbi matrix
    B = np.zeros((N,T), dtype=np.int) # backtrace pointers

    # initialization step
    print("intialiation step ---------------------------")
    for s in range(N): # 1, ..., N ==> 0, ..., N-1
        V[s][0] = init_probs[s] * observation[s][0] # P(s|<s>)*P(s| Janet ) 
        B[s][0] = 0 # <-- it is not necessary because of np.zeros()
        print("V[{}][{}]={:.6f}*{:.6f}={:.6f}".format(s,0,
                                                      init_probs[s], 
                                                      observation[s][0],
                                                      V[s][0]
                                                      )
              )
    
    print("recursion step ---------------------------")
    # recursion step
    for t in range(1,T):
        print("Time : ", t)
        for s in range(N):
            # prev_viterbi[s] * transition[s' --> s]
            _values = []
            for prev_s in range(N):
                _values.append( V[prev_s][t-1] * transition[prev_s][s] )  # it is not efficient version. (for education)
            
            s_prime = np.argmax(_values)
            max_viterbi_value = _values[s_prime]

            prob = max_viterbi_value * observation[s][t]
            V[s][t] = prob

            print("V[{}][{}]=V[{}][{}]={:.6f}*{:.6f}={:.6f}={}".format(s,t, states[s],t,
                                                      max_viterbi_value, 
                                                      observation[s][t],
                                                      V[s][t],
                                                      V[s][t]
                                                      )
              )

            # store backtrace pointer
            B[s][t] = int(s_prime)

    # termination step
    print("termination step ---------------------------")
    best_last_state = np.argmax(V[:,T-1])
    best_path_prob = V[best_last_state,T-1]
    
    best_path = [best_last_state]
    for t in reversed( range(1, T) ):
        prev_best_state = B[best_last_state][t] 
        best_path.append( prev_best_state)
        best_last_state = prev_best_state

    # best path tag sequence
    best_path = reversed(best_path)
    best_tag_seq = [ states[i] for i in best_path ] 
    
    return best_tag_seq, best_path_prob 

def viterbi_edu_log_not_efficient(transition, observation, init_probs):
    # to avoid underflow, we use 'log'
    _, T = observation.shape
    _, N = transition.shape 

    # creat a path probability matrix viterbi[N,T]
    V = np.zeros((N,T)) # viterbi matrix
    B = np.zeros((N,T), dtype=np.int) # backtrace pointers

    # initialization step
    for s in range(N):
        V[s][0] = np.log(init_probs[s]) + np.log(observation[s][0]) # P(s|<s>)*P(s| Janet )  <-- log probability
        B[s][0] = 0 # <-- it is not necessary because of np.zeros()
    
    # recursion step
    for t in range(1,T):
        for s in range(N):
            # prev_viterbi[s] * transition[s' --> s]
            _values = []
            for prev_s in range(N):
                _values.append( V[prev_s][t-1] + np.log(transition[prev_s][s]) )  # it is not efficient version. (for education)
            
            s_prime = np.argmax(_values)
            max_viterbi_value = _values[s_prime]

            log_prob = max_viterbi_value + np.log( observation[s][t] )
            V[s][t] = log_prob

            # store backtrace pointer
            B[s][t] = int(s_prime)

    # termination step
    best_last_state = np.argmax(V[:,T-1])
    best_log_path_prob = V[best_last_state,T-1]    # log-probability form
    best_path_prob = np.exp( best_log_path_prob )  # probability form

    best_path = [best_last_state]
    for t in reversed( range(1, T) ):
        prev_best_state = B[best_last_state][t] 
        best_path.append( prev_best_state)
        best_last_state = prev_best_state

    # best path tag sequence
    best_path = reversed(best_path)
    best_tag_seq = [ states[i] for i in best_path ] 
    
    return best_tag_seq, best_path_prob 


import numpy as np

trans  = np.array(transition_probs)
obs    = np.array(observation_prob)
inits  = np.array(initial_probs)

best_tag_seq, best_path_prob = viterbi_edu_not_efficient(trans, obs, inits)
print(best_tag_seq, best_path_prob)

#best_tag_seq, best_path_prob = viterbi_edu_log_efficient(trans, obs, inits)
#print(best_tag_seq, best_path_prob)

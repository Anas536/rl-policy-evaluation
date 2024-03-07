# POLICY EVALUATION

## AIM
To evaluate and compare the performance of two policies using policy evaluation.

## PROBLEM STATEMENT

1. Given a set of states, actions, and transition probabilities, we are given two policies.
2. We want to evaluate the performance of the two policies by computing their state-value functions.
3. The policy with the higher state-value function is considered to be the better policy.


## POLICY EVALUATION FUNCTION
```
#Name : Mohamed Anas O.I
#Reg no: 212223110028

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V)):
        break
    return V
```

## OUTPUT:

![Screenshot 2024-03-07 191955](https://github.com/Anas536/rl-policy-evaluation/assets/139841834/1beb78cb-e113-4247-a631-38267018d8e7)


## RESULT:
![Screenshot 2024-03-07 192044](https://github.com/Anas536/rl-policy-evaluation/assets/139841834/c0aedd8e-066f-4015-aea1-9a66e78fac9f)


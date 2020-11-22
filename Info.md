**Info**

DDPG = Deep Deterministic Policy Gradient  
     = Deterministic Policy-Gradient Algorithms + Actor-Critic Methods + Deep Q-Network
     
 Total Discount future reward    
    
    R = r1 + gamma*r2 + gamma²*r3 + ... + gamma^n*rn
    
    
 2 Types of policies:  
 *Deterministic policies:*
    
    a = f(s)
 
 *Stochastic policies:*
 
 
    pi(a|s) = P[a|s]
    
 Policy Objective Function
 
 
    L(theta) = E[r1 + gamma*r2 + gamma²r3 + ... |pi_theta(s,a)]
             = E_x~p(x|theta) [R]   
    
    
Actor -> policy function  


    Critic -> value function
    
    
The osu! display screen can be modelized as a Markov state. It captures all the information necessary and we can forget about the past and thus apply Reinforcement learning algorithm.

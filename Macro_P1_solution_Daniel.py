import matplotlib
import numpy as np
import math as math
import matplotlib.pyplot as plt

# The computation for Gridsize = 1000 takes depending on the available computing power 60 sek and longer.
# If you want this program to compute and plot the numerical solutions for Gridsize = 1000 set the following variable to 'True

do_computation_for_gridsize_1000 = False

def value_function_analytical(A,B, Kapital_Grid):
    v = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,Kapital_Grid.size)):
        v[i] = v_analytical(Kapital_Grid[i],A,B)
    return(v)

def policy_function_analytical(alpha,beta,A,Kapital_Grid):
    v = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,Kapital_Grid.size)):
        v[i] = (Kapital_Grid[i] ** alpha) * beta * alpha * A
    return(v)

def policy_function_from_policy_index(policy_index,Kapital_Grid):
    policy_function = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,policy_function.size)):
        policy_function[i] = Kapital_Grid[int(policy_index[i])]
    return(policy_function)
        

def value_function_iteration(alpha, beta, technology, Kapital_Grid,tolerance):
    difference = 2 * tolerance
    lenght_of_grid = Kapital_Grid.size
    v = np.array([0.]*lenght_of_grid)              # vector in which we safe the current itteration
    v_prime = np.array([0.]*lenght_of_grid)        # vector in which we safe the previous itteration
    Dummy = np.array([0.]*lenght_of_grid)
    policy_index = np.array([0.]*lenght_of_grid)
    while difference > tolerance:                                                             # we iterate as long we are above the tolerance epsilon
        for k in list(range(0,Kapital_Grid.size)):                                          # for each possible kapital value 'Capital(k)' we compute the v_n+1(k) (which is v_prime here)
            for j in list(range(0,Kapital_Grid.size)):                                      # in order to do that we need to find the maximum of the term in the next line
                if(Kapital_Grid[j] < f(Kapital_Grid[k])):                                   # for those 'capital(k)' that are smaller than f(capital(k)) otherwise the log is not defined
                    Dummy[j] = math.log(technology(Kapital_Grid[k])-Kapital_Grid[j]) + beta * v[j]
                else: Dummy[j] = None
            
            v_prime[k] = np.max(Dummy)#These lines are just technical stuff. could probably be done more compactly.
            policy_index[k] = np.argmax(Dummy)
        difference = np.max(np.absolute(v-v_prime))     # We update the difference to check later if we need to stop the loop
    
        for k in list(range(0,v.size)):         # Here we set the values of v equal to the values of v_prime
            v[k] = v_prime[k]                   # v = v_prime does not work because v and v_prime are pointers (I think)
        #l= l +1                                # we do that because we need 'v_n' = v to compute 'v_n+1' = v_prime in the next iteration
    return([v,policy_index])                          # the policy function that is returned here does not give you for capital 'k' capital 'k_prime'
                                                # instead it takes as an input the index of 'k' (in Kapital_Grid) and as output it gives you the index of 'k_prime' 
    
def find_optimal_path(k_0,n,policy_index,Kapital_Grid):   # k_0 must be the index of the start capital!
    path_index = np.array([0.]*n)
    path_index[0] = k_0
    
    for i in list(range(1,n)):                            # We first write the indexes of the optimal path into path_index
        path_index[i]=policy_index[int(path_index[i-1])]
    for i in list(range(1,n)):                            # then we 'translate' the indexes into the corresponding values
        path_index[i] = Kapital_Grid[int(path_index[i])]       
    return(path_index)





def v_analytical(k,A,B):
    return (A + B * math.log(k))



Kapital_Grid = np.arange(2,12,2)         # the discrete grid for capital values with 10 entries
Kapital_Grid_100 = np.arange(2,10,0.08)  # the discrete grid for capital values with 100 entries
Kapital_Grid_1000 = np.arange(2,10,0.008)
epsilon = 10**(-6)           # variables as stated in the exercise
alpha = 0.3
beta = 0.6
A = 20
n = 20 + 1                  # number of steps of optimal kapital path to be plotted
k_0 = 0                     # The index of the initial kapital in the Kapital_Grid, for which optimal path should be determined
alpha_0 = (math.log(A * (1 - alpha * beta))) / (1 - beta) + (beta * alpha * math.log(alpha * beta * A)) / ((1 - beta)*(1 - alpha * beta))
alpha_1 = (alpha) / (1 - alpha * beta)



def f(x):                    # the production function as stated in the exercise
    return A*(x ** alpha)
    
def u(c):                    # the ustility function as stated in the exercise
    return math.log(c)
    
difference = 2 * epsilon     # variable to keep track of distance between v and v_prime

l = 0 # running index

[value_function_n,policy_index_n] = value_function_iteration(alpha,beta,f,Kapital_Grid,epsilon)

value_function_a = value_function_analytical(alpha_0,alpha_1,Kapital_Grid)

policy_function_a = policy_function_analytical(alpha,beta,A,Kapital_Grid)

policy_function_n = policy_function_from_policy_index(policy_index_n,Kapital_Grid)


 



fig, ax = plt.subplots()
ax.plot(Kapital_Grid,value_function_n,'-',color = 'b')
ax.plot(Kapital_Grid,value_function_a,'--',color = 'r')

ax.set(xlabel='capital', ylabel='values',
       title='Gridsize: |entries| = 100 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
ax.grid

fig2, ax2 = plt.subplots()
ax2.plot(Kapital_Grid,policy_function_n,'-',color = 'b')
ax2.plot(Kapital_Grid,policy_function_a,'--',color = 'r')

ax2.set(xlabel='capital', ylabel='policy',
       title='Gridsize: |entries| = 100 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
ax2.grid

[value_function_n_100,policy_index_n_100] = value_function_iteration(alpha,beta,f,Kapital_Grid_100,epsilon)

value_function_a_100 = value_function_analytical(alpha_0,alpha_1,Kapital_Grid_100)

policy_function_a_100 = policy_function_analytical(alpha,beta,A,Kapital_Grid_100)

policy_function_n_100 = policy_function_from_policy_index(policy_index_n_100,Kapital_Grid_100)

optimal_path_100 = find_optimal_path(k_0,n,policy_index_n_100,Kapital_Grid_100)

fig_100, ax_100 = plt.subplots()
ax_100.plot(Kapital_Grid_100,value_function_n_100,'-',color = 'b')
ax_100.plot(Kapital_Grid_100,value_function_a_100,'--',color = 'r')

ax_100.set(xlabel='capital', ylabel='values',
       title='Gridsize: |entries| = 100 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
ax_100.grid

fig2_100, ax2_100 = plt.subplots()
ax2_100.plot(Kapital_Grid_100,policy_function_n_100,'-',color = 'b')
ax2_100.plot(Kapital_Grid_100,policy_function_a_100,'--',color = 'r')

ax2_100.set(xlabel='capital', ylabel='policy',
       title='Gridsize: |entries| = 100 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
ax2_100.grid

if do_computation_for_gridsize_1000:

    [value_function_n_1000,policy_index_n_1000] = value_function_iteration(alpha,beta,f,Kapital_Grid_1000,epsilon)

    value_function_a_1000 = value_function_analytical(alpha_0,alpha_1,Kapital_Grid_1000)

    policy_function_a_1000 = policy_function_analytical(alpha,beta,A,Kapital_Grid_1000)

    policy_function_n_1000 = policy_function_from_policy_index(policy_index_n_1000,Kapital_Grid_1000)

    optimal_path_1000 = find_optimal_path(k_0,n,policy_index_n_1000,Kapital_Grid_1000)

    fig_1000, ax_1000 = plt.subplots()
    ax_1000.plot(Kapital_Grid_1000,value_function_n_1000,'-',color = 'b')
    ax_1000.plot(Kapital_Grid_1000,value_function_a_1000,'--',color = 'r')

    ax_1000.set(xlabel='capital', ylabel='values',
       title='Gridsize: |entries| = 1000 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
    ax_1000.grid

    fig2_1000, ax2_1000 = plt.subplots()
    ax2_1000.plot(Kapital_Grid_1000,policy_function_n_1000,'-',color = 'b')
    ax2_1000.plot(Kapital_Grid_1000,policy_function_a_1000,'--',color = 'r')

    ax2_1000.set(xlabel='capital', ylabel='policy',
       title='Gridsize: |entries| = 1000 \n numerical has style: - , and color: blue \n analytical has style: -- , and color: red')
    ax2_1000.grid

    fig3 , ax3 = plt.subplots()
    ax3.plot(list(range(0,n)),optimal_path_100,'-', color = 'b')  
    ax3.set(ylabel = 'next period capital', xlabel = 'this period capital', title = 'Optimal capital sequence k_t, t \in {0,20}' )

plt.show()



print(value_function_n)
print("Done")
    

    
   
   
        
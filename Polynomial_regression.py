import numpy as np
import copy, math
import matplotlib.pyplot as plt

data = np.loadtxt('./dataset.txt')
# print(data)
features = data.shape[1]-1
# print(features)
x_train = np.array(data[:,:features])
print(x_train.shape)
y_train = np.array(data[:,-1])


def generate_polynomial_features(X, degree):
    from itertools import combinations_with_replacement
    n_samples, n_features = X.shape
    
    def iter_combinations():
        for total_degree in range(1, degree + 1):
            for comb in combinations_with_replacement(range(n_features), total_degree):
                yield comb
    
    comb = list(iter_combinations())
    n_output_features = len(comb)
    X_poly = np.empty((n_samples, n_output_features))
    
    for i, indices in enumerate(comb):
        X_poly[:, i] = np.prod(X[:, indices], axis=1)
    
    return X_poly
x_train_poly = generate_polynomial_features(x_train, 2)


def zscore_normalize_features(X):
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      
    return X_norm
x_norm = zscore_normalize_features(x_train_poly)


def zscore_normalize_target(Y):
    # find the mean of each column/feature
    mu     = np.mean(Y, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(Y, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    Y_norm = (Y - mu) / sigma      
    return Y_norm
# y_norm = zscore_normalize_features(y_train)

# def plot_xvy(x_norm, y_norm,feature):
#     plt.scatter(x_norm[:,feature-1],y_norm)
#     plt.show()
# plot_xvy(x_norm, y_norm ,9)


# # check for pattern in the data
# def show_plt(feature1, feature2):
#     plt.scatter(x_norm[:,feature1-1], x_norm[:,feature2-1])
#     plt.show()
#     return None
# show_plt(1,2)


def compute_cost(x_in, y_in, w_in, b_in):
    f_wb=np.dot(x_in,w_in)+b_in 
    cost = (f_wb -y_in)**2
    cost = np.sum(cost)/(2*np.shape(x_in)[0])
    return cost

def compute_gradient(x_in, y_in, w_in, b_in):
    m=x_in.shape[0]
    pred = np.dot(x_in, w_in) + b_in
    costs = pred - y_in
    dj_dw = np.dot(x_in.T, costs)/m
    dj_db = np.sum(costs)/m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradent_function, alpha, iters):
    J_history=[]
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iters):
        dj_dw,dj_db = gradent_function(x, y, w, b)

        w = w - alpha *dj_dw
        b = b - alpha *dj_db

        if i<10000:
            J_history.append(cost_function(x, y, w, b))

        if i%math.ceil(iters /10) ==0:
            print(f"iteration {i:4d}: Cost {J_history[-1]:8.2f}")
    return w,b,J_history


initial_w = np.zeros(x_norm.shape[1])
initial_b = 0
iterations = 90000
alpha = 4.66e-2
w_final, b_final, J_hist = gradient_descent(x_norm, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_norm[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()



y_hat=np.dot(x_norm,w_final)+ b_final
fig,ax=plt.subplots(3,3,figsize=(20,5),sharey=True)
print(ax.shape)
for i in range(ax.shape[1]):
    for j in range(ax.shape[0]):
        ax[i,j].scatter(x_train[:,i],y_train, label = 'target', c='r')
        # ax[i].set_xlabel(X_features[i])
        ax[i,j].scatter(x_train[:,i],y_hat,c='b', label = 'predict')
# ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

data = np.loadtxt('./dataset.txt')
# print(data)
features = data.shape[1]-1
# print(features)
X_train = np.array(data[:,:features])
print(X_train.shape)
y_train = np.array(data[:,-1])


PolyF = PolynomialFeatures(degree = 2)
X_PolyF = PolyF.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_PolyF,axis=0)}")

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_PolyF)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_PolyF,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

lmr = LinearRegression()
lmr.fit(X_norm, y_train)
print(lmr)
# print(f"number of iterations completed: {lr.n_iter_}, number of weight updates: {lr.t_}")

b_norm = lmr.intercept_
w_norm = lmr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# make a prediction using sgdr.predict()
y_pred_sgd = lmr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")


# plt.scatter(X_train,y_train, label = 'target', c='b')
#     # ax[i].set_xlabel(X_features[i])
# plt.scatter(X_train,y_pred,c='r', label = 'predict')
# # ax[0].set_ylabel("Price"); ax[0].legend();
# # fig.suptitle("target versus prediction using z-score normalized model")
# plt.show()

fig, ax = plt.subplots(3, 3, figsize=(20, 5), sharey=True)

# Assuming the number of features is less than or equal to 9 for this grid
for i in range(3):  # Rows
    for j in range(3):  # Columns
        feature_index = i * 3 + j
        if feature_index < features:
            ax[i, j].scatter(X_train[:, feature_index], y_train, label='Target', c='r')
            ax[i, j].scatter(X_train[:, feature_index], y_pred, label='Predict', c='b')
            ax[i, j].set_title(f"Feature {feature_index + 1}")
            ax[i, j].legend()

plt.show()
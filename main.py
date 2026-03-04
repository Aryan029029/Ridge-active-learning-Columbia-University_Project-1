import pandas as pd
import numpy as np

X_train = pd.read_csv("X_train.csv", header=None).values
y_train = pd.read_csv("y_train.csv", header=None).values
X_test = pd.read_csv("X_test.csv", header=None).values

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

def ridge_regression(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    
    # Compute weights using ridge formula
    w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    
    return w

lam = 1.0   # use whatever lambda your assignment specifies

w = ridge_regression(X_train, y_train, lam)

print("Weights shape:", w.shape)
print("Weights:\n", w)

y_pred = X_test @ w

print("Predictions shape:", y_pred.shape)
print("First 5 predictions:\n", y_pred[:5])

def compute_uncertainty(X_current, X_pool, lam):
    A = X_current.T @ X_current + lam * np.eye(X_current.shape[1])
    A_inv = np.linalg.inv(A)
    
    uncertainties = []
    
    for i in range(X_pool.shape[0]):
        x = X_pool[i].reshape(-1, 1)
        variance = x.T @ A_inv @ x
        uncertainties.append(variance.item())
    
    return np.array(uncertainties)

uncertainties = compute_uncertainty(X_train, X_test, lam)

print("Uncertainty shape:", uncertainties.shape)
print("First 5 uncertainty values:", uncertainties[:5])

first_index = np.argmax(uncertainties)

print("First selected index:", first_index)
print("Uncertainty value:", uncertainties[first_index])

def active_learning(X_train, y_train, X_pool, lam, n_select=10):
    
    selected_indices = []
    
    X_current = X_train.copy()
    y_current = y_train.copy()
    
    X_pool_current = X_pool.copy()
    
    # Track original indices
    pool_indices = np.arange(X_pool.shape[0])
    
    for _ in range(n_select):
        
        A = X_current.T @ X_current + lam * np.eye(X_current.shape[1])
        A_inv = np.linalg.inv(A)
        
        uncertainties = []
        
        for i in range(X_pool_current.shape[0]):
            x = X_pool_current[i].reshape(-1, 1)
            variance = x.T @ A_inv @ x
            uncertainties.append(variance.item())
        
        uncertainties = np.array(uncertainties)
        
        idx = np.argmax(uncertainties)
        
        # Store ORIGINAL index
        selected_indices.append(int(pool_indices[idx]))
        
        # Add selected point to training
        X_selected = X_pool_current[idx].reshape(1, -1)
        X_current = np.vstack([X_current, X_selected])
        y_current = np.vstack([y_current, [[0]]])
        
        # Remove from pool AND index tracker
        X_pool_current = np.delete(X_pool_current, idx, axis=0)
        pool_indices = np.delete(pool_indices, idx)
    
    return selected_indices

selected_points = active_learning(X_train, y_train, X_test, lam, n_select=10)

print("Selected indices:", selected_points)
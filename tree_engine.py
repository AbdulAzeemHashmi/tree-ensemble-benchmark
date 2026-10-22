"""
tree_engine.py
Modular implementation of Tree-based algorithms from scratch for ML Assignment 2.
Student: Abdul Azeem (24i-2013), Section: BAI-5A
Roll parameters:
  roll = 2013
  alpha = 1.3
  b = 3
  lam = 0.06
  q = 0.13
  stop_eps = 0.15
  sigma = 0.04
  seed = 2013
"""

import numpy as np
import pandas as pd
import time

ROLL = 2013
ALPHA = 1 + (3 / 10)         # 1.3
BASE_B = 2 + (1 % 3)         # 3
LAMBDA_PENALTY = 0.01 * (2 + 0 + 1 + 3) # 0.06
Q_QUANTILE = 0.13
STOP_EPS = 0.05 * max(1, 2013 % 10)     # 0.15
SIGMA_NOISE = 0.01 * max(1, 2013 % 7)   # 0.04
BASE_SEED = 2013

EPS = 1e-12

def get_rng(offset=0):
    """Create reproducible random number generator using seed + offset."""
    return np.random.default_rng(BASE_SEED + offset)

def manual_stratified_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed_offset=0):
    """
    Perform manual stratified train/val/test split using NumPy only.
    Guarantees balanced representation of each class across splits.
    """
    rng = get_rng(seed_offset)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    
    for c in classes:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        n_total = len(c_indices)
        n_train = int(round(train_ratio * n_total))
        n_val = int(round(val_ratio * n_total))
        
        train_idx.extend(c_indices[:n_train])
        val_idx.extend(c_indices[n_train:n_train + n_val])
        test_idx.extend(c_indices[n_train + n_val:])
        
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    # Shuffle splits so classes are not contiguous
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def standardize_features(X_train, X_val, X_test):
    """
    Standardize features using training-set statistics only to prevent data leakage.
    Returns (X_train_std, X_val_std, X_test_std, mean, std).
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std_adj = np.where(std == 0, 1.0, std)
    
    X_train_std = (X_train - mean) / std_adj
    X_val_std = (X_val - mean) / std_adj
    X_test_std = (X_test - mean) / std_adj
    
    return X_train_std, X_val_std, X_test_std, mean, std_adj

def pca_svd(X, n_components=2):
    """
    Compute PCA projection using np.linalg.svd from scratch.
    """
    X_centered = X - np.mean(X, axis=0)
    # SVD: X = U S Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T
    X_pca = X_centered @ V[:, :n_components]
    explained_variance = (S ** 2) / (len(X) - 1)
    explained_variance_ratio = explained_variance[:n_components] / np.sum(explained_variance)
    return X_pca, explained_variance_ratio

def entropy(y, roll_number=ROLL, sample_weight=None, num_classes=None):
    """
    Compute roll-based entropy:
    H(p) = - sum_i ( p_i^alpha * log_b(p_i + eps) ), eps = 1e-12.
    Supports optional sample weights.
    """
    if len(y) == 0:
        return 0.0
    alpha = 1 + ((roll_number % 10) / 10)
    b = 2 + (((roll_number // 10) % 10) % 3)
    
    if num_classes is None:
        classes = np.unique(y)
    else:
        classes = np.arange(num_classes)
        
    if sample_weight is None:
        total_w = float(len(y))
        counts = np.bincount(y, minlength=10 if num_classes is None else num_classes)
        probs = counts / total_w
    else:
        total_w = np.sum(sample_weight)
        if total_w <= 0:
            return 0.0
        counts = np.bincount(y, weights=sample_weight, minlength=10 if num_classes is None else num_classes)
        probs = counts / total_w
        
    # Filter non-zero probabilities for numerical stability
    p = probs[probs > 0]
    if len(p) <= 1:
        return 0.0
        
    log_b = np.log(p + EPS) / np.log(b)
    h = -np.sum((p ** alpha) * log_b)
    return float(max(0.0, h))

def rho_feature(X, y, num_classes=10):
    """
    Compute rho(f) as the mean absolute Pearson correlation between feature f 
    and the 10 one-hot class indicators.
    Returns 1D array of length d (number of features).
    """
    n_samples, n_features = X.shape
    # One-hot encode y
    Y = np.zeros((n_samples, num_classes), dtype=np.float64)
    Y[np.arange(n_samples), y] = 1.0
    
    # Center X and Y
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    std_X = np.std(X, axis=0)
    std_X = np.where(std_X == 0, 1.0, std_X)
    
    std_Y = np.std(Y, axis=0)
    std_Y = np.where(std_Y == 0, 1.0, std_Y)
    
    # Correlation matrix: (n_features, num_classes)
    # R[f, k] = cov(X_f, Y_k) / (std_X_f * std_Y_k)
    cov = (X_c.T @ Y_c) / n_samples
    R = cov / (std_X[:, None] * std_Y[None, :])
    
    # Mean absolute correlation across all 10 classes
    rho = np.mean(np.abs(R), axis=1)
    return rho

def information_gain(parent_h, y_left, y_right, w_left=None, w_right=None, 
                     rho_val=0.0, lam=LAMBDA_PENALTY, roll_number=ROLL, num_classes=10):
    """
    Compute penalized information gain:
    IG_roll(f, t) = IG(f, t) - lam * rho(f)
    """
    n_l = len(y_left) if w_left is None else np.sum(w_left)
    n_r = len(y_right) if w_right is None else np.sum(w_right)
    total_w = n_l + n_r
    if total_w == 0 or n_l == 0 or n_r == 0:
        return -1e9, 0.0
        
    h_l = entropy(y_left, roll_number=roll_number, sample_weight=w_left, num_classes=num_classes)
    h_r = entropy(y_right, roll_number=roll_number, sample_weight=w_right, num_classes=num_classes)
    
    child_h = (n_l / total_w) * h_l + (n_r / total_w) * h_r
    raw_ig = parent_h - child_h
    penalized_ig = raw_ig - lam * rho_val
    return penalized_ig, raw_ig

def best_split_quantile(X, y, q=Q_QUANTILE, lam=LAMBDA_PENALTY, rho_arr=None, 
                        roll_number=ROLL, sample_weight=None, feature_subset=None, num_classes=10):
    """
    Quantile-based split: the candidate threshold for each feature is its q-quantile 
    computed on node samples.
    """
    n_samples, n_features = X.shape
    if rho_arr is None:
        rho_arr = rho_feature(X, y, num_classes=num_classes)
        
    if feature_subset is None:
        feature_subset = np.arange(n_features)
        
    parent_h = entropy(y, roll_number=roll_number, sample_weight=sample_weight, num_classes=num_classes)
    best_gain = -1e9
    best_raw_gain = -1e9
    best_feat = -1
    best_thresh = None
    
    for f in feature_subset:
        feat_vals = X[:, f]
        thresh = float(np.quantile(feat_vals, q))
        left_mask = feat_vals <= thresh
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
            
        y_l, y_r = y[left_mask], y[right_mask]
        w_l = None if sample_weight is None else sample_weight[left_mask]
        w_r = None if sample_weight is None else sample_weight[right_mask]
        
        pen_gain, raw_gain = information_gain(
            parent_h, y_l, y_r, w_l, w_r, rho_val=rho_arr[f], lam=lam, 
            roll_number=roll_number, num_classes=num_classes
        )
        
        if pen_gain > best_gain:
            best_gain = pen_gain
            best_raw_gain = raw_gain
            best_feat = f
            best_thresh = thresh
            
    return best_feat, best_thresh, best_gain, best_raw_gain

def best_split_exhaustive(X, y, lam=LAMBDA_PENALTY, rho_arr=None, roll_number=ROLL, 
                          sample_weight=None, feature_subset=None, min_samples_leaf=1, num_classes=10):
    """
    Vectorized exhaustive split search:
    Sorts each feature once and evaluates thresholds using cumulative class statistics.
    Returns (best_feat, best_thresh, best_gain, best_raw_gain).
    """
    n_samples, n_features = X.shape
    if rho_arr is None:
        rho_arr = rho_feature(X, y, num_classes=num_classes)
    if feature_subset is None:
        feature_subset = np.arange(n_features)
        
    alpha = 1 + ((roll_number % 10) / 10)
    b = 2 + (((roll_number // 10) % 10) % 3)
    log_b_const = np.log(b)
    
    # Overall parent class counts / weights
    if sample_weight is None:
        total_w = float(n_samples)
        total_counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        sample_w = np.ones(n_samples, dtype=np.float64)
    else:
        sample_w = sample_weight.astype(np.float64)
        total_w = float(np.sum(sample_w))
        total_counts = np.bincount(y, weights=sample_w, minlength=num_classes).astype(np.float64)
        
    parent_probs = total_counts / total_w
    p_nz = parent_probs[parent_probs > 0]
    parent_h = float(-np.sum((p_nz ** alpha) * (np.log(p_nz + EPS) / log_b_const))) if len(p_nz) > 1 else 0.0
    
    best_gain = -1e9
    best_raw_gain = -1e9
    best_feat = -1
    best_thresh = None
    
    for f in feature_subset:
        feat_vals = X[:, f]
        sort_idx = np.argsort(feat_vals)
        x_sorted = feat_vals[sort_idx]
        y_sorted = y[sort_idx]
        w_sorted = sample_w[sort_idx]
        
        # Valid split positions: where feature value changes
        split_mask = x_sorted[:-1] != x_sorted[1:]
        if not np.any(split_mask):
            continue
            
        split_pos = np.where(split_mask)[0]
        # Leaf constraints
        if min_samples_leaf > 1:
            split_pos = split_pos[(split_pos >= min_samples_leaf - 1) & 
                                  (split_pos < n_samples - min_samples_leaf)]
            if len(split_pos) == 0:
                continue
                
        # One-hot weighted matrix for cumulative sum
        W_onehot = np.zeros((n_samples, num_classes), dtype=np.float64)
        W_onehot[np.arange(n_samples), y_sorted] = w_sorted
        
        cum_counts_L = np.cumsum(W_onehot, axis=0)[split_pos]
        cum_w_L = np.cumsum(w_sorted)[split_pos]
        
        cum_counts_R = total_counts - cum_counts_L
        cum_w_R = total_w - cum_w_L
        
        # Calculate entropy of left partitions
        P_L = cum_counts_L / cum_w_L[:, None]
        # P_L^alpha * log_b(P_L + eps)
        # Vectorized safe entropy computation
        log_P_L = np.log(P_L + EPS) / log_b_const
        H_L = -np.sum(np.where(P_L > 0, (P_L ** alpha) * log_P_L, 0.0), axis=1)
        H_L = np.maximum(0.0, H_L)
        
        # Calculate entropy of right partitions
        P_R = cum_counts_R / cum_w_R[:, None]
        log_P_R = np.log(P_R + EPS) / log_b_const
        H_R = -np.sum(np.where(P_R > 0, (P_R ** alpha) * log_P_R, 0.0), axis=1)
        H_R = np.maximum(0.0, H_R)
        
        # Weighted child entropy
        child_H = (cum_w_L / total_w) * H_L + (cum_w_R / total_w) * H_R
        raw_gains = parent_h - child_H
        penalized_gains = raw_gains - lam * rho_arr[f]
        
        max_idx = np.argmax(penalized_gains)
        feat_best_gain = penalized_gains[max_idx]
        
        if feat_best_gain > best_gain:
            best_gain = feat_best_gain
            best_raw_gain = raw_gains[max_idx]
            best_feat = f
            best_split_idx = split_pos[max_idx]
            best_thresh = (x_sorted[best_split_idx] + x_sorted[best_split_idx + 1]) / 2.0
            
    return best_feat, best_thresh, best_gain, best_raw_gain

class TreeNode:
    """Node in Decision Tree storing structure, splits, and local statistics."""
    def __init__(self, depth=0):
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.gain = 0.0
        self.raw_gain = 0.0
        self.n_samples = 0
        self.weighted_n = 0.0
        self.class_counts = None
        self.class_probs = None
        self.prediction = None
        self.left = None
        self.right = None
        self.is_leaf = False

def build_tree(X, y, depth=0, max_depth=None, roll_number=ROLL, sample_weight=None,
               max_features=None, min_samples_leaf=1, stop_eps=STOP_EPS, lam=LAMBDA_PENALTY,
               rho_arr=None, split_strategy='exhaustive', q=Q_QUANTILE, rng=None, num_classes=10):
    """
    Recursively build decision tree with stopping criteria and stored node info.
    """
    node = TreeNode(depth=depth)
    node.n_samples = len(y)
    
    if sample_weight is None:
        node.weighted_n = float(node.n_samples)
        counts = np.bincount(y, minlength=num_classes)
    else:
        node.weighted_n = float(np.sum(sample_weight))
        counts = np.bincount(y, weights=sample_weight, minlength=num_classes)
        
    node.class_counts = counts
    total_w = node.weighted_n
    node.class_probs = counts / (total_w if total_w > 0 else 1.0)
    node.prediction = int(np.argmax(counts))
    node.impurity = entropy(y, roll_number=roll_number, sample_weight=sample_weight, num_classes=num_classes)
    
    # Check stopping conditions:
    # 1. Pure node
    if node.impurity <= 0.0 or len(np.unique(y)) <= 1:
        node.is_leaf = True
        return node
        
    # 2. Max depth reached
    if max_depth is not None and depth >= max_depth:
        node.is_leaf = True
        return node
        
    # 3. Insufficient samples to split
    if node.n_samples < 2 * min_samples_leaf:
        node.is_leaf = True
        return node
        
    # Select feature subset (for Random Forest / max_features)
    n_total_feats = X.shape[1]
    if max_features is None:
        feat_subset = np.arange(n_total_feats)
    elif max_features == 'sqrt':
        k = max(1, int(np.round(np.sqrt(n_total_feats))))
        if rng is None: rng = get_rng()
        feat_subset = rng.choice(n_total_feats, size=k, replace=False)
    elif max_features == 'log2':
        k = max(1, int(np.round(np.log2(n_total_feats))))
        if rng is None: rng = get_rng()
        feat_subset = rng.choice(n_total_feats, size=k, replace=False)
    elif isinstance(max_features, (int, np.integer)):
        k = min(n_total_feats, max(1, int(max_features)))
        if rng is None: rng = get_rng()
        feat_subset = rng.choice(n_total_feats, size=k, replace=False)
    elif isinstance(max_features, float):
        k = min(n_total_feats, max(1, int(np.round(max_features * n_total_feats))))
        if rng is None: rng = get_rng()
        feat_subset = rng.choice(n_total_feats, size=k, replace=False)
    else:
        feat_subset = np.arange(n_total_feats)
        
    # Find best split
    if split_strategy == 'quantile':
        feat, thresh, gain, raw_gain = best_split_quantile(
            X, y, q=q, lam=lam, rho_arr=rho_arr, roll_number=roll_number,
            sample_weight=sample_weight, feature_subset=feat_subset, num_classes=num_classes
        )
    else:
        feat, thresh, gain, raw_gain = best_split_exhaustive(
            X, y, lam=lam, rho_arr=rho_arr, roll_number=roll_number,
            sample_weight=sample_weight, feature_subset=feat_subset,
            min_samples_leaf=min_samples_leaf, num_classes=num_classes
        )
        
    # Stopping condition: information gain < stop_eps
    if gain < stop_eps or feat == -1 or thresh is None:
        node.is_leaf = True
        return node
        
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    
    if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
        node.is_leaf = True
        return node
        
    node.feature = feat
    node.threshold = thresh
    node.gain = gain
    node.raw_gain = raw_gain
    
    # Recurse left and right
    w_left = None if sample_weight is None else sample_weight[left_mask]
    w_right = None if sample_weight is None else sample_weight[right_mask]
    
    node.left = build_tree(
        X[left_mask], y[left_mask], depth=depth + 1, max_depth=max_depth,
        roll_number=roll_number, sample_weight=w_left, max_features=max_features,
        min_samples_leaf=min_samples_leaf, stop_eps=stop_eps, lam=lam,
        rho_arr=rho_arr, split_strategy=split_strategy, q=q, rng=rng, num_classes=num_classes
    )
    
    node.right = build_tree(
        X[right_mask], y[right_mask], depth=depth + 1, max_depth=max_depth,
        roll_number=roll_number, sample_weight=w_right, max_features=max_features,
        min_samples_leaf=min_samples_leaf, stop_eps=stop_eps, lam=lam,
        rho_arr=rho_arr, split_strategy=split_strategy, q=q, rng=rng, num_classes=num_classes
    )
    
    return node

def predict_one(node, x):
    """Predict class label for a single observation."""
    curr = node
    while not curr.is_leaf:
        if x[curr.feature] <= curr.threshold:
            curr = curr.left
        else:
            curr = curr.right
    return curr.prediction

def predict_proba_one(node, x):
    """Predict class probability distribution for a single observation."""
    curr = node
    while not curr.is_leaf:
        if x[curr.feature] <= curr.threshold:
            curr = curr.left
        else:
            curr = curr.right
    return curr.class_probs

def predict(node, X):
    """Predict class labels for an array of observations."""
    return np.array([predict_one(node, x) for x in X])

def predict_proba(node, X):
    """Predict class probability distribution for an array of observations."""
    return np.array([predict_proba_one(node, x) for x in X])

def accuracy(y_true, y_pred):
    """Compute accuracy score."""
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix: row=true, col=predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_metrics_manual(y_true, y_pred, num_classes=10):
    """
    Compute per-class Precision, Recall, F1, and Macro-F1 manually.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f1s = np.zeros(num_classes)
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions[c] = prec
        recalls[c] = rec
        f1s[c] = f1
        
    macro_f1 = float(np.mean(f1s))
    acc = accuracy(y_true, y_pred)
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'precision_per_class': precisions,
        'recall_per_class': recalls,
        'f1_per_class': f1s,
        'confusion_matrix': cm
    }

def inject_noise(X, roll_number=ROLL, sigma=None, rng=None):
    """
    Inject Gaussian noise X' = X + N(0, sigma^2) reproducibly.
    Default sigma = 0.01 * max(1, roll % 7).
    """
    if sigma is None:
        sigma = 0.01 * max(1, roll_number % 7)
    if rng is None:
        rng = get_rng(100)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

def count_leaves(node):
    """Count number of terminal leaf nodes."""
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

def get_tree_depth(node):
    """Compute maximum depth of the tree."""
    if node.is_leaf:
        return node.depth
    return max(get_tree_depth(node.left), get_tree_depth(node.right))

def prune_tree(tree, X_val, y_val):
    """
    Perform reduced-error pruning on the validation set.
    Greedily collapses subtrees into leaves if validation accuracy does not decrease.
    Returns (pruned_tree, leaf_history, val_acc_history).
    """
    import copy
    current_tree = copy.deepcopy(tree)
    best_acc = accuracy(y_val, predict(current_tree, X_val))
    leaf_history = [count_leaves(current_tree)]
    val_acc_history = [best_acc]
    
    improved = True
    while improved:
        improved = False
        candidates = []
        
        def find_collapsible_nodes(node):
            if node.is_leaf:
                return
            if node.left.is_leaf and node.right.is_leaf:
                candidates.append(node)
            else:
                find_collapsible_nodes(node.left)
                find_collapsible_nodes(node.right)
                
        find_collapsible_nodes(current_tree)
        if not candidates:
            break
            
        best_candidate = None
        best_cand_acc = best_acc
        
        for cand in candidates:
            cand.is_leaf = True
            acc_after = accuracy(y_val, predict(current_tree, X_val))
            cand.is_leaf = False
            
            # Prune if accuracy improves or remains identical (smaller tree preferred)
            if acc_after >= best_cand_acc:
                best_cand_acc = acc_after
                best_candidate = cand
                
        if best_candidate is not None and best_cand_acc >= best_acc:
            best_candidate.is_leaf = True
            best_candidate.left = None
            best_candidate.right = None
            best_acc = best_cand_acc
            improved = True
            leaf_history.append(count_leaves(current_tree))
            val_acc_history.append(best_acc)
            
    return current_tree, leaf_history, val_acc_history

def feature_importance(tree, n_features=216):
    """
    Compute feature importance from scratch using total weighted impurity decrease.
    """
    importances = np.zeros(n_features, dtype=np.float64)
    
    def traverse(node):
        if node.is_leaf:
            return
        w_parent = node.weighted_n
        w_l = node.left.weighted_n
        w_r = node.right.weighted_n
        
        imp_parent = node.impurity
        imp_l = node.left.impurity
        imp_r = node.right.impurity
        
        # Weighted impurity decrease
        decrease = (w_parent * imp_parent) - (w_l * imp_l + w_r * imp_r)
        if decrease > 0:
            importances[node.feature] += decrease
            
        traverse(node.left)
        traverse(node.right)
        
    traverse(tree)
    tot = np.sum(importances)
    if tot > 0:
        importances /= tot
    return importances

def get_decision_path(node, x):
    """Trace and return the decision path for a single observation."""
    path = []
    curr = node
    while not curr.is_leaf:
        val = x[curr.feature]
        branch = "Left (<=)" if val <= curr.threshold else "Right (>)"
        path.append({
            'depth': curr.depth,
            'feature': curr.feature,
            'threshold': curr.threshold,
            'val': val,
            'branch': branch,
            'impurity': curr.impurity,
            'samples': curr.n_samples
        })
        if val <= curr.threshold:
            curr = curr.left
        else:
            curr = curr.right
    return path, curr.prediction, curr.class_probs

# ----------------- Part B: Random Forest -----------------

def bootstrap_sample(X, y, sample_weight=None, rng=None):
    """Draw a bootstrap sample of size N with replacement."""
    if rng is None:
        rng = get_rng()
    n = len(y)
    indices = rng.choice(n, size=n, replace=True)
    w_boot = None if sample_weight is None else sample_weight[indices]
    return X[indices], y[indices], indices, w_boot

class RandomForest:
    """Random Forest Classifier implemented from scratch."""
    def __init__(self, n_trees=50, max_depth=None, max_features='sqrt',
                 stop_eps=STOP_EPS, lam=LAMBDA_PENALTY, min_samples_leaf=1,
                 roll_number=ROLL, seed_offset=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.stop_eps = stop_eps
        self.lam = lam
        self.min_samples_leaf = min_samples_leaf
        self.roll_number = roll_number
        self.seed_offset = seed_offset
        self.trees = []
        self.oob_indices_list = []
        
    def fit(self, X, y, sample_weight=None, num_classes=10):
        self.trees = []
        self.oob_indices_list = []
        n_samples = len(y)
        rho_arr = rho_feature(X, y, num_classes=num_classes)
        
        for i in range(self.n_trees):
            rng = get_rng(self.seed_offset + i * 10)
            X_b, y_b, boot_idx, w_b = bootstrap_sample(X, y, sample_weight=sample_weight, rng=rng)
            
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_indices = np.where(oob_mask)[0]
            self.oob_indices_list.append(oob_indices)
            
            tree_rng = get_rng(self.seed_offset + i * 10 + 1)
            tree = build_tree(
                X_b, y_b, depth=0, max_depth=self.max_depth, roll_number=self.roll_number,
                sample_weight=w_b, max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf, stop_eps=self.stop_eps,
                lam=self.lam, rho_arr=rho_arr, rng=tree_rng, num_classes=num_classes
            )
            self.trees.append(tree)
        return self
        
    def predict_majority(self, X):
        """Predict class via majority voting across trees."""
        tree_preds = np.array([predict(t, X) for t in self.trees]) # (n_trees, n_samples)
        # Mode per sample
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=np.int64)
        for j in range(n_samples):
            counts = np.bincount(tree_preds[:, j], minlength=10)
            final_preds[j] = np.argmax(counts)
        return final_preds
        
    def predict_proba(self, X):
        """Predict class probabilities via averaging leaf probability distributions."""
        all_probs = np.array([predict_proba(t, X) for t in self.trees]) # (n_trees, n_samples, n_classes)
        return np.mean(all_probs, axis=0)
        
    def predict_proba_voting(self, X):
        """Predict class via probability averaging."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
    def compute_oob_accuracy(self, X, y, num_classes=10):
        """Compute Out-Of-Bag accuracy."""
        n_samples = len(y)
        oob_votes = np.zeros((n_samples, num_classes), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int64)
        
        for tree, oob_idx in zip(self.trees, self.oob_indices_list):
            if len(oob_idx) > 0:
                preds = predict(tree, X[oob_idx])
                for idx, p in zip(oob_idx, preds):
                    oob_votes[idx, p] += 1
                    oob_counts[idx] += 1
                    
        valid = oob_counts > 0
        if np.sum(valid) == 0:
            return 0.0
        final_oob_preds = np.argmax(oob_votes[valid], axis=1)
        return float(np.mean(final_oob_preds == y[valid]))

def permutation_importance(model, X_val, y_val, metric_fn=accuracy, rng=None, n_repeats=3):
    """
    Compute permutation feature importance on validation set.
    """
    if rng is None:
        rng = get_rng(300)
    baseline_score = metric_fn(y_val, model.predict_proba_voting(X_val) if hasattr(model, 'predict_proba_voting') else predict(model, X_val))
    n_features = X_val.shape[1]
    importances = np.zeros(n_features)
    
    for f in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_indices = rng.permutation(len(X_val))
            X_perm[:, f] = X_perm[perm_indices, f]
            
            if hasattr(model, 'predict_proba_voting'):
                pred = model.predict_proba_voting(X_perm)
            else:
                pred = predict(model, X_perm)
            scores.append(metric_fn(y_val, pred))
        # Drop in score is the feature importance
        importances[f] = baseline_score - np.mean(scores)
        
    return importances

# ----------------- Part C: AdaBoost SAMME Multiclass -----------------

class AdaBoostSAMME:
    """AdaBoost.M2 / SAMME multiclass boosting from scratch."""
    def __init__(self, n_estimators=50, max_depth=1, roll_number=ROLL, seed_offset=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.roll_number = roll_number
        self.seed_offset = seed_offset
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        self.sample_weight_history = {}
        
    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=10, num_classes=10):
        n_samples = len(y)
        w = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
        K = float(num_classes)
        rho_arr = rho_feature(X, y, num_classes=num_classes)
        
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        self.sample_weight_history = {1: w.copy()}
        
        best_val_err = 1e9
        patience_counter = 0
        
        for m in range(self.n_estimators):
            round_rng = get_rng(self.seed_offset + m * 5)
            # Train weak learner with current sample weights
            tree = build_tree(
                X, y, depth=0, max_depth=self.max_depth, roll_number=self.roll_number,
                sample_weight=w, max_features=None, min_samples_leaf=1, stop_eps=0.0, # pure stump allowed
                lam=0.0, rho_arr=rho_arr, rng=round_rng, num_classes=num_classes
            )
            
            preds = predict(tree, X)
            incorrect = (preds != y)
            err = np.sum(w * incorrect) / np.sum(w)
            
            # Bound error to prevent division by zero or negative weights
            err = max(1e-10, min(err, 1.0 - 1e-10))
            self.estimator_errors.append(err)
            
            # SAMME weight update:
            # alpha = ln((1 - err) / err) + ln(K - 1)
            alpha_m = float(np.log((1.0 - err) / err) + np.log(K - 1.0))
            if alpha_m < 0:
                # Learner is worse than random guessing for SAMME
                alpha_m = 0.0
                
            self.estimators.append(tree)
            self.estimator_weights.append(alpha_m)
            
            # Update sample weights
            w = w * np.exp(alpha_m * incorrect.astype(np.float64))
            w = w / np.sum(w)
            
            # Store history at key rounds
            round_num = m + 1
            if round_num in [1, 10, 50, 100, 200]:
                self.sample_weight_history[round_num] = w.copy()
                
            # Bonus C4: Early stopping check
            if early_stopping and X_val is not None and y_val is not None:
                val_preds = self._predict_rounds(X_val, num_rounds=round_num, num_classes=num_classes)
                val_err = 1.0 - accuracy(y_val, val_preds)
                if val_err < best_val_err:
                    best_val_err = val_err
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    # Early stop triggered
                    break
                    
        return self
        
    def _predict_rounds(self, X, num_rounds=None, num_classes=10):
        if num_rounds is None:
            num_rounds = len(self.estimators)
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, num_classes), dtype=np.float64)
        
        for m in range(num_rounds):
            tree = self.estimators[m]
            alpha = self.estimator_weights[m]
            preds = predict(tree, X)
            for i, p in enumerate(preds):
                scores[i, p] += alpha
                
        return np.argmax(scores, axis=1)
        
    def predict(self, X, num_classes=10):
        return self._predict_rounds(X, num_rounds=len(self.estimators), num_classes=num_classes)

print('tree_engine module compiled successfully!')

import sys
import numpy as np
import scipy.stats as sps
from sklearn.linear_model import TheilSenRegressor

# for fit_ushape_robust ---- 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# for fit_ushape_robust ---- 

def sample_one_per_group(data, labels, rng):
    """
    Generalizable sampler: Picks one random index for each unique label.
    
    Parameters:
    - data: The object to sample from (DataFrame, Numpy array, or List).
    - labels: An array/list of group identifiers corresponding to the rows in data.
    - rng: A numpy random generator (e.g., np.random.default_rng(seed)).
    
    Returns:
    - The sampled subset of the original data.
    """
    # 1. Convert labels to a numpy array for efficient processing (deep copy)
    labels = np.array(labels)
    n_rows = len(labels)
    
    # 2. Create an array of indices [0, 1, 2, ..., n-1] and shuffle them
    indices = np.arange(n_rows)
    shuffled_indices = rng.permutation(indices)
    
    # 3. Reorder labels according to the shuffle
    shuffled_labels = labels[shuffled_indices]
    
    # 4. Find the FIRST occurrence of each unique label in the shuffled list
    # np.unique with return_index returns the first index encountered for each group
    _, first_occurrence_indices = np.unique(shuffled_labels, return_index=True)
    
    # 5. Map these back to the original row positions
    selected_rows = shuffled_indices[first_occurrence_indices]
    rng.shuffle(selected_rows)
    #input(selected_rows)
    
    # 6. Return the data using the appropriate slicing method
    if hasattr(data, 'iloc'):  # It's a Pandas DataFrame/Series
        return data.iloc[selected_rows]
    elif hasattr(data, '__getitem__'): # It's a Numpy array or List
        if isinstance(data, np.ndarray):
            return data[selected_rows]
        return [data[i] for i in selected_rows]
    else:
        raise ValueError("Data format not supported for indexing.")

def bstrp_cvg(orgests, truep, bs_ests, lvl = 0.975, type = '-orgest'):
    if type == '-orgest':
        bs_std = np.sqrt(np.square(bs_ests - orgests).mean(axis = 0))
        bs_half = sps.norm.ppf(lvl, scale = bs_std)
        bs_CIs = np.array([orgests - bs_half, orgests + bs_half])

    elif type == 'std_bstrp':
        bs_std = bs_ests.std(axis = 0, ddof = 1)
        bs_half = sps.norm.ppf(lvl, scale = bs_std)
        bs_CIs = np.array([orgests - bs_half, orgests + bs_half])
        
    elif type == 'updn_pctl':
        bs_CIs = np.quantile(bs_ests, [1 - lvl, lvl], axis = 0, method = 'weibull')

        z = sps.norm.ppf(lvl, scale = 1)
        bs_std = (bs_CIs[1] - bs_CIs[0]) / (2 * z)

    elif type == 'bc_bstrp':
        z0 = sps.norm.ppf(np.clip(np.mean(bs_ests < orgests, axis=0), 1e-5, 1-1e-5))
    
        # Calculate adjusted percentiles
        z_low, z_high = sps.norm.ppf([1 - lvl, lvl])
        p_low = sps.norm.cdf(2 * z0 + z_low) * 100
        p_high = sps.norm.cdf(2 * z0 + z_high) * 100
        
        # Extract values and return as 2x6 array
        lows = [np.percentile(bs_ests[:, i], p_low[i]) for i in range(truep.size)]
        highs = [np.percentile(bs_ests[:, i], p_high[i]) for i in range(truep.size)]
        
        bs_CIs = np.vstack([lows, highs])

        z = sps.norm.ppf(lvl, scale = 1)
        bs_std = (bs_CIs[1] - bs_CIs[0]) / (2 * z)

    elif type == 'mad_normal':
        bs_std = sps.median_abs_deviation(bs_ests, axis=0, scale='normal')
        bs_half = sps.norm.ppf(lvl, scale = bs_std)
        bs_CIs = np.array([orgests - bs_half, orgests + bs_half])

    olps = bs_CIs - truep
    CI_flag = np.maximum( -np.sign( olps[0] * olps[1]), 0).astype(int)

    return bs_std, CI_flag

def select_bandwidth(h_grid, cv_scores, tau, S=1.0, direction="decreasing", concavity="convex"):
    """
    Selects h using an interior minimum if it exists;
    otherwise, uses the Elbow (Distance) method.

    Parameters:
    h_grid (np.array): 1D array of bandwidth values
    cv_scores (np.array): 1D array of corresponding CV scores

    Returns:
    float: The selected optimal bandwidth h_star
    """
    n = len(h_grid)
    min_idx = np.argmin(cv_scores)

    # 1. Check for Interior Minimum
    # If the minimum isn't at the very first or very last index, it's a "U-shape"
    if 0 < min_idx < n - 1:
        print(f"Interior minimum found at index {min_idx}, c:{h_grid[min_idx]}.")
        return 'Interior', min_idx

    # 2. Fallback to Elbow Method (Distance from Secant Line)
    #print("No interior minimum found. Trying to find Elbow as bandwidth index...")

    # 1. Smoothing (Step 1)
    # The C# code often assumes pre-smoothed or uses a specific spline.
    # UnivariateSpline is the standard Python equivalent.
    x = h_grid
    try:
        y_smoothed = fit_loglinear_robust(h_grid, cv_scores)
    except:
        # Fallback to raw data if the curve is too linear for exponential fit
        sys.stderr.write("Fallback to raw data AS it is too linear for exponential fit\n")
        y_smoothed = cv_scores

    # 2. Normalization (Step 2)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y_smoothed), np.max(y_smoothed)
    x_sn = (x - x_min) / (x_max - x_min)
    y_sn = (y_smoothed - y_min) / (y_max - y_min)

    # 3. Difference Curve (Step 3)
    # Based on the paper: y_d = y_sn - x_sn (or rotated equivalent)
    # For a decreasing/convex CV curve, the C# equivalent logic uses:
    if direction == "decreasing" and concavity == "convex":
        y_d = (1 - x_sn) - y_sn
    else:
        # Standard increasing/concave case
        y_d = y_sn - x_sn

    # 4. Find Local Maxima (Step 4) - Candidate Knees
    lmx_indices = []
    for i in range(1, len(y_d) - 1):
        if y_d[i] > y_d[i-1] and y_d[i] > y_d[i+1]:
            lmx_indices.append(i)

    if not lmx_indices:
        elbow_idx = np.argmax(y_d)

    # 5 & 6. Thresholding Logic (The "S" Loop)
    # This matches the core loop in Kneedle.cs
    # T_lmx = y_d[lmx] - S * (avg_x_interval)
    avg_x_interval = 1.0 / (len(x) - 1)
    threshold_penalty = S * avg_x_interval

    for lmx_idx in lmx_indices:
        threshold = y_d[lmx_idx] - threshold_penalty

        # Look for the first point after lmx that drops below the threshold
        # before the next local maximum is reached.
        is_knee = False
        for j in range(lmx_idx + 1, len(y_d)):
            # If we hit another local maximum before dropping below threshold,
            # this candidate is ignored (it wasn't stable enough)
            if any(idx == j for idx in lmx_indices):
                break

            if y_d[j] <= threshold:
                is_knee = True
                break

        if is_knee:
            elbow_idx = lmx_idx
    if any(lmx_indices):
        elbow_idx = lmx_indices[0]
    else:
        #raise IndexError("Try additional bandwidth candidates!!")
        print('-- No elbow_idx found, use min at {min_idx}')
        elbow_idx = min_idx
    
    print(f"Elbow found at index {elbow_idx}: c:{h_grid[elbow_idx]}.")

    return 'Elbow', elbow_idx

def fit_loglinear_robust(x, y):
    # 1. Estimate the asymptote 'c'.
    # It must be slightly less than the minimum y to allow for log(y-c)
    # we iterate to find the 'c' that maximizes the linearity of the log-transform
    c_candidates = np.min(y) - (np.max(y) - np.min(y)) * np.logspace(-4, -1, 20)
    best_c = c_candidates[0]
    best_r2 = -np.inf

    for c in c_candidates:
        log_y = np.log(y - c)
        # Quick check for linearity
        r = np.corrcoef(x, log_y)[0, 1]
        if r**2 > best_r2:
            best_r2 = r**2
            best_c = c

    # 2. Perform Robust Regression in Log-Space
    # Theil-Sen is much more stable than OLS for noisy CV data
    log_y_final = np.log(y - best_c).reshape(-1, 1)
    x_reshaped = x.reshape(-1, 1)

    robust_model = TheilSenRegressor(random_state=1650)
    robust_model.fit(x_reshaped, log_y_final.ravel())

    # 3. Transform back to linear space
    # y = exp(intercept + slope * x) + c
    y_fitted = np.exp(robust_model.predict(x_reshaped)) + best_c
    return y_fitted

def fit_ushape_robust(x, y, degree=4):
    """
    Robustly fits a polynomial to a U-shaped curve and finds the minimum.
    Degree 2 is a standard parabola; Degree 4 allows for asymmetric U-shapes.
    """
    # 1. Create a Robust Polynomial Pipeline
    # Theil-Sen is used instead of OLS to ignore 'jumpy' CV points
    model = make_pipeline(PolynomialFeatures(degree), TheilSenRegressor(random_state=42))

    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, y)

    # 2. Create a dense grid for sub-grid precision
    #x_dense = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
    y_smooth = model.predict(x_reshaped)

    # 3. Find the minimum of the smooth robust curve
    #min_idx = np.argmin(y_smooth)
    #h_star = x_dense[min_idx][0]

    return y_smooth


if __name__=="__main__":
    print('in realDataFuncs.py')


#!/usr/bin/env python
import numpy as np
import csv
from scipy.stats import chi2

# Function to return our two functions we're setting to 0
# Plus the Jacobian
def funjac(mu, var, nvec, yarr, ybarvec):
    """
    Function that returns a vector of f and g functions evaluated at the
    specified variable values plus the Jacobian matrix.

    Parameters
    ----------
    mu : float.
         Mean of our response variable for our current iteration of Newton's.
    var : NumPy array of floats.
         Variance of our response variable for our current iteration of 
         Newton's.
    nvec: NumPy array of integers.
         The sample sizes of each group.
    yarr: NumPy array of floats.
         The response variable. Should be m (number of groups) x max(nvec) in size.
    ybarvec : NumPy array of floats.
         Means of each treatment group. Should be of size m x 1.

    Returns
    -------
    F : NumPy array of floats.
        Contains f and g values.
    J : NumPy array of floats.
        Contains the Jacobian.
    """
    # Function vector
    muHat = (np.sum(nvec*ybarvec/var))/(np.sum(nvec/var))
    f = mu - muHat
    g = var - 1/nvec * np.sum((yarr-mu)**2, axis=1)
    F = np.insert(g, 0, f)
    
    # Number of groups
    m = np.size(var)

    # Build Jacobian
    # Partial f/partial varj
    J0 = nvec/(var**2*np.sum(nvec/var))*(ybarvec - muHat)

    # partial f/partial mu = 1
    J0 = np.insert(J0, 0, 1)

    # This creates the entries for partial gk/partial varj
    J = np.eye(m+1, m+1)

    # Insert first row which is of f's partial derivatives
    J[0, :] = J0

    # partial gk/partial mu
    J[1:m+1, 0] = 2*(ybarvec-mu)
    
    return F, J

# Read the most important pieces of data from the CSV
def readData(fileName, groupNo, depVarNo):
    """
    Returns group variable and dependent variable in fileName that are in the
    columns specified by groupNo and depVarNo. 

    Parameters
    ----------
    fileName : string.
               The CSV file we're reading data from.
    groupNo  : int.
               An integer indicating the column in fileName in which our 
               grouping variable is.
    depVarNo : int.
               An integer indicating the column in fileName in which our
               dependent variable is.

    Returns
    -------
    group : NumPy array of integers.
            Contains the grouping variable for each observation.
    y     : NumPy array of floats.
            Contains the dependent variable value for each observation.
    """
    ifile = open(fileName)
    reader = csv.reader(ifile)
    y = np.array([])
    group = np.array([])
    count = 0
    for row in reader:
        # Do not include headers
        if (count != 0):
            group = np.append(group, int(row[groupNo]))
            y = np.append(y, float(row[depVarNo]))
        count += 1
    ifile.close()

    return group, y

def main():
    # All approximated via sample estimators
    group, y = readData("ProjectData.csv", 0, 5)
    m = int(np.max(group))
    nvec = np.tile(0, m)
    for i in range(0, m):
        nvec[i] = int(np.size(y[group == i+1]))

    ni = int(np.max(nvec))
    ybarvec = np.array([])
    yarr = np.tile(0.5, (m, ni))

    # yarr rows correspond to different groups
    # columns different observations
    for i in range(0, m):
        for j in range (0, ni):
            yarr[i, j] = y[group == i + 1][j]

    # Ybar_i
    ybarvec = np.mean(yarr, axis=1)
    # Initial guess of mu under the null
    muNullInit = np.mean(y)
    # Initial guess of var under the null
    varNullInit = np.var(yarr, axis=1)
    # Initialize variables for our loop below
    muNull = muNullInit
    varNull = varNullInit

    # Get variables for the first iteration
    F, J = funjac(muNull, varNull, nvec, yarr, ybarvec)
    eps = -np.linalg.solve(J, F)
    diff = np.sqrt(sum(eps**2)/(m+1))

    # Parameters of the loop
    iteration = 0
    itMax = 1e3
    tol = 1e-13
    param = np.tile(0.0, (m+1, 1))

    # Iterate until we get muNull and varNull to required
    # tolerance level or we run out of iterations
    while (tol < diff and iteration < itMax):
        muNull += eps[0]
        varNull += eps[1:m+1]
        F, J = funjac(muNull, varNull, nvec, yarr, ybarvec)
        eps = -np.linalg.solve(J, F)
        param[0] = muNull
        param[1:m+1] = np.reshape(varNull, (m, 1))
        epsRel = np.reshape(eps, (m+1, 1))/param
        diff = np.sqrt(np.sum(epsRel**2)/(m+1))
        iteration += 1

    # Hypothesis testing
    # Make an array of ybar values corresponding to each element of y
    ybararr = np.tile(1, np.shape(yarr))
    for i in range(0, m):
        ybararr[i, :] = ybarvec[i] * np.ones((1, nvec[i]))

    # Unrestricted MLE of variance
    varUnrest = (1/nvec * np.sum((yarr - ybararr)**2, axis=1))

    # Likelihood ratio
    lam = np.prod(np.power(varUnrest/varNull, nvec/2))
    # Test statistic -2ln(lam)
    stat = -2*np.log(lam)
    # Associated p-value
    pval = 1-chi2.cdf(stat, m-1)

    # Print relevant outputs
    print(f"Null mean             = {muNull:.3e}")
    for i in range(0, m):
        groupNo = i + 1
        print("-------------------------------------")
        print(f"For group             = {groupNo}")
        print(f"Null variance         = {varNull[i]:.3e}")
        print(f"Unrestricted mean     = {ybarvec[i]:.3e}")
        print(f"Unrestricted variance = {varUnrest[i]:.3e}")

    print("-------------------------------------")
    print(f"Test statistic        = {stat:.3f}")
    print(f"P-value               = {pval:.3e}")

if __name__ == "__main__":
    main()
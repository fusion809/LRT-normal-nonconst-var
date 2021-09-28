#!/usr/bin/env python
# Written in September 2021 by Brenton Horne
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
    mu      : float.
              Mean of our response variable for our current iteration of Newton's.
    var     : NumPy array of floats.
              Variance of our response variable for our current iteration of 
              Newton's.
    nvec    : NumPy array of integers.
              The sample sizes of each group.
    yarr    : NumPy array of floats.
              The response variable. Should be m (number of groups) x max(nvec)
              in size.
    ybarvec : NumPy array of floats.
              Means of each treatment group. Should be of size m x 1.

    Returns
    -------
    F       : NumPy array of floats.
              Contains f and g values.
    J       : NumPy array of floats.
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
    group    : NumPy array of integers.
               Contains the grouping variable for each observation.
    y        : NumPy array of floats.
               Contains the dependent variable value for each observation.
    """
    # Initialize variables for reading from file
    ifile = open(fileName)
    reader = csv.reader(ifile)

    # Initialize arrays to store variable data
    y = np.array([])
    group = np.array([])

    # Initialize count for loop below
    count = 0

    # Loop through the rows in input file
    for row in reader:
        # Do not include headers
        if (count != 0):
            group = np.append(group, int(row[groupNo]))
            y = np.append(y, float(row[depVarNo]))
        count += 1

    ifile.close()

    return group, y

def printVars(muNull, varNull, ybarvec, varUnrest, stat, pval, m, decPlaces):
    """
    Print variables of interest.

    muNull    : float.
                MLE of the mean under the null.
    varNull   : NumPy array of floats.
                MLE of the variance under the null.
    ybarvec   : NumPy array of floats.
                Mean of each sample.
    varUnrest : NumPy array of floats.
                MLE of the variance of each sample.
    stat      : float.
                Test statistic (-2 ln(lambda)).
    pval      : float.
                P-value for our likelihood-ratio test.
    m         : int.
                Number of groups.
    decPlaces : int.
                Number of decimal places to be displayed.
    """

    print(f"Null mean             = {muNull:.{decPlaces}e}")
    for i in range(0, m):
        groupNo = i + 1
        print("-------------------------------------")
        print(f"For group             = {groupNo}")
        print(f"Null variance         = {varNull[i]:.{decPlaces}e}")
        print(f"Unrestricted mean     = {ybarvec[i]:.{decPlaces}e}")
        print(f"Unrestricted variance = {varUnrest[i]:.{decPlaces}e}")

    print("-------------------------------------")
    print(f"Test statistic        = {stat:.{decPlaces}f}")
    print(f"P-value               = {pval:.{decPlaces}e}")

def getVars(group, y):
    """
    Calculate various variables we need from group and y.

    Parameters
    ----------
    group   : NumPy array of ints.
              Group variable corresponding to each observation.
    y       : NumPy array of floats.
              Dependent variable value corresponding to each observation.

    Returns
    -------
    m       : int.
              Number of groups.
    muNull  : NumPy array of floats.
              Initial estimate of our MLE for the mean under the null.
    varNull : NumPy array of floats.
              Initial estimate of our MLE for the variance under the null.
    nvec    : NumPy array of ints.
              Vector of sample sizes for each value of the grouping variable.
    yarr    : NumPy array of floats.
              Array of values of the dependent variable for each observation 
              with each row corresponding to a different value of the grouping
              variable.
    ybarvec : NumPy array of floats.
              Means of the dependent variable for each value of the grouping 
              variable.
    """
    # Number of groups
    m = int(np.max(group))

    # Vector of sample sizes
    nvec = np.tile(0, m)
    for i in range(0, m):
        nvec[i] = int(np.size(y[group == i+1]))

    # Maximum sample size
    ni = int(np.max(nvec))

    # Initialize 2D array for storing y values categorized by treatment group
    yarr = np.tile(0.5, (m, ni))

    # yarr rows correspond to different groups
    # columns different observations
    for i in range(0, m):
        for j in range (0, nvec[i]):
            yarr[i, j] = y[group == i + 1][j]

    # Ybar_i
    ybarvec = np.mean(yarr, axis=1)

    # Initial guess of mu under the null
    muNull = np.mean(y)

    # Initial guess of var under the null
    varNull = np.var(yarr, axis=1)

    return m, ni, muNull, varNull, nvec, yarr, ybarvec

def newtons(m, muNull, varNull, nvec, yarr, ybarvec):
    """
    Apply Newton's method to estimate the MLEs of the mean and variance under 
    the null hypothesis.

    Parameters
    ----------
    m       : int.
              Number of groups.
    muNull  : float.
              Initial guess of the MLE of the mean under the null.
    varNull : NumPy array of floats.
              Initial guess of the MLE of the variance under the null.
    nvec    : NumPy array of ints.
              Sample sizes for each value of our grouping variable.
    yarr    : NumPy array of floats.
              Dependent variable with each row corresponding to different
              values of the grouping variable and the columns corresponding
              to different observations.
    ybarvec : NumPy array of floats.
              Mean of the dependent variable for each value of the grouping
              variable.

    Returns
    -------
    muNull  : float.
              MLE of the mean under the null.
    varNull : NumPy array of floats.
              MLE of the variance under the null.
    """
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
        # Current iteration of Newton's
        muNull += eps[0]
        varNull += eps[1:m+1]

        # Set up vectors for next iteration of Newton's
        F, J = funjac(muNull, varNull, nvec, yarr, ybarvec)
        eps = -np.linalg.solve(J, F)

        # Put data from current iteration of Newton's into param vector
        param[0] = muNull
        param[1:m+1] = np.reshape(varNull, (m, 1))

        # Scaling eps, making it relative
        epsRel = np.reshape(eps, (m+1, 1))/param

        # Root mean square of epsRel
        diff = np.sqrt(np.sum(epsRel**2)/(m+1))

        # Up iteration counter by 1
        iteration += 1

    return muNull, varNull

def main():
    # All approximated via sample estimators
    group, y = readData("ProjectDataOutlierRm.csv", 0, 4)
    m, ni, muNull, varNull, nvec, yarr, ybarvec = getVars(group, y)

    # Use Newton's method to estimate mu and var under the null
    muNull, varNull = newtons(m, muNull, varNull, nvec, yarr, ybarvec)

    # Hypothesis testing
    # Make an array of ybar values corresponding to each element of y
    ybararr = np.tile(1, np.shape(yarr))
    for i in range(0, m):
        ybararr[i, :] = ybarvec[i] * np.ones((1, ni))

    # Unrestricted MLE of variance
    varUnrest = (1/nvec * np.sum((yarr - ybararr)**2, axis=1))

    # Likelihood ratio
    lam = np.prod(np.power(varUnrest/varNull, nvec/2))

    # Test statistic -2ln(lam)
    stat = -2*np.log(lam)

    # Associated p-value
    pval = 1-chi2.cdf(stat, m-1)

    # Print relevant outputs
    printVars(muNull, varNull, ybarvec, varUnrest, stat, pval, m, 3)

if __name__ == "__main__":
    main()
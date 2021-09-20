#!/usr/bin/env python
import numpy as np
import csv
from scipy.stats import chi2

# Function to return our two functions we're setting to 0
# Plus the Jacobian
def funjac(mu, var, nvec, yarr, ybarvec):
    muHat = (np.sum(nvec*ybarvec/var))/(np.sum(nvec/var))
    f = mu - muHat
    g = var - 1/nvec * np.sum((yarr-mu)**2, axis=1)
    F = np.insert(g, 0, f)
    m = np.size(var)
    J0 = nvec/(var**2*np.sum(nvec/var))*(ybarvec - muHat)
    J0 = np.insert(J0, 0, 1)
    # This creates the entries for partial gk/partial varj
    J = np.eye(m+1, m+1)
    J[0, :] = J0
    J[1:m+1, 0] = 2*(ybarvec-mu)
    return F, J

# Read the most important pieces of data from the CSV
def readData():
    ifile = open("ProjectData.csv")
    reader = csv.reader(ifile)
    y = np.array([])
    group = np.array([])
    count = 0
    for row in reader:
        if (count != 0):
            group = np.append(group, int(row[0]))
            y = np.append(y, float(row[5]))
        count += 1
    ifile.close()

    return group, y

# All approximated via sample estimators
group, y = readData()
m = int(np.max(group))
n = np.size(y)
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
tol = 1e-8

# Iterate until we get muNull and varNull to required
# tolerance level
while (tol < diff and iteration < itMax):
    muNull += eps[0]
    varNull += eps[1:m+1]
    F, J = funjac(muNull, varNull, nvec, yarr, ybarvec)
    eps = -np.linalg.solve(J, F)
    diff = np.sqrt(sum(eps**2)/(m+1))
    iteration += 1

# Hypothesis testing
# Make an array of ybar values corresponding to each element of y
ybararr = np.tile(1, np.shape(yarr))
for i in range(0, 6):
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
print("Null mean = ", muNull)
print("Null variance = ", varNull)
print("Unrestricted mean = ", ybarvec)
print("Unrestricted variance = ", varUnrest)
print("Test statistic = ", stat)
print("P-value = ", pval)
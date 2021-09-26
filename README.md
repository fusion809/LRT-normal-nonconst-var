# Likelihood-ratio test for samples from normally distributed populations with non-constant variances
This repository contains a TeX document (including .tex file, .toc file and .pdf file output for it) showing the statistical working for a likelihood-ratio test that should be roughly equivalent to the Welch's ANOVA procedure.

The Python script solver.py implements the test. Currently it is set up to use ProjectData.csv, a file I, the author of this repository, have in my local copy of it, that comes from the experimental project I did in STA3300 at USQ in Semester 1, 2021. This script uses Newton's method to derive the maximum likelihood estimators under the null, and uses an analytical formula derived in the TeX document in this repository for the unrestricted maximum likelihood. 

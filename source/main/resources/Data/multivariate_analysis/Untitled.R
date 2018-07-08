# http://www2.stat.duke.edu/~rcs46/lectures_2015/02-multivar2/02-multivar2.pdf
library(mvtnorm)
# Fixing the seed gives us a consistent set of simulated returns
set.seed(42)


x.points <- seq(-3,3,length.out=100)
y.points <- x.points
z <- matrix(0,nrow=100,ncol=100)
mu <- c(0,0)
sigma <- matrix(c(2,0,0,1),nrow=2)
for (i in 1:100) {
  for (j in 1:100) {
    z[i,j] <- dmvnorm(c(x.points[i],y.points[j]),
                      mean=mu,sigma=sigma)
  }
}


# Alt
w <- c(0.5, 0.5)
MC <- rmvnorm (10000 , mu , sigma)
var_mc <- MC %*%w 
#hist(var_mc, freq=F, breaks=12)
#lines(density(var_mc), col="red")
plot(density(var_mc), col="red")
abline(v=var_q <- quantile(var_mc ,p=0.05), col="blue")


# contour plot of the joint pdf of bivariate normal distribution as defined above
# Probability contours are ellipses
# Contour lines define regions of probability density (from high to low).
contour(x.points,y.points,z)


# What can we say in general about the MVN density?
# 1) The spectral decomposition theorem tells us that the contours of the multivariate normal distribution are ellipsoids.
# 2) The axes of the ellipsoids correspond to eigenvectors of the covariance matrix.
# 3) The radii of the ellipsoids are proportional to square roots ofthe eigenvalues of the covariance matrix.

HDIofICDF = function( ICDFname , credMass=0.95, tol=1e-8 , ... ) {
  # Arguments:
  #   ICDFname is R's name for the inverse cumulative density function
  #     of the distribution.
  #   credMass is the desired mass of the HDI region.
  #   tol is passed to R's optimize function.
  # Return value:
  #   Highest density iterval (HDI) limits in a vector.
  # Example of use: For determining HDI of a beta(30,12) distribution, type
  #   HDIofICDF( qbeta , shape1 = 30 , shape2 = 12 )
  #   Notice that the parameters of the ICDFname must be explicitly named;
  #   e.g., xdoes not work.
  # Adapted and corrected from Greg Snow's TeachingDemos package.
  incredMass =  1.0 - credMass
  intervalWidth = function( lowTailPr , ICDFname , credMass , ... ) {
    ICDFname( credMass + lowTailPr , ... ) - ICDFname( lowTailPr , ... )
  }
  optInfo = optimize( intervalWidth , c( 0 , incredMass ) , ICDFname=ICDFname ,
                      credMass=credMass , tol=tol , ... )
  HDIlowTailPr = optInfo$minimum
  return( c( ICDFname( HDIlowTailPr , ... ) ,
             ICDFname( credMass + HDIlowTailPr , ... ) ) )
}

mat_cov = matrix(c(1, 0, 0, 1), nrow=2, ncol=2) 
mat_cov
qmvnorm(p=0.05, interval=c(-5, 0), sigma=diag(10), tail="lower.tail")$quantile
qmvnorm(p=0.05, sigma=diag(10), tail="lower.tail")$quantile

HDIofICDF(ICDFname=qmvnorm, mean=c(0,0), sigma=diag(2), credMass=0.95)


HDIofICDF(ICDFname=qnorm, mean=0, sd=1, credMass=0.90)

##### Hyndman example for bivariate data
install.packages("hdrcde")
library(hdrcde)
# Simple bimodal example
x <- c(rnorm(100,0,1), rnorm(100,5,1), rnorm(100,-2,1))
x <- rnorm(1000000, 0, 1)
par(mfrow=c(1,2))
boxplot(x)
hdr.boxplot(x)
par(mfrow=c(1,1))
hdr.den(x, prob = c(90))$hdr
qnorm(p=c(0.95), mean = 0, sd = 1, lower.tail = TRUE)

# Bivariate example
x <- c(rnorm(200,0,1),rnorm(200,4,1)); #hdr.den(x)
y <- c(rnorm(200,0,1),rnorm(200,4,1)); #hdr.den(y)
x <- rnorm(10, 0, 1)
y <- rnorm(10, 0, 1)
hdrinfo <- hdr.2d(x=x, y=y, prob=c(0.90))
hdrinfo


####
x <- seq(0.4,12,0.4)
px <-  c(0,0, 0, 0, 0, 0, 0.0002, 0.0037, 0.018, 0.06, 0.22 ,0.43, 0.64,0.7579, 0.7870, 0.72, 0.555, 0.37, 0.24, 0.11, 0.07, 0.02, 0.009, 0.005, 0.0001, 0,0.0002, 0, 0, 0)
xx <- seq(min(x), max(x), by = 0.001)
# interpolate function from the sample
fx <- splinefun(x, px) # interpolating function
pxx <- pmax(0, fx(xx)) # normalize so prob > 0
# Find highest density region numerically
const <- sum(pxx)
spxx <- sort(pxx, decreasing = TRUE) / const
crit <- spxx[which(cumsum(spxx) >= 0.95)[1]] * const





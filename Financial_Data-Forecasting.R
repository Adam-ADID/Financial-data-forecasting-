library(rugarch)
library(tseries)
library(fBasics)
library(zoo) #Convert to time series data type
library(lmtest)
library(forecast)
library(quantmod)
# Download GOOG stock data from Yahoo Finance
getSymbols("IBM", src = "yahoo", from = "2002-01-01", to = "2017-12-31")

# Check the structure of the downloaded data
head(IBM)
ibm <- zoo(IBM$IBM.Close)
#Time series plot (non-stationary with varying mean and variance)
plot(ibm, type='l', ylab = " adj close price", main="Plot of 2002-2017 daily IBM stock prices")

acf(coredata(ibm), main="ACF plot of the 2002-2017 daily IBM stock prices")
pacf(coredata(ibm), main="PACF plot of the 2002-2017 daily IBM stock prices")

#log and One-Time Differencing Transformation
#To attain stationarity: log return time series

ibm_rets <- log(ibm/lag(ibm,-1))
plot(ibm_rets, type='l', ylab = " adj close price", main="Log return plot of 2002-2017 daily IBM stock prices")

#Augmented Dickey Fuller (ADF) Test
adf.test(ibm) #Original
adf.test(ibm_rets) #Log-return

#strip off the dates and create numeric object
ibm_ret_num <- coredata(ibm_rets)

#Autoregressive Conditional Heteroskedasticity (ARCH) Test
library(zoo)
library(FinTS)

ArchTest(ibm_ret_num,lag=12)

#the log return of time series data has ARCH effect and therefore GARCH model can be fitted.
#Exploratory Analysis
#Compute statistics
basicStats(ibm_rets) #mean is 0 and the distribution of log returns has large heavy tai


#QQ-plot
qqnorm(ibm_rets)
qqline(ibm_rets, col = 2)
kurtosis(ibm_rets) #positive, heavy-tailed distribution


#Time plot of square of log return of prices
# mean is constant and nearly 0
plot(ibm_rets^2,type='l', ylab = "square of stock price return", main="Plot of 2002-2017 daily IBM stock price squared return")


#Time plot of absolute value of log return of prices
plot(abs(ibm_rets),type='l', ylab = "abs value of stock price return", main="Plot of 2002-2017 daily IBM stock price abs return")

par(mfrow=c(3,1)) #show three plots in a figure 
acf(ibm_ret_num) #non-linear dependence
acf(ibm_ret_num^2) #strong non-linear dependence
acf(abs(ibm_ret_num)) #strong non-linear dependence
dev.off() 


install.packages('TSA')
library(TSA)

eacf(ibm_ret_num) #
eacf(abs(ibm_ret_num)) #suggest garch 11

#garch 11
g11=garch(ibm_ret_num,order=c(1,1))
g11

summary(g11) #checking p value

AIC(g11)

gBox(g11,method='squared')

#garch 12
g12=garch(ibm_ret_num,order=c(1,2))
summary(g12)

AIC(g12)
gBox(g12,method='squared') #not good model conduct diagonise checking 

#---1. GARCH(1,1) with normally distributed errors
garch11.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#estimate model 
garch11.fit=ugarchfit(spec=garch11.spec, data=ibm_rets)
garch11.fit

#---2. GARCH(1,1) model with t-distribution
garch11.t.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
garch11.t.fit=ugarchfit(spec=garch11.t.spec, data=ibm_rets)
garch11.t.fit

#---3. GARCH(1,1) model with skewed t-distribution
garch11.skt.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "sstd")
#estimate model 
garch11.skt.fit=ugarchfit(spec=garch11.skt.spec, data=ibm_rets)
garch11.skt.fit

#---4. eGARCH(1,1) model with t-distribution
egarch11.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
egarch11.t.fit=ugarchfit(spec=egarch11.t.spec, data=ibm_rets)
egarch11.t.fit

#---5. fGARCH(1,1) model with t-distribution
fgarch11.t.spec=ugarchspec(variance.model=list(model = "fGARCH", garchOrder=c(1,1), submodel = "APARCH"), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
fgarch11.t.fit=ugarchfit(spec=fgarch11.t.spec, data=ibm_rets)
fgarch11.t.fit

#---6. iGARCH (1,1) Model with Normal Distribution
igarch11.t.spec=ugarchspec(variance.model=list(model = "iGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0 , 0 )), distribution.model = "std")
igarch11.t.fit=ugarchfit(spec=igarch11.t.spec, data=ibm_rets)
igarch11.t.fit

# Step 3: Forecasting
# Set up the model for out-of-sample forecasting
rff = ugarchfit(spec = fgarch11.t.spec, data = ibm_rets, out.sample = 500)

# Forecast 20 steps ahead, rolling the forecast 450 times
rf = ugarchforecast(rff, n.ahead = 20, n.roll = 450)

# Step 4: Plot the forecast results
plot(rf, which = "all")

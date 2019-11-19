using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
include("PCA.jl")
(X,mu,sigma) = standardizeCols(X)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features
j1 = rand(1:d)
j2 = rand(1:d)
figure(2)
clf()
model = PCA(X,14)
w = model.W
Z = (w'\X')'
m = rand(1:n,10)
# plot(Z[:,1],Z[:,2],".")
# for i in 1:length(Z[:,1])
#     annotate(dataTable[i+1,1],
# 	xy=[Z[i,1],Z[i,2]],
# 	xycoords="data")
# end

(n,d) = size(X)
mu = mean(X,dims=1)
X -= repeat(mu,n,1)

show(norm((Z*w)-X)/norm(X))

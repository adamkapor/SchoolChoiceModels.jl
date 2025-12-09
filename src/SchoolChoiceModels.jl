module SchoolChoiceModels

using DynamicHMC, Distributions, Random, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, ForwardDiff


########################################################
##  regression coefficients
########################################################
"""delta = BayesOLS(A,deltabar,ustar,Xstar):
Draw coefficients from regression of ustar on Xstar.
Assumes that regression errors ustar-Xstar'delta are distributed N(0,1) iid
Prior is delta ~ MVN(deltabar,A^-1).
"""
function BayesOLS(ustar,Xstar,prior::MvNormal)
	(deltabar,Ainv) = params(prior)
	A = inv(Ainv)
	V = inv(Xstar'*Xstar .+ A)  #Xstar has dimensions (I*J,nDelta)
	deltatil = V*(Xstar'*vec(ustar) .+ A*deltabar)  #from AS appendix
	V_herm = (V + V')/2
	posterior = MvNormal(deltatil,V_herm)
end

function BayesGLS(u::Array{S},Sigma,X::Array{T},prior::MvNormal) where {S<:Number,T<:Array}
    C = getInvCholeskyFactor(Sigma)
    (deltabar,Ainv) = params(prior)
	A = Array{eltype(u),2}(inv(Ainv))
    invV = A
    deltatil = A*deltabar
    Xstar = similar(X[1])
    ustar = zeros(size(u,1))
    for ii=1:length(X)
        mul!(Xstar,C',X[ii]) #Xstar = C'*X[ii]
        mul!(ustar,C',view(u,:,ii)) #ustar = C'*view(u,:,ii)
        mul!(invV,Xstar',Xstar,1,1)
        mul!(deltatil,Xstar',ustar,1,1)
    end
    V = inv(invV)
    posterior = MvNormal(V*deltatil,Hermitian(V))
end

function BayesGLS(u::Array{S},Sigma,X::Array{T},prior::MvNormal) where  {S<:Array,T<:Array}
    C = getInvCholeskyFactor(Sigma)
    (deltabar,Ainv) = params(prior)
    A = Array{Float64,2}(inv(Ainv))
    invV = A
    deltatil = A*deltabar
    Xstar = similar(X[1])
    ustar = similar(LV.u[1])
    for ii=1:length(X)
        mul!(Xstar,C',X[ii]) #Xstar = C'*X[ii]
        mul!(ustar,C',u[ii]) #ustar = C'*view(u,:,ii)
        mul!(invV,Xstar',Xstar,1,1)
        mul!(deltatil,Xstar',ustar,1,1)
    end
    V = inv(invV)
    posterior = MvNormal(V*deltatil,Hermitian(V))
end


getInvCholeskyFactor(Sigma) = cholesky(Hermitian(inv(Sigma))).U #(J,J) upper triangular cholesky factor such that C'*C==Σ
getInvCholeskyFactor(sigma2::Number) = 1/sqrt(sigma2)

########################################################
##  means and variances
########################################################
"""BayesVariance: posterior variance of normal random variable v with mean mu."""
function BayesVariance(v, mu, prior = InverseGamma(1,1))
	(initial_df,B0) = params(prior)
	posterior = InverseGamma( initial_df+length(v)/2, B0+(sum((v.-mu).^2))/2)
end

"""BayesVCV: draw posterior VCV matrix of u~MultivariateNormal with mean 0"""
function BayesVCV(u,prior::InverseWishart=InverseWishart(size(u,1)+1,diagm(ones(size(u,1)))),inds=1:size(u,2))
	nRC = size(u,1)
	nInd = length(inds)
	df,Ψ0 = params(prior)
	Ψ = Matrix{Float64}(Ψ0)
	eps_i = zeros(nRC)
	for ii = inds
		@views eps_i .= u[:,ii] #if mean is not zero, would need to subtract mean here!
		mul!(Ψ,eps_i,eps_i',1.0,1.0) #Ψ += eps_i*eps_i'
	end
	posterior = InverseWishart(nInd+df,Ψ)
end

"""BayesMean: mean of iid normal random variable v with variance sigma2"""
function BayesNormalMean(v::AbstractArray,sigma2::Number,prior::Normal=Normal(0.0,10.0))
	BayesNormalMean(length(v),sum(v),sigma2,prior)
end
function BayesNormalMean(lengthv::Number,sumv::Number,sigma2::Number,prior::Normal=Normal(0.0,10.0))
	(mu0,sig0) = params(prior)
	term1 = (1/sig0^2 + lengthv/sigma2)
	posterior = Normal( (mu0/sig0^2 + sumv/sigma2)/term1, sqrt(1/term1))
end

end

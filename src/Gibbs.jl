function update_u!(LV,CC)
    for ii=1:II
        ui = LV.u[ii]
        μi = CC.x[ii]*LV.b
        ci = CC.c[ii]
        for jj=1:JJ
            if ci == jj
                u_2nd = -Inf
                for kk=1:length(ui)
                    kk==jj && continue
                    u_2nd = max(u_2nd,ui[kk])
                end
                lb = u_2nd
                ub = Inf
            else ci > 0
                lb = -Inf
                ub = ui[ci]
            end
            ui[jj] = rand(truncated(Gumbel(μi[jj]),lb,ub))
        end
    end
end

#likelihood of coefa
function ll_b(b,u,x)
    y = zeros(eltype(b),length(u[1]))
    ll = zero(eltype(b))
    for ii=1:II
        y .= u[ii]
        mul!(y,x[ii],b,-1,1)
        for jj=1:JJ
            ll += logpdf(Gumbel(),y[jj])
        end
    end
    ll
end

function update_b_mh!(LV,CC)
    b_proposed = LV.b .+ .005 .* randn.()
    ll_new = ll_b(b_proposed,LV.u,CC.x)
    ll_old = ll_b(LV.b,LV.u,CC.x)
    a = exp(ll_new - ll_old)
    if rand() < a
        LV.b .= b_proposed
        return true
    else
        return false
    end
end



Random.seed!(1234)
const II = 10000
const JJ = 10
const nX = 5
x = [[zeros(1,nX) ; [ones(JJ-1,1);;  randn(JJ-1,nX-1)]] for ii=1:II] #"observed" x's
btrue = ones(5) #truth!
u = [x_i*btrue .+ rand.(Gumbel()) for x_i in x]
c = [findmax(u_i)[2] for u_i in u]   #"observed" choices

CC = (x=x, c=c)
LV = (b=zeros(nX), u=deepcopy(u))


#########
# set up HMC
#########
struct RegressionCoefficientPosterior{S,T}
    u::S
    x::T
end
(problem::RegressionCoefficientPosterior)(θ) = ll_b(θ,problem.u,problem.x)
grad_problem(problem,x) = ForwardDiff.gradient(problem,x)

#need to define the following three methods: dimension, capabilities, logdensity
LogDensityProblems.dimension(::RegressionCoefficientPosterior) = nX
function LogDensityProblems.capabilities(::Type{<:RegressionCoefficientPosterior})
    LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.logdensity(problem::RegressionCoefficientPosterior, x)
    problem(x)
end
function LogDensityProblems.logdensity_and_gradient(problem::RegressionCoefficientPosterior, x)
    logdens = problem(x)
    grad = grad_problem(problem, x)
    logdens, grad
end

function HMCstep(LV,CC)
    rng = TaskLocalRNG()
    algo = DynamicHMC.NUTS(max_depth=10,min_Δ=-1000,turn_statistic_configuration=Val{:generalized}())
    κ = DynamicHMC.GaussianKineticEnergy(Diagonal(fill(0.01,nX)))
    ϵ = .01
    prob = RegressionCoefficientPosterior(LV.u,CC.x)
    steps = DynamicHMC.mcmc_steps(rng,algo,κ,prob,ϵ)
    q = LV.b
    lq, ∇lq = LogDensityProblems.logdensity_and_gradient(prob,q)
    Q0 = DynamicHMC.EvaluatedLogDensity(q,lq,∇lq) #q, lq, ∇lq
    Q,_ = DynamicHMC.mcmc_next_step(steps,Q0)
    res = Q.q
end

for t=1:100000
    update_u!(LV,CC)
    LV.b .= HMCstep(LV,CC)
    #updated = update_b_mh!(LV,CC)
    mod(t,10)==1 && println("iter $t: b=$(round.(LV.b, digits=3))")
end


LV = (b=copy(btrue), u=deepcopy(u))
for t=1:100000
    update_u!(LV,CC)
    updated = update_b_mh!(LV,CC)
    mod(t,100)==1 && println("iter $t: updated=$updated, b=$(round.(LV.b .- btrue, digits=3))")
end




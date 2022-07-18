using NeuralPDE, Cubature, Quadrature, Flux, ModelingToolkit
using GalacticOptim, Optim, DiffEqFlux, Noise, SpecialFunctions
using JLD2, LinearAlgebra, Statistics, Random
import ModelingToolkit: Interval, infimum, supremum
# TODO
# Integrate errors
# Velocity defined uncertainty -- Done
# Extend time domain to past 1 period -- Doesn't work
# Run without pressure data -- Done
# Run with fewer GridTraining points -- Constrained by some limit, investigate further
# Add ϕ component

## TEST CASE: Constant heat flux pipe flow
case_title = "Pipe_Constantq_3D_CPU"
##

@parameters t, r, ϕ, z
@variables vr(..), vϕ(..), vz(..), p(..), T(..)
Dt = Differential(t)
Dr = Differential(r)
Dϕ = Differential(ϕ)
Dz = Differential(z)
Drr = Differential(r)^2
Dϕϕ = Differential(ϕ)^2
Dzz = Differential(z)^2

# # Constants for water at 20 deg C
ρ = 988.21 # kg/m3
μ = 0.0010016 # kg/m-s 
nu = μ/ρ #m2/s
k = 0.598 # W/m-K
Cp = 4184 # J/kg-K
alpha = k/(ρ*Cp)
D = 0.1 
R = D/2
ReD = 1000
Um = (ReD*nu)/D
delP = (-Um*8*μ)/(R^2)
P = 2*pi*R
M = pi*R^2*ρ*Um
Pr = 3.57
Tmi = 20+273 #K
Qs = 500 # heat flux W/m2
Lh = D*0.056*ReD # hydrodynamic entrance length
Lt = D*0.043*Pr*ReD # thermal entrance length
Li = round(Lt;digits=0)
Lo = 5+Li
L = 1

# 3D NSE + Temperature equation for cylindrical coords (My god this is horrible I want to die)
eqs = [
        Dr(r*vr(t,r,ϕ,z)) + Dϕ(vϕ(t,r,ϕ,z)) + r*Dz(vz(t,r,ϕ,z)) ~ 0.0 ,

        r*ρ*(r*Dt(vr(t,r,ϕ,z)) 
        + r*vr(t,r,ϕ,z)*Dr(vr(t,r,ϕ,z)) 
        + vϕ(t,r,ϕ,z)*Dϕ(vr(t,r,ϕ,z)) 
        + r*vz(t,r,ϕ,z)*Dz(vr(t,r,ϕ,z))
        - vϕ(t,r,ϕ,z)^2) ~ 
        -r^2*Dr(p(t,r,ϕ,z)) + μ*(-vr(t,r,ϕ,z) + r*Dr(vr(t,r,ϕ,z)) + r^2*Drr(vr(t,r,ϕ,z)) + Dϕϕ(vr(t,r,ϕ,z)) + r^2*Dzz(vr(t,r,ϕ,z)) - 2*Dϕ(vϕ(t,r,ϕ,z))),

        r*ρ*(r*Dt(vϕ(t,r,ϕ,z)) 
        + r*vr(t,r,ϕ,z)*Dr(vϕ(t,r,ϕ,z)) 
        + vϕ(t,r,ϕ,z)*Dϕ(vϕ(t,r,ϕ,z)) 
        + r*vz(t,r,ϕ,z)*Dz(vϕ(t,r,ϕ,z))
        - vϕ(t,r,ϕ,z)*vr(t,r,ϕ,z)) ~ 
        - r*Dϕ(p(t,r,ϕ,z)) + μ*(-vϕ(t,r,ϕ,z) + r*Dr(vϕ(t,r,ϕ,z)) + r^2*Drr(vϕ(t,r,ϕ,z)) + Dϕϕ(vϕ(t,r,ϕ,z)) + r^2*Dzz(vϕ(t,r,ϕ,z)) + 2*Dϕ(vϕ(t,r,ϕ,z))),

        r*ρ*(r*Dt(vz(t,r,ϕ,z)) 
        + r*vr(t,r,ϕ,z)*Dr(vz(t,r,ϕ,z))
        + vϕ(t,r,ϕ,z)*Dϕ(vz(t,r,ϕ,z))
        + r*vz(t,r,ϕ,z)*Dz(vz(t,r,ϕ,z))) ~ 
        - r^2*Dz(p(t,r,ϕ,z)) + μ*(r*Dr(vz(t,r,ϕ,z)) + r^2*Drr(vz(t,r,ϕ,z)) + Dϕϕ(vz(t,r,ϕ,z)) + r^2*Dzz(vz(t,r,ϕ,z))),

        r*ρ*(r*Dt(T(t,r,ϕ,z))
        + r*vr(t,r,ϕ,z)*Dr(T(t,r,ϕ,z))
        + vϕ(t,r,ϕ,z)*Dϕ(T(t,r,ϕ,z))
        + r*vz(t,r,ϕ,z)*Dz(T(t,r,ϕ,z))) ~
        k*(r*Dr(T(t,r,ϕ,z)) + r^2*Drr(T(t,r,ϕ,z)) + Dϕϕ(T(t,r,ϕ,z)) + r^2*Dzz(T(t,r,ϕ,z)))
]

indvars = [t,r,ϕ,z]
depvars = [vr(t,r,ϕ,z),vϕ(t,r,ϕ,z),vz(t,r,ϕ,z),p(t,r,ϕ,z),T(t,r,ϕ,z)]

# Space and time domains
tspan = [0,1]; rspan = [0,R]; ϕspan = [0,2pi]; zspan = [Li,Lo]
domains = [
        t ∈ Interval(tspan[1],tspan[end]),
        r ∈ Interval(rspan[1],rspan[end]),
        ϕ ∈ Interval(ϕspan[1],ϕspan[end]),
        z ∈ Interval(zspan[1],zspan[end])
        ]

# Analytic generator
analytic_sol_func_vr(t,r,ϕ,z) = 0.0*r*t*z
analytic_sol_func_vϕ(t,r,ϕ,z) = 0.0*r*t*z
analytic_sol_func_vz(t,r,ϕ,z) = 2*Um*(1-((r^2)/(R^2)))
analytic_sol_func_p(t,r,ϕ,z) = delP*(z-Li)-(Lo-Li)*delP
analytic_sol_func_T(t,r,ϕ,z) = Tmi + ((4*Qs/(k*R))*(((r^2)/4)- ((r^4)/(16*(R^2))))) - ((7*R*Qs)/(24*k)) + (((2*Qs*alpha)/(R*k*Um))*z)


nt = 5;                # Temporal points
nr = 4; nϕ = 8; nz = 5  
np = nr*nϕ*nz; np = 200 # Spatial points

C = ((np-1)*pi*(3-sqrt(5)))/R; # Fibonacci spiral
fib(r) = C*r

tr = LinRange(0,tspan[end],nt); 
rr = LinRange(0,R,np);
ϕr = mod.(fib(rr),2pi);
# rr = LinRange(0,R,nr)
# ϕr = LinRange(0,2pi*(7/8),nϕ)
zr = shuffle(LinRange(Li,Lo,np))
coords = zeros(length(indvars),nt*np)     

for i in 1:nt, j in 1:np
    coords[:,j+(i-1)*(np)] = [tr[i],rr[j],ϕr[j],zr[j]]       
end

# for i in 1:nt, j in 1:nrϕ, k in 1:nz
#         coords[:,k+(j-1)*(nz)+(i-1)*(nrϕ*nz)] = [tr[i],rr[j],ϕr[j],zr[k]]       
# end
# for i in 1:nt, j in 1:nr, k in 1:nϕ, l in 1:nz
#         coords[:,l+(k-1)*(nz)+(j-1)*(nϕ*nz) + (i-1)*(np)] = [tr[i],rr[j],ϕr[k],zr[l]]       
# end
# Boundary conditions/data points

vr_vec = [analytic_sol_func_vr(tr[i],rr[j],ϕr[j],zr[j]) for i in 1:nt for j in 1:np]
vϕ_vec = [analytic_sol_func_vϕ(tr[i],rr[j],ϕr[j],zr[j]) for i in 1:nt for j in 1:np]
vz_vec = [analytic_sol_func_vz(tr[i],rr[j],ϕr[j],zr[j]) for i in 1:nt for j in 1:np]
p_vec = [analytic_sol_func_p(tr[i],rr[j],ϕr[j],zr[j]) for i in 1:nt for j in 1:np]
T_vec = [analytic_sol_func_T(tr[i],rr[j],ϕr[j],zr[j]) for i in 1:nt for j in 1:np]

# vr_vec = [analytic_sol_func_vr(tr[i],rr[j],ϕr[j],zr[k]) for i in 1:nt for j in 1:nrϕ for k in 1:nz]
# vϕ_vec = [analytic_sol_func_vϕ(tr[i],rr[j],ϕr[j],zr[k]) for i in 1:nt for j in 1:nrϕ for k in 1:nz]
# vz_vec = [analytic_sol_func_vz(tr[i],rr[j],ϕr[j],zr[k]) for i in 1:nt for j in 1:nrϕ for k in 1:nz]
# p_vec = [analytic_sol_func_p(tr[i],rr[j],ϕr[j],zr[k]) for i in 1:nt for j in 1:nrϕ for k in 1:nz]
# T_vec = [analytic_sol_func_T(tr[i],rr[j],ϕr[j],zr[k]) for i in 1:nt for j in 1:nrϕ for k in 1:nz]

# vr_vec = [analytic_sol_func_vr(tr[i],rr[j],ϕr[k],zr[l]) for i in 1:nt for j in 1:nr for k in 1:nϕ for l in 1:nz]
# vϕ_vec = [analytic_sol_func_vϕ(tr[i],rr[j],ϕr[k],zr[l]) for i in 1:nt for j in 1:nr for k in 1:nϕ for l in 1:nz]
# vz_vec = [analytic_sol_func_vz(tr[i],rr[j],ϕr[k],zr[l]) for i in 1:nt for j in 1:nr for k in 1:nϕ for l in 1:nz]
# p_vec = [analytic_sol_func_p(tr[i],rr[j],ϕr[k],zr[l]) for i in 1:nt for j in 1:nr for k in 1:nϕ for l in 1:nz]
# vz_vec = mult_gauss(vz_vec,0.0,1)
data = reshape.([vr_vec,vϕ_vec,vz_vec,p_vec,T_vec],1,nt*np)

bcs = [
        vr(t,R,ϕ,z) ~ 0,
        vϕ(t,R,ϕ,z) ~ 0,
        vz(t,R,ϕ,z) ~ 0,
        Dz(p(t,r,ϕ,z)) ~ delP,
        # vr(t,r,ϕ,z) ~ vr(t,r,ϕ+2pi,z),
        # vϕ(t,r,ϕ,z) ~ vϕ(t,r,ϕ+2pi,z),
        # vz(t,r,ϕ,z) ~ vz(t,r,ϕ+2pi,z),
        # p(t,r,ϕ,z) ~ p(t,r,ϕ+2pi,z),
        Dr(vz(t,0,ϕ,z)) ~ 0,
        Dr(T(t,0,ϕ,z)) ~ 0,
        Dϕ(vz(t,r,ϕ,z)) ~ 0,
        Dr(vr(t,r,ϕ,z)) ~ 0,
        Dϕ(vr(t,r,ϕ,z)) ~ 0,
        Dr(vϕ(t,r,ϕ,z)) ~ 0,
        Dϕ(vϕ(t,r,ϕ,z)) ~ 0,
        Dr(p(t,r,ϕ,z)) ~ 0,
        Dϕ(p(t,r,ϕ,z)) ~ 0,
        Dr(T(t,R,ϕ,z)) ~ Qs/k   
        # p(0,r,0) ~ P0 + 0.5*ρ*K*L,
        # p(0,r,L) ~ P0 - 0.5*ρ*K*L,
        # p(2pi/ω,r,0) ~ P0 + 0.5*ρ*K*L,
        # p(2pi/ω,r,L) ~ P0 - 0.5*ρ*K*L,
        # p(t,r,L/2) ~ P0,
        # Dz(p(t,r,ϕ,z)) ~ -ρ*K*cos(ω*t),
        # p(0,r,z) ~ p(2pi/ω,r,z),
        ]
# bcs = []

@named pde_system = PDESystem(eqs,bcs,domains,indvars,depvars)

# Neural Network
input_ = length(indvars)
dx = [tspan[end]/nt,R/nr,2pi/nϕ,L/nz]  
act_func = Flux.tanh_fast
# n = 60
# chain = [FastChain(FastDense(input_,n,act_func),
#                 FastDense(n,n,act_func),
#                 FastDense(n,n,act_func),
#                 FastDense(n,n,act_func),
#                 FastDense(n,1)) for _ in 1:length(depvars)
# ] 
n = 60
chain = [FastChain(FastDense(input_,n,act_func),FastDense(n,n,act_func),FastDense(n,1)) for _ in 1:length(eqs)]

strategy = GridTraining(Float32.(dx))
# initθ = map(c -> Float32.(c), DiffEqFlux.initial_params.(chain))
initθ = load(case_title*"_$n"*".jld2")["single_stored_object"]
flat_initθ = reduce(vcat,initθ)

eltypeθ = eltype(initθ[1])
parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])

# discretization = PhysicsInformedNN(chain, strategy, init_params= initθ)

phi = NeuralPDE.get_phi.(chain,parameterless_type_θ)

map(phi_ -> phi_(rand(Float32,length(indvars),10), flat_initθ),phi)
# prob = NeuralPDE.discretize(pde_system,discretization)

derivative = NeuralPDE.get_numeric_derivative()

_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,nothing,
                                                    chain,initθ,strategy) for eq in eqs]
map(loss_f -> loss_f(rand(Float32,length(indvars),10), flat_initθ),_pde_loss_functions)

bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                                    phi,derivative,nothing,chain,initθ,strategy,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
map(loss_f -> loss_f(rand(Float32,length(indvars),10), flat_initθ),_bc_loss_functions)

train_sets = NeuralPDE.generate_training_sets(domains,dx,eqs,bcs,eltypeθ,indvars,depvars)
train_domain_set, train_bound_set = train_sets

pde_loss_functions = [NeuralPDE.get_loss_function(_loss,_set,
                                                eltypeθ,parameterless_type_θ,
                                                strategy) for (_loss,_set) in zip(_pde_loss_functions,train_domain_set)]
map(l->l(flat_initθ), pde_loss_functions)

bc_loss_functions = [NeuralPDE.get_loss_function(_loss,_set,
                                                 eltypeθ, parameterless_type_θ,
                                                 strategy) for (_loss,_set) in zip(_bc_loss_functions,train_bound_set)]

map(l->l(flat_initθ) ,bc_loss_functions)

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]

function vr_loss(θ)
        return sum(abs2, phi[1](coords,θ[sep[1]]) .- data[1])/np
end

function vϕ_loss(θ)
        return sum(abs2, phi[2](coords,θ[sep[2]]) .- data[2])/np
end

function vz_loss(θ)
        return sum(abs2, phi[3](coords,θ[sep[3]]) .- data[3])/np
end

function p_loss(θ)
        return sum(abs2, phi[4](coords,θ[sep[4]]) .- data[4])/np
end

function T_loss(θ)
        return sum(abs2, phi[5](coords,θ[sep[5]]) .- data[5])/np
end

# LOSS FUNCTION

loss_functions = [pde_loss_functions; bc_loss_functions; T_loss]# vr_loss; vϕ_loss; vz_loss]#; p_loss]    

pde_weights = [1,1,1,100,100]
bc_weights = [1,1,10,10,10,10,1,1,1,1,1,1,1,10]
data_weights = [10000]
λ = [pde_weights;bc_weights;data_weights]
λ = λ/norm(λ)

function loss_function(θ,p)
        return sum(λ.*map(l->l(θ) ,loss_functions))
end

wghts = []
loss_list = []
pde_loss = []
bc_loss = []
data_loss = []
cb_ = function (p,l)
        if l != NaN
        iter = length(wghts) + 1
        println("$iter: Loss is: $l")
        push!(loss_list, l)
        push!(wghts,p)
        push!(pde_loss,[λ[i].*loss_functions[i](p) for i in 1:length(eqs)])
        push!(bc_loss,[λ[i].*loss_functions[i](p) for i in length(eqs)+1:length(eqs)+length(bcs)])      
        push!(data_loss,[λ[i].*loss_functions[i](p) for i in length(eqs)+length(bcs)+1:length(loss_functions)])
        #println("loss: ", l , "losses: ", map(l -> l(p), loss_functions))
        #println()
                return false
        else
                println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                return true
        end
end

cb = function (p,l)
        println("Current loss is: $l")
        return false
end

# optimizer
f_ = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)
# res = GalacticOptim.solve(prob,ADAM(1e-2); cb = cb_, maxiters=4)
# res = GalacticOptim.solve(prob,LBFGS(); cb = cb_, maxiters=2)
nAdam = 2.5*[1e2,1e2,1e3,1e3]
rAdam = [1e-2,1e-3,1e-4,1e-5]
for i in 1:length(rAdam) 
        println("Stage $i -- $(nAdam[i])xADAM($(rAdam[i])):")
        global res = GalacticOptim.solve(prob, ADAM(rAdam[i]); cb = cb_, maxiters=nAdam[i]-1, maxtime=4*3600)
        global prob = remake(prob,u0=res.minimizer)
        println("")
end

# res = GalacticOptim.solve(prob,ADAM(1e-2); cb = cb_, maxiters=1000)
# println("Stage 2")
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb_,allow_f_increases = true, maxiters=3000, maxtime=36000)

minimizers_ = [res.minimizer[s] for s in sep]
function save_weights()
        save_object(case_title*"_$n"*".jld2",minimizers_)
        println("OPTIMISED WEIGHTS SAVED")
end
save_weights()
pde_loss = [[pde_loss[i][j] for i in 1:length(pde_loss)] for j in 1:length(eqs)]
bc_loss = [[bc_loss[i][j] for i in 1:length(bc_loss)] for j in 1:length(bcs)]
data_loss = [[data_loss[i][end] for i in 1:length(data_loss)]]

using Plots
gr()

# ns = [30,24,24,24]
# ts, rs, ϕs, zs = [LinRange(infimum(domains[i].domain),supremum(domains[i].domain),ns[i]) for i in 1:length(domains)]

# u_predict  = [[Array(phi[i]([t,r,ϕ,z],minimizers_[i]))[1] for r in rs for ϕ in ϕs for z in zs for t in ts] for i in 1:length(depvars)]
# vr_predict = [reshape(u_predict[1][i],(length(rs),length(zs)))[1,:] for i in 1:length(ts)]
# vz_predict = [reshape(u_predict[2][i],(length(rs),length(zs)))[50,:] for i in 1:length(ts)]
# p_predict = [reshape(u_predict[3][i],(length(rs),length(zs)))[:,50] for i in 1:length(ts)]
# p_predict_field = [u_predict[3][i] for t in ts]

# vz_analytic = [[analytic_sol_func_vz(t,rs[i],nothing,nothing) for i in 1:length(rs)] for t in ts]
# p_analytic = [[analytic_sol_func_p(t,nothing,nothing,zs[i]) for i in 1:length(rs)] for t in ts]

# p_analytic_field = [[analytic_sol_func_p(t,r,ϕ,z) for r in rs for ϕ in ϕs for z in zs] for t in ts]
# p_analytic_volume = [reshape(p_analytic_field[i],(length(rs),length(ϕs),length(zs))) for i in 1:length(ts)]


# v_diff = [abs.(vz_predict[i] .- vz_analytic[i]) for i in 1:length(ts)]
# p_diff = [abs.(u_predict[3][i] .- p_analytic_field[i]) for i in 1:length(ts)]

ts = LinRange(tspan[1],tspan[end],50)

function plot_loss()
        plot(loss_list, yaxis=:log,title = "ADAM cascade" , xlabel = "Iterations", ylabel = "loss")
        loss_type = ["PDE","BC","data"]; loss_amt = [sum(pde_loss[end]),sum(bc_loss[end]),sum(data_loss[end])]
        pie(loss_type,loss_amt,title = "Source of losses at last iter.",autopct="%1.1f%%",l=0.5)
end

function plot_polar_vz(z)
    rs = LinRange(rspan[1],rspan[end],30)
    ϕs = LinRange(0,2pi,30)
        anim = @animate for i ∈ 1:length(ts)
                t = round(ts[i],digits=2)
                vz_flat = hcat([analytic_sol_func_vz(ts[i],r,ϕ,z) for r in rs for ϕ in ϕs])
                p1 = heatmap(ϕs,rs/R,vz_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
                p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
                vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
                p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
                p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
                plot(p1,p2)
        end
        gif(anim, "Velocity_polar_evolution.gif", fps = 50)
end

function plot_polar_p(z)
        rs = LinRange(rspan[1],rspan[end],30)
        ϕs = LinRange(0,2pi,30)
            anim = @animate for i ∈ 1:length(ts)
                    t = round(ts[i],digits=2)
                    p_flat = hcat([analytic_sol_func_p(ts[i],r,ϕ,z) for r in rs for ϕ in ϕs])
                    p1 = heatmap(ϕs,rs/R,p_flat,projection = :polar,clim=(1,3))
                    p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
                    p_centre_fluc = Array(phi[4]([ts[1],0,0,L/2],minimizers_[4]))[1] - analytic_sol_func_p(ts[i],0,0,L/2)
                    p_predict_flat = hcat([Array(phi[4]([ts[i],r,ϕ,z],minimizers_[4]))[1] for r in rs for ϕ in ϕs]) .- p_centre_fluc
                    p2 = heatmap(ϕs,rs/R,p_predict_flat,projection = :polar,clim=(1,3))
                    p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
                    plot(p1,p2)
            end
            gif(anim, "Pressure_polar_evolution.gif", fps = 50)
end

function plot_v(z)
        anim = @animate for i ∈ 1:length(ts)
                t = round(ts[i],digits=2)
                rs = LinRange(rspan[1],rspan[end],20)
                vz_predict =    [[Array(phi[3]([ts[i],r,Float64(pi),z],minimizers_[3]))[1] for r in rs[end:-1:1]];
                                [Array(phi[3]([ts[i],r,0,z],minimizers_[3]))[1] for r in rs]]
                vz_analytic =   [[analytic_sol_func_vz(ts[i],r,Float64(pi),z) for r in rs[end:-1:1]];
                                [analytic_sol_func_vz(ts[i],r,0,z) for r in rs]]
                v_diff = abs.(vz_predict.-vz_analytic)               
                rs = [-rs[end:-1:1];rs]
                p1 = plot(vz_predict/u0,rs/R, linestyle = :dash, lw = 4, xlims = (-1.2,1.2), color = "red", label = "Predicted");
                p1 = plot!(vz_analytic/u0,rs/R, color = "blue", title = "Velocity profile at t = $t s", label = "Analytic");
                p2 = plot(v_diff/u0,rs/R, xlims=(0,0.5), title = "Velocity Error");
                plot(p1,p2)
        end
        gif(anim, "Velocity_profile_evolution.gif", fps = 50)
end

function plot_p(r)
        zs = LinRange(zspan[1],zspan[end],100)
        anim = @animate for i ∈ 1:length(ts)
                t = round(ts[i],digits=2)
                p_predict = [Array(phi[4]([ts[i],r,0,z],minimizers_[4]))[1] for z in zs]
                p_analytic =  [analytic_sol_func_p(ts[i],r,0,z) for z in zs]
                p1 = plot(zs/L, p_predict .- (p_predict[50]-p_analytic[50]),linestyle = :dash, lw = 4, color = "red", title = "Predicted pressure at t = $t s", label = "Predicted");
                p1 = plot!(zs/L, p_analytic, color = "blue", title = "Centreline pressure", label = "Analytic", xlabel="z/L", ylabel="Pressure");
                plot(p1)
        end
        gif(anim, "Pressure gradient.gif", fps = 50)
end


using NeuralPDE, CUDA, Cubature, Quadrature, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Noise, SpecialFunctions, MAT, Dierckx, StatsBase
import ModelingToolkit: Interval, infimum, supremum

# TODO

@parameters t, x, y
@variables u(..), v(..), p(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
#Dz = Differential(z)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
#Dzz = Differential(z)^2

# 3D PDE
ρ = 1.0; μ = 0.01; g = 0.0;

eqs = [
        Dx(u(t,x,y)) + Dy(v(t,x,y)) ~ 0.0,

        ρ*(Dt(u(t,x,y)) + u(t,x,y)*Dx(u(t,x,y)) + v(t,x,y)*Dy(u(t,x,y))) ~ 
        -Dx(p(t,x,y)) + μ*(Dxx(u(t,x,y)) + Dyy(u(t,x,y))),

        ρ*(Dt(v(t,x,y)) + u(t,x,y)*Dx(v(t,x,y)) + v(t,x,y)*Dy(v(t,x,y))) ~ 
        -Dy(p(t,x,y)) + μ*(Dxx(v(t,x,y)) + Dyy(v(t,x,y)))
]

indvars = [t,x,y]
depvars = [u(t,x,y),v(t,x,y),p(t,x,y)]

# Load data
data_vort = matread("cylinder_nektar_t0_vorticity.mat")
data_wake = matread("cylinder_nektar_wake.mat")

U_star = data_wake["U_star"]            # N x 2 x T
P_star = data_wake["p_star"]            # N x T
t_star = data_wake["t"]                 # T x 1
X_star = data_wake["X_star"]            # N x 2

N = size(X_star)[1]
T = size(t_star)[1]

# Rearrange data
XX = repeat(X_star[:,1],outer = [1,T])  # N x T
YY = repeat(X_star[:,2],outer = [1,T])  # N x T
TT = repeat(t_star', inner = [N,1])     # N x T

UU = U_star[:,1,:]                      # N x T
VV = U_star[:,2,:]                      # N x T
PP = P_star                             # N x T

x_vec = vec(XX)                         # NT x 1
y_vec = vec(YY)                         # NT x 1
t_vec = vec(TT)                         # NT x 1

u_vec = vec(UU)                         # NT x 1
v_vec = vec(VV)                         # NT x 1
p_vec = vec(PP)                         # NT x 1

# Training data

N_train = 2000
idx = sample(1:N*T,N_train)

x_train = x_vec[idx]                    # N_train x 1
y_train = y_vec[idx]                    # N_train x 1
t_train = t_vec[idx]                    # N_train x 1
u_train = u_vec[idx]                    # N_train x 1
v_train = v_vec[idx]                    # N_train x 1
p_train = p_vec[idx]                    # N_train x 1

nx = 100; ny = 50
coords = zeros(length(indvars),N_train) # 3 x N_train

for i in 1:N_train
        coords[:,i] = [x_vec[i],y_vec[i],t_vec[i]]       
end

data = reshape.([u_train,v_train,p_train],1,N_train)

bcs = []

# Space and time domains
lb = minimum(X_star; dims=1); ub = maximum(X_star; dims=1)
domains = [
        x ∈ Interval(lb[1],ub[1]),
        y ∈ Interval(lb[2],ub[2]),
        t ∈ Interval(t_star[1],t_star[end])]

@named pde_system = PDESystem(eqs,bcs,domains,indvars,depvars)

# Neural Network
input_ = length(indvars)
n = 48
dx = [t_star[end]/40,ub[1]/20,ub[2]/20]  
chain = [FastChain(FastDense(input_,n,Flux.tanh_fast),FastDense(n,n,Flux.tanh_fast),FastDense(n,n,Flux.tanh_fast),FastDense(n,1)) for _ in 1:length(depvars)]
strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=CubatureJLh(),reltol= 1e-6,abstol= 1e-4,maxiters=1e4)
# strategy = GridTraining(dx)
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) |> gpu
flat_initθ = reduce(vcat,initθ)

eltypeθ = eltype(initθ[1])
parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])

phi = NeuralPDE.get_phi.(chain,parameterless_type_θ)

map(phi_ -> phi_(rand(3,10), flat_initθ),phi)

derivative = NeuralPDE.get_numeric_derivative()

_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,nothing,
                                                        chain,initθ,strategy) for eq in eqs]
map(loss_f -> loss_f(rand(3,10), flat_initθ),_pde_loss_functions)

bc_indvars = NeuralPDE.get_argument(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,nothing,
                                                        chain,initθ,strategy,bc_indvars = bc_indvar) 
                                                        for (bc,bc_indvar) in zip(bcs,bc_indvars)]
map(loss_f -> loss_f(rand(3,10), flat_initθ), _bc_loss_functions)

# train_sets = NeuralPDE.generate_training_sets(domains,dx,eqs,bcs,eltypeθ,indvars,depvars)     # GRIDTRAINING
# train_domain_set, train_bound_set = train_sets                                                # GRIDTRAINING

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,eqs,bcs,eltypeθ,indvars,depvars,strategy) # QUADRATURE

plbs,pubs = pde_bounds                                                                          # QUADRATURE
pde_loss_functions = [NeuralPDE.get_loss_function(_loss,lb,ub,
                                                eltypeθ,parameterless_type_θ,
                                                strategy) for (_loss,lb,ub) in zip(_pde_loss_functions,plbs,pubs)]      # QUADRATURE
# pde_loss_functions = [NeuralPDE.get_loss_function(_loss,_set,
#                                                 eltypeθ,parameterless_type_θ,
#                                                 strategy) for (_loss,_set) in zip(_pde_loss_functions,train_domain_set)] # GRIDTRAINING
map(l->l(flat_initθ), pde_loss_functions)

blbs,bubs = bcs_bounds
bc_loss_functions = [NeuralPDE.get_loss_function(_loss,lb,ub,
                                                 eltypeθ, parameterless_type_θ,
                                                 strategy) for (_loss,lb,ub) in zip(_bc_loss_functions,blbs,bubs)]    # QUADTRATURE
# bc_loss_functions = [NeuralPDE.get_loss_function(_loss,_set,
#                                                  eltypeθ, parameterless_type_θ,
#                                                  strategy) for (_loss,_set) in zip(_bc_loss_functions,train_bound_set)] #GRIDTRAINING

map(l->l(flat_initθ) ,bc_loss_functions)

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]

function u_loss(θ)
        return sum(abs2, phi[1](coords,θ[sep[1]]) .- data[1])/(N*T)
end

function v_loss(θ)
        return sum(abs2, phi[2](coords,θ[sep[2]]) .- data[2])/(N*T)
end

function p_loss(θ)
        return sum(abs2, phi[3](coords,θ[sep[3]]) .- data[3])/(N*T)
end

# LOSS FUNCTION
# data_weights = [10 .* length(pde_loss_functions);1 .* length(bc_loss_functions);1;1]
loss_functions = [pde_loss_functions; bc_loss_functions; u_loss; v_loss]#; p_loss]    

function loss_function(θ,p)
        sum(map(l->l(θ) ,loss_functions))
end

wghts = []
loss_list = []
pde_loss = []
bc_loss = []
data_loss = []
# Add timer
cb_ = function (p,l)
        iter = length(wghts) + 1
        println("$iter: Loss is: $l")
        push!(loss_list, l)
        push!(wghts,p)
        push!(pde_loss,[loss_functions[i](p) for i in 1:length(eqs)])
        push!(bc_loss,[loss_functions[i](p) for i in length(eqs)+1:length(eqs)+length(bcs)])      
        push!(data_loss,[loss_functions[i](p) for i in length(eqs)+length(bcs)+1:length(loss_functions)])
        #println("loss: ", l , "losses: ", map(l -> l(p), loss_functions))
        #println()
        return false
        # If condition return true
end

cb = function (p,l)
        println("Current loss is: $l")
        return false
end

# optimizer
f_ = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)

res = GalacticOptim.solve(prob,ADAM(1e-3); cb = cb_, maxiters=40)
# println("Stage 2")
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb_, maxiters=5000, maxtime=14400)

minimizers_ = [res.minimizer[s] for s in sep]

using Plots

np = 200

x_data = reshape(X_star[:,1],(nx,ny))[:,1]
y_data = reshape(X_star[:,2],(nx,ny))[1,:]

xs, ys, ts = [LinRange(infimum(d.domain),supremum(d.domain),np) for d in domains]
data_predict  = [[[phi[i]([x,y,t],minimizers_[i])[1] for x in xs for y in ys] for t in ts] for i in 1:length(depvars)]
u_predict = [data_predict[1][i] for i in 1:np]
v_predict = [data_predict[2][i] for i in 1:np]
p_predict = [data_predict[3][i] for i in 1:np]

function u_gif()
        anim = @animate for i ∈ 1:length(ts)
                p1 = plot(x_data, y_data, UU[:,i],linetype=:contourf,title = "Actual Streamwise Velocity field",c=:haline,clim=(-0.2,1.3));
                p2 = plot(xs, ys, data_predict[1][i],linetype=:contourf,title = "Predicted Streamwise Velocity field",c=:haline,clim=(-0.2,1.3));
                plot(p1,p2,layout = (2,1));
        end
        gif(anim, "Streamwise_evolution.gif", fps = 60)
end

function v_gif()
        anim = @animate for i ∈ 1:length(ts)
                p1 = plot(x_data, y_data, VV[:,i],linetype=:contourf,title = "Actual Transverse Velocity field",c=:haline,clim=(-0.2,1.3));
                p2 = plot(xs, ys, data_predict[2][i],linetype=:contourf,title = "Predicted Transverse Velocity field",c=:haline,clim=(-0.2,1.3));
                plot(p1,p2,layout = (2,1));
        end
        gif(anim, "Transverse_evolution.gif", fps = 60)
end

function p_gif()
        anim = @animate for i ∈ 1:length(ts)
                p1 = plot(x_data, y_data, PP[:,i],linetype=:contourf,title = "Actual Pressure field",c=:haline,clim=(-0.2,1.3));
                p2 = plot(xs, ys, data_predict[3][i],linetype=:contourf,title = "Predicted Pressure field",c=:haline,clim=(-0.2,1.3));
                plot(p1,p2,layout = (2,1));
        end
        gif(anim, "Pressure_evolution.gif", fps = 60)
end
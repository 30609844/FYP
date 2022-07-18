using NeuralPDE, Cubature, Quadrature, Flux, ModelingToolkit
using GalacticOptim, Optim, DiffEqFlux, Noise, SpecialFunctions
using JLD2, MAT, LinearAlgebra, Statistics, Random
import ModelingToolkit: Interval, infimum, supremum

## TEST CASE: Cavity driven flow at re = 100
case_title = "Cavity_driven_flow_re100"
##

@parameters t, x, y
@variables u(..), v(..), p(..), T(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

## parameters
Re = 100; Pe = 100;
ρ = 1.0; μ = 1/Re; α = 1/Pe; g = 0.0;

## 2D NSE + Heat eqn
eqs = [
        Dx(u(t,x,y)) + Dy(v(t,x,y)) ~ 0,

        ρ*(Dt(u(t,x,y)) + u(t,x,y)*Dx(u(t,x,y)) + v(t,x,y)*Dy(u(t,x,y))) ~ 
        -Dx(p(t,x,y)) + μ*(Dxx(u(t,x,y)) + Dyy(u(t,x,y))),

        ρ*(Dt(v(t,x,y)) + u(t,x,y)*Dx(v(t,x,y)) + v(t,x,y)*Dy(v(t,x,y))) ~ 
        -Dy(p(t,x,y)) + μ*(Dxx(v(t,x,y)) + Dyy(v(t,x,y))),

        Dt(T(t,x,y)) + u(t,x,y)*Dx(T(t,x,y)) + v(t,x,y)*Dy(T(t,x,y)) ~ 
        α*(Dxx(T(t,x,y))+Dyy(T(t,x,y)))
]

indvars = [t,x,y]
depvars = [u(t,x,y),v(t,x,y),p(t,x,y),T(t,x,y)]

## Space and time domains
tspan = [0,1]; xspan = [0,1]; yspan = [0,1]; 
domains = [
        t ∈ Interval(tspan[1],tspan[end]),
        x ∈ Interval(xspan[1],xspan[end]),
        y ∈ Interval(yspan[1],yspan[end])
        ]

## Training data
data_mat = matread("cavity_variables_100.mat")
udata = data_mat["usave"]
vdata = data_mat["vsave"]
Tdata = data_mat["tsave"]

n = size(udata)[1]
m = size(udata)[2]

xcords = repeat(LinRange(0,xspan[end],n),1,m)
ycords = transpose(repeat(LinRange(0,yspan[end],m),1,n))

stencil = zeros(n,m)
stencil[1:8:end,1:8:end] .= 1;
stencil[1:8,end-7:end] .= 1;
stencil[end-7:end,end-7:end] .= 1

udata_vec = udata[Bool.(stencil)]
vdata_vec = vdata[Bool.(stencil)]
Tdata_vec = Tdata[Bool.(stencil)]
xvec = xcords[Bool.(stencil)]
yvec = ycords[Bool.(stencil)]

nt = 5
data = reshape.([repeat(udata_vec,nt),repeat(vdata_vec,nt),repeat(Tdata_vec,nt)],1,length(udata_vec)*nt)
tvec = repeat(collect(LinRange(tspan[1],tspan[end],nt)),inner = length(xvec))
coords = zeros(length(indvars),nt*length(xvec))
coords = hcat(repeat(xvec,nt),repeat(yvec,nt),tvec)'

bcs = [
        u(t,0,y) ~ 0,
        u(t,1,y) ~ 0,
        u(t,x,1) ~ 1,
        u(t,x,0) ~ 0,
        v(t,0,y) ~ 0,
        v(t,1,y) ~ 0,
        v(t,x,1) ~ 0,
        v(t,x,0) ~ 0,
        T(t,x,0) ~ 1,
        T(t,x,1) ~ 0,
        Dx(T(t,0,y)) ~ 0,
        Dx(T(t,1,y)) ~ 0,
        p(t,0,0) ~ 0
        ]
# bcs = []

@named pde_system = PDESystem(eqs,bcs,domains,indvars,depvars)

## Neural Network
input_ = length(indvars)
dx = [tspan[end]/nt,1/n,1/m]  
act_func = Flux.tanh_fast
n = 60
chain = [FastChain(FastDense(input_,n,act_func),FastDense(n,n,act_func),FastDense(n,1)) for _ in 1:length(eqs)]

strategy = GridTraining(dx)
initθ = map(c -> Float32.(c), DiffEqFlux.initial_params.(chain))
flat_initθ = reduce(vcat,initθ)

eltypeθ = eltype(initθ[1])
parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])

# discretization = PhysicsInformedNN(chain, strategy, init_params= initθ)

phi = NeuralPDE.get_phi.(chain,parameterless_type_θ)

map(phi_ -> phi_(rand(Float32,length(indvars),10), flat_initθ),phi)
# prob = NeuralPDE.discretize(pde_system,discretization)

derivative = NeuralPDE.get_numeric_derivative()

## Low Level API

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

function u_loss(θ)
        return sum(abs2, phi[1](coords,θ[sep[1]]) .- data[1])/np
end

function v_loss(θ)
        return sum(abs2, phi[2](coords,θ[sep[2]]) .- data[2])/np
end

function p_loss(θ)
        return sum(abs2, phi[3](coords,θ[sep[3]]) .- data[3])/np
end

function T_loss(θ)
        return sum(abs2, phi[4](coords,θ[sep[4]]) .- data[3])/np
end

## LOSS FUNCTION

loss_functions = [pde_loss_functions; bc_loss_functions; u_loss; v_loss; T_loss]    

pde_weights = [1,1,1,1]
bc_weights = 100*[1,1,1,1,1,1,1,1,1,1,1,1,1]
data_weights = [1,1,1]
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
res = GalacticOptim.solve(prob,ADAM(1e-2); cb = cb_, maxiters=500)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,LBFGS(); cb = cb_, maxiters=500)

minimizers_ = [res.minimizer[s] for s in sep]
function save_weights()
        save_object(case_title*"_$n"*".jld2",minimizers_)
        println("OPTIMISED WEIGHTS SAVED")
end
# save_weights()
pde_loss = [[pde_loss[i][j] for i in 1:length(pde_loss)] for j in 1:length(eqs)]
bc_loss = [[bc_loss[i][j] for i in 1:length(bc_loss)] for j in 1:length(bcs)]
data_loss = [[data_loss[i][end] for i in 1:length(data_loss)]]

using Plots
gr()

ts = LinRange(tspan[1],tspan[end],50)

function plot_loss()
        plot(loss_list, yaxis=:log,title = "ADAM cascade" , xlabel = "Iterations", ylabel = "loss")
        loss_type = ["PDE","BC","data"]; loss_amt = [sum(pde_loss[end]),sum(bc_loss[end]),sum(data_loss[end])]
        pie(loss_type,loss_amt,title = "Source of losses at last iter.",autopct="%1.1f%%",l=0.5)
end

function plot_u()
    xs = LinRange(xspan[1],xspan[end],257)
    ys = LinRange(yspan[1],yspan[end],257)
        anim = @animate for i ∈ 1:length(ts)
            t = round(ts[i],digits=2)
            u_predict = reshape([Array(phi[1]([ts[i],x,y],minimizers_[1]))[1] for x in xs for y in ys],(length(xs),length(ys)))
            p1 = plot(xs,ys,u_predict',linetype=:contourf,clim=(-0.25,1))
            # p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            # vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
            # p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
            # p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            plot(p1)
    end
    gif(anim, "u_velocity_field_Re100.gif", fps = 50)
end

function plot_v()
    xs = LinRange(xspan[1],xspan[end],100)
    ys = LinRange(yspan[1],yspan[end],100)
        anim = @animate for i ∈ 1:length(ts)
            t = round(ts[i],digits=2)
            v_predict = reshape([Array(phi[2]([ts[i],x,y],minimizers_[2]))[1] for x in xs for y in ys],(length(xs),length(ys)))
            p1 = plot(xs,ys,v_predict',linetype=:contourf,clim=(-0.5,0.5))
            # p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            # vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
            # p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
            # p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            plot(p1)
    end
    gif(anim, "v_velocity_field_Re100.gif", fps = 50)
end

function plot_T()
    xs = LinRange(xspan[1],xspan[end],100)
    ys = LinRange(yspan[1],yspan[end],100)
        anim = @animate for i ∈ 1:length(ts)
            t = round(ts[i],digits=2)
            T_predict = reshape([Array(phi[4]([ts[i],x,y],minimizers_[4]))[1] for x in xs for y in ys],(length(xs),length(ys)))
            p1 = plot(xs,ys,T_predict,linetype=:contourf,clim=(0.0,1))
            # p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            # vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
            # p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
            # p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            plot(p1)
    end
    gif(anim, "Temperature_field_Re100.gif", fps = 50)
end

function plot_p()
    xs = LinRange(xspan[1],xspan[end],100)
    ys = LinRange(yspan[1],yspan[end],100)
        anim = @animate for i ∈ 1:length(ts)
            t = round(ts[i],digits=2)
            p_predict = reshape([Array(phi[3]([ts[i],x,y],minimizers_[3]))[1] for x in xs for y in ys],(length(xs),length(ys)))
            p1 = plot(xs,ys,p_predict,linetype=:contourf)
            # p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            # vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
            # p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
            # p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
            plot(p1)
    end
    gif(anim, "Pressure_field_Re100.gif", fps = 50)
end


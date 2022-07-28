using NeuralPDE, Cubature, Quadrature, Flux, ModelingToolkit
using GalacticOptim, Optim, DiffEqFlux, Noise, SpecialFunctions
using JLD2, MAT, LinearAlgebra, Statistics, Random
import ModelingToolkit: Interval, infimum, supremum

## TEST CASE: Steady_Cavity driven flow at re = 100
case_title = "Steady_Cavity_driven_flow_re100"
##

@parameters x, y
@variables u(..), v(..), p(..), T(..)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# parameters
Re = 100; Pe = 100;
ρ = 1.0; μ = 1/Re; k = 1/Pe; g = 0.0;

## 2D NSE + Heat eqn (steady)
eqs = [
        Dx(u(x,y)) + Dy(v(x,y)) ~ 0,

        ρ*(u(x,y)*Dx(u(x,y)) + v(x,y)*Dy(u(x,y))) ~ 
        -Dx(p(x,y)) + μ*(Dxx(u(x,y)) + Dyy(u(x,y))),

        ρ*(u(x,y)*Dx(v(x,y)) + v(x,y)*Dy(v(x,y))) ~ 
        -Dy(p(x,y)) + μ*(Dxx(v(x,y)) + Dyy(v(x,y))),

        u(x,y)*Dx(T(x,y)) + v(x,y)*Dy(T(x,y)) ~ k*(Dxx(T(x,y))+Dyy(T(x,y)))
]

indvars = [x,y]
depvars = [u(x,y),v(x,y),p(x,y),T(x,y)]

# Space and time domains
xspan = [0,1]; yspan = [0,1]; 
domains = [
        x ∈ Interval(xspan[1],xspan[end]),
        y ∈ Interval(yspan[1],yspan[end])
        ]

# Training data

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
stencil[1:4:29,end-28:4:end] .= 1;
stencil[end-28:4:end,end-28:4:end] .= 1

udata_vec = udata[Bool.(stencil)]
vdata_vec = vdata[Bool.(stencil)]
Tdata_vec = Tdata[Bool.(stencil)]
xvec = xcords[Bool.(stencil)]
yvec = ycords[Bool.(stencil)]

data = reshape.([udata_vec,vdata_vec,Tdata_vec],1,length(udata_vec))
coords = zeros(length(indvars),length(xvec))
coords = hcat(xvec,yvec)'
np = length(udata_vec)

bcs = [
        u(0,y) ~ 0,
        u(1,y) ~ 0,
        u(x,1) ~ 1,
        u(x,0) ~ 0,
        v(0,y) ~ 0,
        v(1,y) ~ 0,
        v(x,1) ~ 0,
        v(x,0) ~ 0,
        T(x,0) ~ 1,
        T(x,1) ~ 0,
        Dx(T(0,y)) ~ 0,
        Dx(T(1,y)) ~ 0,
        p(0,0) ~ 0
        ]
# bcs = []

@named pde_system = PDESystem(eqs,bcs,domains,indvars,depvars)

# Neural Network
input_ = length(indvars)
dx = [1/33,1/33]  
act_func = Flux.tanh_fast
n = 60
chain = [FastChain(FastDense(input_,n,act_func),
                FastDense(n,n,act_func),
                FastDense(n,n,act_func),
                FastDense(n,n,act_func),
                FastDense(n,1)) for _ in 1:length(depvars)
] 
# n = 60
# chain = [FastChain(FastDense(input_,n,act_func),FastDense(n,n,act_func),FastDense(n,1)) for _ in 1:length(eqs)]

strategy = GridTraining(dx)
initθ = map(c -> Float32.(c), DiffEqFlux.initial_params.(chain))
# initθ = load(case_title*"_$n"*".jld2")["single_stored_object"]
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

# LOSS FUNCTION

loss_functions = [pde_loss_functions; bc_loss_functions; u_loss; v_loss; T_loss]    

pde_weights = [1,1,1,1]
bc_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1]
data_weights = [1,1,1000]
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
res = GalacticOptim.solve(prob,ADAM(1e-2); cb = cb_, maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,LBFGS(); cb = cb_, maxiters=3000)

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


function plot_loss()
        plot(loss_list, yaxis=:log,title = "ADAM cascade" , xlabel = "Iterations", ylabel = "loss")
        loss_type = ["PDE","BC","data"]; loss_amt = [sum(pde_loss[end]),sum(bc_loss[end]),sum(data_loss[end])]
        pie(loss_type,loss_amt,title = "Source of losses at last iter.",autopct="%1.1f%%",l=0.5)
end

function plot_u()
        xs = LinRange(xspan[1],xspan[end],257)
        ys = LinRange(yspan[1],yspan[end],257)
        u_predict = reshape([Array(phi[1]([x,y],minimizers_[1]))[1] for x in xs for y in ys],(length(xs),length(ys)))
        p1 = plot(xs,ys,u_predict,linetype=:contourf,clim=(-0.25,1))
        p2 = plot(xs,ys,udata',linetype=:contourf,clim=(-0.25,1))
        plot(p1,p2)
end

function plot_v()
        xs = LinRange(xspan[1],xspan[end],257)
        ys = LinRange(yspan[1],yspan[end],257)
        v_predict = reshape([Array(phi[2]([x,y],minimizers_[2]))[1] for x in xs for y in ys],(length(xs),length(ys)))
        p1 = plot(xs,ys,v_predict,linetype=:contourf,clim=(-0.7,0.5))
        p2 = plot(xs,ys,vdata',linetype=:contourf,clim=(-0.7,0.5))
        plot(p1,p2)
end

function plot_T()
        xs = LinRange(xspan[1],xspan[end],257)
        ys = LinRange(yspan[1],yspan[end],257)
        T_predict = reshape([Array(phi[4]([x,y],minimizers_[4]))[1] for x in xs for y in ys],(length(xs),length(ys)))
        p1 = plot(xs,ys,T_predict,linetype=:contourf,clim=(0,1))
        p2 = plot(xs,ys,Tdata',linetype=:contourf,clim=(0,1),title = "")
        plot(p1,p2)
end

function plot_p()
        xs = LinRange(xspan[1],xspan[end],257)
        ys = LinRange(yspan[1],yspan[end],257)
        p_predict = reshape([Array(phi[3]([x,y],minimizers_[3]))[1] for x in xs for y in ys],(length(xs),length(ys)))
        p1 = plot(xs,ys,p_predict,linetype=:contourf,clim=(0,1))
        plot(p1)
end
        # xs = LinRange(xspan[1],xspan[end],257)
        # ys = LinRange(yspan[1],yspan[end],257)
        # T_predict = reshape([Array(phi[4]([x,y],minimizers_[4]))[1] for x in xs for y in ys],(length(xs),length(ys)))
        # p1 = plot(xs,ys,T_predict,linetype=:contourf,clim=(0,1))
        # p2 = plot(xs,ys,Tdata',linetype=:contourf,clim=(0,1))
        # plot(p1,p2)


# function plot_polar_vz(z)
#     rs = LinRange(rspan[1],rspan[end],30)
#     ϕs = LinRange(0,2pi,30)
#         anim = @animate for i ∈ 1:length(ts)
#                 t = round(ts[i],digits=2)
#                 vz_flat = hcat([analytic_sol_func_vz(ts[i],r,ϕ,z) for r in rs for ϕ in ϕs])
#                 p1 = heatmap(ϕs,rs/R,vz_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
#                 p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
#                 vz_predict_flat = hcat([Array(phi[3]([ts[i],r,ϕ,z],minimizers_[3]))[1] for r in rs for ϕ in ϕs])
#                 p2 = heatmap(ϕs,rs/R,vz_predict_flat/u0,projection = :polar,clim=(-1.2,1.2),color=:bluesreds)
#                 p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
#                 plot(p1,p2)
#         end
#         gif(anim, "Velocity_polar_evolution.gif", fps = 50)
# end

# function plot_polar_p(z)
#         rs = LinRange(rspan[1],rspan[end],30)
#         ϕs = LinRange(0,2pi,30)
#             anim = @animate for i ∈ 1:length(ts)
#                     t = round(ts[i],digits=2)
#                     p_flat = hcat([analytic_sol_func_p(ts[i],r,ϕ,z) for r in rs for ϕ in ϕs])
#                     p1 = heatmap(ϕs,rs/R,p_flat,projection = :polar,clim=(1,3))
#                     p1 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
#                     p_centre_fluc = Array(phi[4]([ts[1],0,0,L/2],minimizers_[4]))[1] - analytic_sol_func_p(ts[i],0,0,L/2)
#                     p_predict_flat = hcat([Array(phi[4]([ts[i],r,ϕ,z],minimizers_[4]))[1] for r in rs for ϕ in ϕs]) .- p_centre_fluc
#                     p2 = heatmap(ϕs,rs/R,p_predict_flat,projection = :polar,clim=(1,3))
#                     p2 = plot!(ϕr,rr/R,projection = :polar,linetype = :scatter,markershape = :xcross,markersize = 8, legend = false);
#                     plot(p1,p2)
#             end
#             gif(anim, "Pressure_polar_evolution.gif", fps = 50)
# end

# function plot_v(z)
#         anim = @animate for i ∈ 1:length(ts)
#                 t = round(ts[i],digits=2)
#                 rs = LinRange(rspan[1],rspan[end],20)
#                 vz_predict =    [[Array(phi[3]([ts[i],r,Float64(pi),z],minimizers_[3]))[1] for r in rs[end:-1:1]];
#                                 [Array(phi[3]([ts[i],r,0,z],minimizers_[3]))[1] for r in rs]]
#                 vz_analytic =   [[analytic_sol_func_vz(ts[i],r,Float64(pi),z) for r in rs[end:-1:1]];
#                                 [analytic_sol_func_vz(ts[i],r,0,z) for r in rs]]
#                 v_diff = abs.(vz_predict.-vz_analytic)               
#                 rs = [-rs[end:-1:1];rs]
#                 p1 = plot(vz_predict/u0,rs/R, linestyle = :dash, lw = 4, xlims = (-1.2,1.2), color = "red", label = "Predicted");
#                 p1 = plot!(vz_analytic/u0,rs/R, color = "blue", title = "Velocity profile at t = $t s", label = "Analytic");
#                 p2 = plot(v_diff/u0,rs/R, xlims=(0,0.5), title = "Velocity Error");
#                 plot(p1,p2)
#         end
#         gif(anim, "Velocity_profile_evolution.gif", fps = 50)
# end

# function plot_p(r)
#         zs = LinRange(zspan[1],zspan[end],100)
#         anim = @animate for i ∈ 1:length(ts)
#                 t = round(ts[i],digits=2)
#                 p_predict = [Array(phi[4]([ts[i],r,0,z],minimizers_[4]))[1] for z in zs]
#                 p_analytic =  [analytic_sol_func_p(ts[i],r,0,z) for z in zs]
#                 p1 = plot(zs/L, p_predict .- (p_predict[50]-p_analytic[50]),linestyle = :dash, lw = 4, color = "red", title = "Predicted pressure at t = $t s", label = "Predicted");
#                 p1 = plot!(zs/L, p_analytic, color = "blue", title = "Centreline pressure", label = "Analytic", xlabel="z/L", ylabel="Pressure");
#                 plot(p1)
#         end
#         gif(anim, "Pressure gradient.gif", fps = 50)
# end


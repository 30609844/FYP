# SciML PINN technique for solving PDEs using NeuralPDE.jl package. https://neuralpde.sciml.ai/stable/
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Zygote, LaTeXStrings, QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using Plots, Printf, Quadrature,Cubature, QuadGK, Cuba, JLD2, LineSearches

# Define the independent and dependent variables, and the functional operators
@parameters x,r
@variables T(..),u(..)
Dr = Differential(r)
Dx = Differential(x)
Dxx = Differential(x)^2
Drr = Differential(r)^2

# # Constants for water at 20 deg C
rho = 988.21 # kg/m3
mu = 0.0010016 # kg/m-s 
nu = mu/rho
k = 0.598 # W/m-K
Cp = 4184 # J/kg-K
alpha = k/(rho*Cp)
D = 0.05
R = D/2
ReD = 1000
Um = (ReD*nu)/D
delP = (-Um*8*mu)/(R^2)
P = 2*pi*R
M = pi*R^2*rho*Um
Pr = 3.57
Tmi = 20+273
Qs = 200 # heat flux W/m2
Lh = D*0.056*ReD
Lt = D*0.043*Pr*ReD
Li = 0#round(Lt;digits=0)
Lo = 5+Li
L = 1

# Fully developed flow in channel with uniform flux
eqs = [alpha*(r*Drr(T(x,r))+ Dr(T(x,r))) ~ r*u(x,r)*Dx(T(x,r)), (r*Drr(u(x,r))+ Dr(u(x,r))) ~ (r*delP/mu)]

# BCs 
bcs = [u(x,R) ~ 0.0#Dr(T(x,0)) ~ 0, Dr(T(x,R)) ~ Qs/k]#Dr(T(x,0)) ~ 0, Dr(T(x,R)) ~ Qs/k]#, Dx(T(x,r)) ~ ((2*Qs*alpha)/(R*k*Um))]
        ]

# Spatial domains
domains = [x ∈ Interval(Li,Lo),r ∈ Interval(0,R)]

# # Set up the spatial arrays for temperature input
dx = 0.2 
dr = 0.001 # spatial intervals at which the solution is plotted
xs, rs = [infimum(d.domain):dx:supremum(d.domain) for (dx,d) in zip([dx,dr],domains)]

# Analytical temperature solution
Ti(x,r) = Tmi + ((4*Qs/(k*R))*(((r^2)/4)- ((r^4)/(16*(R^2))))) - ((7*R*Qs)/(24*k)) + (((2*Qs*alpha)/(R*k*Um))*x)
Ti_xr = reshape([Ti(x,r) for x in xs for r in rs], (length(rs),length(xs)))
Ts(x) = Tmi + ((11*R*Qs)/(24*k)) + (((2*Qs*alpha)/(R*k*Um))*x)
Ts_x = reshape([Ts(x) for x in xs], (length(xs),1))
Tm = Tmi .+ (((2*Qs*alpha)/(R*k*Um)).*xs)
ua(t) = 2*Um*(1-((t^2)/(R^2)))
Ti(t) = Tmi + ((4*Qs/(k*R))*(((t^2)/4)- ((t^4)/(16*(R^2))))) - ((7*R*Qs)/(24*k)) + (((2*Qs*alpha)/(R*k*Um))*xs[1])
uinte,erru = quadgk(t -> (ua(t)*t), 0, R, rtol = 1e-5)
Tinte,errT = quadgk(t -> (ua(t)*Ti(t)*t), 0, R, rtol = 1e-5)
ua_r = reshape([ua(t) for t in rs], (length(rs),1))

# Arrays (x, r, T) for temperature input data
xd = [xs[i] for i in 1:6:length(xs)]
rd = [rs[i] for i in 1:6:length(rs)]

xd1 = [repeat([i],length(rd)) for i in xd]
xd1 = cat(xd1...; dims=1)
xd1 = reshape(xd1,(1,length(xd1)))

rd1 = repeat(rd,length(xd))
rd1 = reshape(rd1,(1,length(rd1)))
xrd1 = vcat(xd1, rd1)

Ti_xr1 = reshape([Ti(x,r) for x in xd for r in rd], (length(rd),length(xd)))
Ti_xr1 = cat(Ti_xr1...;dims=1)
Ti_xr1 = trunc.(reshape(Ti_xr1,(1,length(Ti_xr1))))

# Neural network
densein = 50 # no. of neurons
act_func = Flux.σ # activation function
# chain1 = FastChain(FastDense(2,densein,act_func),
#             FastDense(densein,densein,act_func),
#             FastDense(densein,1)) # network of layers
# chain2 = FastChain(FastDense(1,densein,act_func),
#             FastDense(densein,densein,act_func),
#             FastDense(densein,1))
chain1 = FastChain(FastDense(2,densein,act_func,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32),
            FastDense(densein,densein,act_func,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32),
                FastDense(densein,1,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32)) # network of layers
chain2 = FastChain(FastDense(1,densein,act_func,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32),
    FastDense(densein,densein,act_func,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32),
              FastDense(densein,1,bias=true,initW=Flux.glorot_normal,initb = Flux.zeros32))
chain = [chain1,chain2]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
# initθ = load("optim-weights-fullequ-mod-14.jld","minim-14")
flat_initθ = (reduce(vcat,initθ))
eltypeθ = eltype(flat_initθ) # determine the type of the elements
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ) # generate a parameter-less version of a type
phi = NeuralPDE.get_phi.(chain,parameterless_type_θ) # trial solution of the NN - phi([t,x],θ)
strategy = NeuralPDE.QuadratureTraining()
phi[1](rand(2,10), flat_initθ)
phi[2](rand(1,10), flat_initθ)
phi[1](xrd1, flat_initθ)

# # Low-level framework begins from here
indvars = [x,r] # define independent variables
depvars = [T(x,r),u(r)]#,Dxu(x,y),Dyu(x,y)] # define dependent variables
dim = length(domains)
varbls = NeuralPDE.get_vars(indvars, depvars) # get all the variables in a dictionary
pde_varbls = NeuralPDE.get_variables(eqs,indvars,depvars) # get the variables in the pde
bcs_varbls = NeuralPDE.get_variables(bcs,indvars,depvars) # get the variables in the BCs
bcs_varbls2 = NeuralPDE.get_argument(bcs,indvars,depvars)

derivative = NeuralPDE.get_numeric_derivative() # calculate the derivative (based from Zygote)
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative) # calculate the integral
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,eqs,bcs,eltypeθ,indvars,depvars,strategy)
plbs,pubs = pde_bounds
blbs,bubs = bcs_bounds

symbol_pde_loss = [NeuralPDE.build_symbolic_loss_function(eq,indvars,depvars,varbls,phi,derivative,integral,chain,initθ,strategy)
                    for eq in eqs]
# build PDE loss function
build_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,integral,
                        chain,initθ,strategy,bc_indvars = pde_vars) for (eq, pde_vars) in zip(eqs,pde_varbls)]
map(loss_f -> loss_f(rand(3,10), flat_initθ),build_pde_loss_functions)
# build_pde_loss_functions[1](rand(2,10), flat_initθ)
# build_pde_loss_functions[2](rand(1,10), flat_initθ)
# Obtain the PDE loss function
get_pde_loss_functions = [NeuralPDE.get_loss_function(_loss_funcs,lb,ub,eltypeθ,parameterless_type_θ,
                        strategy) for (_loss_funcs,lb,ub) in zip(build_pde_loss_functions,plbs,pubs)]
temp3 = map(l->l(flat_initθ) ,get_pde_loss_functions)

# build BCs loss function
build_bcs_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,
                            chain,initθ,strategy,bc_indvars = bcs_vars) for (bc,bcs_vars) in zip(bcs,bcs_varbls2)]
temp4 = map(loss_f -> loss_f(rand(1,10), flat_initθ),build_bcs_loss_functions)
# Obtain the BCs loss function
get_bc_loss_functions = [NeuralPDE.get_loss_function(_loss_funcs,lb,ub,eltypeθ,parameterless_type_θ,
                        strategy) for (_loss_funcs,lb,ub) in zip(build_bcs_loss_functions,blbs,bubs)]
temp5 = map(l->l(flat_initθ) ,get_bc_loss_functions)

function build_additional_loss(phi, θ, p)
    return 10*(sum(abs2, phi[1](xrd1, θ) .-Ti_xr1)/length(Ti_xr1))
end
function get_additional_loss(θ)
    return build_additional_loss(phi, θ, 1)
end

# Put all the loss functions in a vector array and sum them together
loss_funcs = [get_pde_loss_functions; get_bc_loss_functions;get_additional_loss]

# # The total loss function
function loss_func_(θ,p)
    sum(map(l->l(θ) ,loss_funcs))
end

# Frame the optimization problem
f = OptimizationFunction(loss_func_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, flat_initθ)

# # Get the weights of the NN, and values of the total, pde, data, and BCs loss functions at each iteration
wghts = []
loss_list = []
pde_loss = []
bc_loss = []
add_loss = []
cb_ = function (p,l)
       println("Current loss is: $l")
       push!(loss_list, l)
       push!(wghts,p)
       push!(pde_loss,[loss_funcs[i](p) for i in 1:length(eqs)])
       push!(bc_loss,[loss_funcs[i](p) for i in length(eqs)+1:length(eqs)+length(bcs)])
       push!(add_loss,[loss_funcs[i](p) for i in length(eqs)+length(bcs)+1:length(loss_funcs)])
       return false
end

# # # Choose the optimizer and solve the optimization problem by minimizing the total loss function
# res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb_, maxiters=200)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb_, maxiters=150)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.001); cb = cb_, maxiters=300)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.0001); cb = cb_, maxiters=20000)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.000001); cb = cb_, maxiters=1000)

# res = GalacticOptim.solve(prob, Optim.LBFGS(); cb = cb_,allow_f_increases = true, maxiters=100)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.0000001); cb = cb_, maxiters=100)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, Optim.LBFGS(); cb = cb_,allow_f_increases = true, maxiters=50)

res = GalacticOptim.solve(prob, Optim.LBFGS(alphaguess = LineSearches.InitialHagerZhang()); cb = cb_, allow_f_increases = true,maxiters=1000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, Optim.BFGS(alphaguess = LineSearches.InitialHagerZhang()); cb = cb_,allow_f_increases = true, maxiters=1000) 

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep] #initθ
u_predict2  = [phi[2]([r],minimizers_[2])[1] for r in rs] 
u_predict1  = [phi[1]([x,r],minimizers_[1])[1] for x in xs for r in rs] 
u_predict22  = reshape(u_predict2, (length(rs),1))
u_predict11  = reshape(u_predict1, (length(rs),length(xs)))
uaa = 2*Um*(1.0.-((rs.^2)/(R^2)))

iter = 1:length(loss_list)
plot(iter, loss_list,ylims=[1e+0, 1e+2],xlims=[2e+4, 2.5e+4])

xs_nd = (xs.-xs[1])/R
rs_nd = rs/R
plot(uaa/Um,rs_nd,xlabel=L"u/U", ylabel = L"r/R",label= L"u_{analytical}")
plot!(u_predict22/Um,rs_nd,xlabel=L"u/U", ylabel = L"r/R",label= L"u_{predicted}")
# savefig("U_Pipe-constant-flux-temperature-fullequ-mod-9.png")

xi = 20
plot((Ti_xr[:,xi].-Ti_xr[end,xi])/(Tmi-Ti_xr[end,xi]),rs_nd,xlabel=L"T/T_{mi}", ylabel = L"r/R",label= L"T_{analytical}")
plot!((u_predict11[:,xi].-u_predict11[end,xi])/(Tmi-u_predict11[end,xi]),rs_nd,xlabel=L"T/T_{mi}", ylabel = L"r/R",label= L"T_{predicted}")
# savefig("T_Pipe-constant-flux-temperature-fullequ-mod-9.png")

# plot((uaa-u_predict22)/Um,rs_nd,xlabel=L"", ylabel = L"r/R",label= L"u")
# plot((Ti_xr[:,10]./u_predict11[:,10]),rs_nd,xlabel=L"(T_{a}-T_{p})/(T_{mi}),(u_{a}-u_{p})/(U)", ylabel = L"r/R",label= L"T")
# savefig("Error_T-u_Pipe-constant-flux-temperature-fullequ-mod.png")

# save("optim-weights-fullequ-mod-14.jld","minim-14",minimizers_)

# # --------# Plots from here
# Plot sparse data with analytical temperature
# xs_nd = (xs.-xs[1])/R
# rs_nd = rs/R
# plot(xs_nd,rs_nd,Ti_xr,linetype=:contourf,xlabel=L"x/R", ylabel = L"r/R", title= L"T_{analytical} (K)",
#     ylims=[0, 1], xlims=[0, 200])
# xd_nd = (xd1.-xd1[1])/R
# rd_nd = rd1/R
# plot!(xd_nd,rd_nd,seriestype = :scatter,markercolor=:cyan,markershape=:x,label= "")
# savefig("Sparse-data_Pipe-constant-flux-temperature-fullequ-mod.png")

# # Plot predicted temperature
# plot(xs_nd,rs_nd,u_predict1,linetype=:contourf,xlabel=L"x/R", ylabel = L"r/R", title= L"T_{predicted} (K)",
#     ylims=[0, 1], xlims=[0, 200])
# savefig("Predicted-temperature_Pipe-constant-flux-temperature-fullequ.png")

# # Plot analytical and predicted temperatures
# xll = 290
# xul = 300
# plot(Ti_xr[:,6],rs_nd,shape=:circle,markercolor=:black,linecolor=:black,linestyle=:solid,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Analytical-x/R=40",legend=:topleft)
# plot!(u_predict11[:,6],rs_nd,shape=:circle,markercolor=:black,linecolor=:black,linestyle=:dash,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Predicted-x/R=40",legend=:topleft)
# # plot!(Ti_xr[:,11],rs_nd,shape=:circle,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Analytical-x/R=80",legend=:topleft)
# # plot!(u_predict11[:,11],rs_nd,shape=:circle,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Predicted-x/R=80",legend=:topleft)
# plot!(Ti_xr[:,21],rs_nd,shape=:circle,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Analytical-x/R=160",legend=:topleft)
# plot!(u_predict11[:,21],rs_nd,shape=:circle,xlabel=L"T", ylabel = L"r/R", ylims=[0, 1], xlims=[xll, xul],label=L"Predicted-x/R=160",legend=:topleft)
# savefig("Predicted-vs-analytical-temperature_Pipe-constant-flux-temperature-fullequ.png")

# Plot losses vs iterations
# iter = 1:length(loss_list)
# # plot(iter, loss_list,ylims=[0, 1e+6])
# plot(iter, loss_list,ylims=[8e+5, 1e+6])
# ytics = [1e+6,1e+4,1e+2,1e+0,1e-2]
# Losses = [loss_list,pde_loss[1],pde_loss[2],bc_loss,add_loss]
# Losses_names = ["Total loss", "T-PDE loss", "u-PDE loss", "BC loss","Data loss"]
# plot()
# plot!(iter, Losses[1], yaxis = (:log10, (1e-2,1e+6)), xlabel="Iterations", 
# ylabel = "Loss", xlims=[0, 2000], label=Losses_names[1],yticks=ytics)
# for i = 2:length(Losses)
#     plot!(iter, Losses[i], yaxis = (:log10, (1e-2,1e+6)), xlabel="Iterations", ylabel = "Loss", xlims=[0, 2000], label=Losses_names[i],yticks=ytics) |>display
# end
# savefig("Loss-vs-iters_Pipe-constant-flux-temperature-fullequ-mod.png")

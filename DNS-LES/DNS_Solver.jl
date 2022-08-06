##
# Created 31/07/22 

# @author: Kevin Liu

# DNS solver for 2D decaying homegenous isotropic turbulence problem for 
# cartesian periodic domain [0,2pi] X [0,2pi].
# Discretized uniformly in x and y direction
# The solver uses pseudo-spectral method for solving 2D incompressible NSE
# in vorticity-streamfunction formulation. The solver uses 
# hybrid third-order Runge-Kutta implicit Crank-Nicolson scheme for time integration. 

##
using Printf
println(string(Threads.nthreads())*" THREADS")
using FFTW
FFTW.set_num_threads(Threads.nthreads())
using Plots, DelimitedFiles
gr()

#%%
function exact_tgv(nx,ny,t,re)
    
    
    # compute exact solution for 2D TGV problem
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # t : time at which the exact solution is to be computed
    # re : Reynolds number
    
    # Output
    # ------
    # ue : exact solution for TGV problem
    
    
    ue = Array{Float64}(undef,nx+1,ny+1)
    x =LinRange(0.0,2.0*pi,nx+1)
    y = LinRange(0.0,2.0*pi,ny+1)
    
    nq = 4.0
    ue = @. 2.0*nq*cos(nq*x)*cos(nq*y)'*exp(-2.0*nq^2*t/re)

    return ue
end
#%%
function tgv_2D_ic(nx,ny)
    
    
    # compute initial condition for 2D TGV problem
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    
    # Output
    # ------
    # w : initial condiition for vorticity for TGV problem
    
    
    w = Array{Float64}(undef,nx+1,ny+1)
    nq = 4.0
    x = LinRange(0.0,2.0*pi,nx+1)
    y = LinRange(0.0,2.0*pi,ny+1)
    
    w = @. 2.0*nq*cos(nq*x)*cos(nq*y)'
    
    return w
end
#%%
function vm_ic(nx,ny)
    
    
    # compute initial condition for vortex-merger problem
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    
    # Output
    # ------
    # w : initial condiition for vorticity for vortex-merger problem
    
    
    w = Array{Float64}(undef,nx+1,ny+1)

    sigma = pi
    xc1 = pi-pi/4.0
    yc1 = pi
    xc2 = pi+pi/4.0
    yc2 = pi
    
    x = LinRange(0.0,2.0*pi,nx+1)
    y = LinRange(0.0,2.0*pi,ny+1)
    
    # x, y = np.meshgrid(x, y, indexing='ij')
    
    w = @. exp(-sigma*((x-xc1)^2 + (y'-yc1)^2)) + exp(-sigma*((x-xc2)^2 + (y'-yc2)^2))

    return w
end
#%%
function pbc(nx,ny,u)
    
    
    # assign periodic boundary condition in physical space
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # u : solution field
    
    # Output
    # ------
    # u : solution field with periodic boundary condition applied
        
    
    u[:,end] = u[:,1]
    u[end,:] = u[1,:]
    u[end,end] = u[1,1]
end
#%%
# set initial condition for decay of turbulence problem
function decay_ic(nx,ny,dx,dy,iP)
    
    
    # assign initial condition for vorticity for DHIT problem
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # dx,dy : grid spacing in x and y direction
    # iP : IFFT matrix
    
    # Output
    # ------
    # w : initial condition for vorticity for DHIT problem
    
    
    w = Array{Float64}(undef,nx+1,ny+1)
    
    epsilon = 1.0e-6
    
    kx = Array{Float64}(undef,nx)
    ky = Array{Float64}(undef,ny)
    
    kx[1:Int(nx/2)] = 2*pi/(Float64(nx)*dx)*Float64.(range(0,Int(nx/2)-1,step=1))
    kx[Int(nx/2)+1:nx] = 2*pi/(Float64(nx)*dx)*Float64.(range(-Int(nx/2),-1,step=1))

    ky[1:Int(ny/2)] = 2*pi/(Float64(ny)*dy)*Float64.(range(0,Int(ny/2)-1,step=1))
    ky[Int(ny/2)+1:ny] = 2*pi/(Float64(ny)*dy)*Float64.(range(-Int(ny/2),-1,step=1))
    
    kx[1] = epsilon
    ky[1] = epsilon
    
    ksi = 2.0*pi*rand(Int(nx/2+1), Int(ny/2+1))
    eta = 2.0*pi*rand(Int(nx/2+1), Int(ny/2+1))
    # ksi = [2.62022653e+00 4.52593227e+00 7.18638172e-04;1.89961158e+00 9.22094457e-01 5.80180502e-01;1.17030742e+00 2.17122208e+00 2.49296356e+00]
    # eta = [3.38548539 2.63387681 4.3053611;1.28461137 5.51737457 0.17208132;4.21267161 2.6220034  3.51035172]
    phase = zeros(Complex{Float64},nx,ny)
    wf = zeros(Complex{Float64},nx,ny)
    
    phase[2:Int(nx/2),2:Int(ny/2)]          = complex.(cos.(ksi[2:Int(nx/2),2:Int(ny/2)] +
                                            eta[2:Int(nx/2),2:Int(ny/2)]), 
                                            sin.(ksi[2:Int(nx/2),2:Int(ny/2)] +
                                            eta[2:Int(nx/2),2:Int(ny/2)]))

    phase[end:-1:Int(nx/2)+2,2:Int(ny/2)]   = complex.(cos.(-ksi[2:Int(nx/2),2:Int(ny/2)] +
                                            eta[2:Int(nx/2),2:Int(ny/2)]), 
                                            sin.(-ksi[2:Int(nx/2),2:Int(ny/2)] +
                                            eta[2:Int(nx/2),2:Int(ny/2)]))

    phase[2:Int(nx/2),end:-1:Int(ny/2)+2]   = complex.(cos.(ksi[2:Int(nx/2),2:Int(ny/2)] -
                                            eta[2:Int(nx/2),2:Int(ny/2)]), 
                                            sin.(ksi[2:Int(nx/2),2:Int(ny/2)] -
                                            eta[2:Int(nx/2),2:Int(ny/2)]))

    phase[end:-1:Int(nx/2)+2,end:-1:Int(ny/2)+2]    = complex.(cos.(-ksi[2:Int(nx/2),2:Int(ny/2)] -
                                                    eta[2:Int(nx/2),2:Int(ny/2)]), 
                                                    sin.(-ksi[2:Int(nx/2),2:Int(ny/2)] -
                                                    eta[2:Int(nx/2),2:Int(ny/2)]))

    k0 = 10.0
    c = 4.0/(3.0*sqrt(pi)*(k0^5))           
    
    kk = @. sqrt((kx^2)' + ky^2)
    es = @. c*(kk^4)*exp(-(kk/k0)^2)
    wf = @. sqrt((kk*es/pi)) * phase*(nx*ny)
            
    ut = real(iP*wf) 
    
    #periodicity
    w[1:end-1,1:end-1] = ut
    w[:,end] = w[:,1]
    w[end,:] = w[1,:]
    w[end,end] = w[1,1] 
    
    return w
end
#%%
function wave2phy(nx,ny,uf,iP)
    
    
    # Converts the field form frequency domain to the physical space.
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # uf : solution field in frequency domain (excluding periodic boundaries)
    # P : FFT matrix
    
    
    # Output
    # ------
    # u : solution in physical space (along with periodic boundaries)
    
    
    u = Array{Float64}(undef,nx+1,ny+1)

    u[1:nx,1:ny] = real(iP*uf)
    # periodic BC
    u[:,end] = u[:,1]
    u[end,:] = u[1,:]
    
    return u
end
#%%
# compute the energy spectrum numerically
function energy_spectrum(nx,ny,w,P)
    
    
    # Computation of energy spectrum and maximum wavenumber from vorticity field
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # w : vorticity field in physical spce (including periodic boundaries)
    # P : FFT matrix
    
    # Output
    # ------
    # en : energy spectrum computed from vorticity field
    # n : maximum wavenumber
    
    
    epsilon = 1.0e-6

    kx = Array{Float64}(undef,nx)
    ky = Array{Float64}(undef,ny)
    
    kx[1:Int(nx/2)] = 2*pi/(Float64(nx)*dx)*Float64.(range(0,Int(nx/2)-1,step=1))
    kx[Int(nx/2)+1:nx] = 2*pi/(Float64(nx)*dx)*Float64.(range(-Int(nx/2),-1,step=1))

    ky[1:ny] = kx[1:ny]
    
    kx[1] = epsilon
    ky[1] = epsilon

    # kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    wf = P*w[1:end-1,1:end-1]
    
    es = Array{Float64}(undef,nx,ny)
    
    kk = @. sqrt((kx^2)' + ky^2)
    es = @. pi*((abs(wf)/(nx*ny))^2)/kk
    # es = c*(kk.^4).*exp.(-(kk/k0).^2)
    n = Int(round(sqrt(nx^2 + ny^2)/2.0))-1
    
    en = zeros(n+1)
    enind = falses(nx,ny)
    for k in 1:n
        en[k+1] = 0.0
        ic = 0
        ind = @. (kk[2:end,2:end]>(k-0.5)) & (kk[2:end,2:end]<(k+0.5))
        ic = length(kk[2:end,2:end][ind])
        enind[2:end,2:end] = ind
        en[k+1] = sum(es[enind])/ic
    end
    return en, n
end
#%%
# fast poisson solver using second-order central difference scheme
function fps(nx,ny,dx,dy,k2,f,iP)
    
    
    # FFT based fast poisson solver 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # dx,dy : grid spacing in x and y direction
    # k2 : absolute wavenumber over 2D domain
    # f : right hand side of poisson equation in frequency domain (excluding periodic boundaries)
    # iP : IFFT matrix
    
    # Output
    # ------
    # u : solution to the Poisson eqution in physical space (including periodic boundaries)
    
    
    u = zeros(nx+1,ny+1)
       
    # the denominator is based on the scheme used for discrtetizing the Poisson equation
    soln = f./(-k2)
    
    # compute the inverse fourier transform
    u[1:nx,1:ny] = real(iP*soln)
    pbc(nx,ny,u)
    
    return u
end

#%%
function coarsen(nx,ny,nxc,nyc,uf) 
    
    
    # coarsen the data along with the size of the data 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # nxc,nyc : number of grid points in x and y direction on coarse grid
    # uf : solution field on fine grid in frequency domain (excluding periodic boundaries)
    
    # Output
    # ------
    # ufc : caorsened solution in frequency domain (excluding periodic boundaries)
    
    
    ufc = zeros(Complex{Float64},nxc,nyc)
    
    ufc[1:Int(nxc/2)-1,1:Int(nyc/2)-1] = uf[1:Int(nxc/2)-1,1:Int(nyc/2)-1]
    ufc[Int(nxc/2):end,1:Int(nyc/2)-1] = uf[Int(nx-nxc/2):end,1:Int(nyc/2)-1]    
    ufc[1:Int(nxc/2)-1,Int(nyc/2):end] = uf[1:Int(nxc/2)-1,Int(ny-nyc/2):end]
    ufc[Int(nxc/2):end,Int(nyc/2):end] = uf[Int(nx-nxc/2):end,Int(ny-nyc/2):end] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    return ufc
end
       
#%%
function nonlineardealiased(nx,ny,kx,ky,k2,wf,iP,rP)   
    
    
    # compute the Jacobian with 3/2 dealiasing 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : absolute wave number over 2D domain
    # wf : vorticity field in frequency domain (excluding periodic boundaries)
    # iP : IFFT matrix
    # rP2 : FFT matrix with real coeffs
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    
    nxe = Int(nx*2)
    nye = Int(ny*2)

    j1f_padded = zeros(Complex{Float64},nxe,nye)
    j2f_padded = zeros(Complex{Float64},nxe,nye)
    j3f_padded = zeros(Complex{Float64},nxe,nye)
    j4f_padded = zeros(Complex{Float64},nxe,nye)
    
    j1f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j1f[1:Int(nx/2),1:Int(ny/2)]
    j1f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j1f[Int(nx/2)+1:end,1:Int(ny/2)]    
    j1f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j1f[1:Int(nx/2),Int(ny/2)+1:end]    
    j1f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j1f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    j2f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j2f[1:Int(nx/2),1:Int(ny/2)]
    j2f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j2f[Int(nx/2)+1:end,1:Int(ny/2)]    
    j2f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j2f[1:Int(nx/2),Int(ny/2)+1:end]    
    j2f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j2f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    j3f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j3f[1:Int(nx/2),1:Int(ny/2)]
    j3f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j3f[Int(nx/2)+1:end,1:Int(ny/2)]    
    j3f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j3f[1:Int(nx/2),Int(ny/2)+1:end]    
    j3f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j3f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    j4f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j4f[1:Int(nx/2),1:Int(ny/2)]
    j4f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j4f[Int(nx/2)+1:end,1:Int(ny/2)]    
    j4f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j4f[1:Int(nx/2),Int(ny/2)+1:end]    
    j4f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j4f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
    j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
    j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
    j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
    
    j1 = real(iP*j1f_padded)
    j2 = real(iP*j2f_padded)
    j3 = real(iP*j3f_padded)
    j4 = real(iP*j4f_padded)
    
    jacp = @. j1*j2 - j3*j4

    jacpf = rP*jacp

    
    jf = zeros(Complex{Float64},nx,ny)
    
    jf[1:Int(nx/2),1:Int(ny/2)]             = jacpf[1:Int(nx/2),1:Int(ny/2)]
    jf[Int(nx/2)+1:end,1:Int(ny/2)]         = conj.(jacpf[Int(nx/2)+1:-1:2,[1;end:-1:end-Int(nx/2)+2]])    
    jf[1:Int(nx/2),Int(ny/2)+1:end]         = jacpf[1:Int(nx/2),Int(nye-ny/2)+1:end]    
    jf[Int(nx/2)+1:end,Int(ny/2)+1:end]     = conj.(jacpf[Int(nx/2)+1:-1:2,Int(nx/2)+1:-1:2])
    
    jf = jf*(nx*ny)/(nxe*nye)
    
    return jf
end
#%%
function nonlinear(nx,ny,kx,ky,k2,wf,iP,P) 
    
    
    # compute the Jacobian without dealiasing 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : absolute wave number over 2D domain
    # wf : vorticity field in frequency domain (excluding periodic boundaries)
    # iP : IFFT matrix
    # P : FFT matrix
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    
    j1 = real(iP*j1f)
    j2 = real(iP*j2f)
    j3 = real(iP*j3f)
    j4 = real(iP*j4f)
    
    jac = @. j1*j2 - j3*j4
    
    jf = P*jac
    
    return jf
end

function w_plot(nx,ny,dt,w0,w,folder,n)
    c1 = heatmap(LinRange(0,2pi,nx+1),LinRange(0,2pi,ny+1),w0,
        title = "t = 0.0",
        clim=(minimum(w0),maximum(w0)),
        axis = nothing)
    c2 = heatmap(LinRange(0,2pi,nx+1),LinRange(0,2pi,ny+1),w,
        title = "t = $(n*dt)",
        clim=(minimum(w0),maximum(w0)),
        axis = nothing)
    filename = "spectral/"*folder*"/field_spectral_"*string(n)*".png"
    plot(c1,c2,size = (1400,600))
    savefig(filename)
end

#%% coarsening
function write_data(jc,jcoarse,sgs,w,s,n,folder)
    
    
    # write the data to .csv files for post-processing
    # Inputs
    # ------
    # n : Iteration number 
    # folder : Destination folder
    # Outputs/ write
    # ------
    # jc : coarsening of Jacobian computed at fine grid
    # jcoarse : Jacobian computed for coarsed solution field
    # sgs : subgrid scale term
    # w : vorticity in physical space for fine grid (including periodic boundaries)
    # s : streamfunction in physical space for fine grid (including periodic boundaries)
    

    if !isdir("spectral/"*folder)
        mkdir("spectral/"*folder)
        mkdir("spectral/"*folder*"/01_coarsened_jacobian_field")
        mkdir("spectral/"*folder*"/02_jacobian_coarsened_field")
        mkdir("spectral/"*folder*"/03_subgrid_scale_term")
        mkdir("spectral/"*folder*"/04_vorticity")
        mkdir("spectral/"*folder*"/05_streamfunction")
    end

    filename = "spectral/"*folder*"/01_coarsened_jacobian_field/J_fourier_"*string(n)*".csv"  
    writedlm(filename,jc,',')
    filename = "spectral/"*folder*"/02_jacobian_coarsened_field/J_coarsen_"*string(n)*".csv"
    writedlm(filename,jcoarse,',')
    filename = "spectral/"*folder*"/03_subgrid_scale_term/sgs_"*string(n)*".csv"
    writedlm(filename,sgs,',')
    filename = "spectral/"*folder*"/04_vorticity/w_"*string(n)*".csv"
    writedlm(filename,w,',')
    filename = "spectral/"*folder*"/05_streamfunction/s_"*string(n)*".csv"
    writedlm(filename,s,',')
end
#%% 

function main()
    # read input file
    l1 = []
    filename = "input.txt"
    open(filename,"r")
    l1 = readdlm(filename,comment_char='!')[:,1]

    nd = Int64(l1[1])       #NXF=NYF, resolution
    nt = Int64(l1[2])       #NT, number of time step
    re = Float64(l1[3])     #Re, Reynolds number 
    dt = Float64(l1[4])     #dt; time step
    ns = Int64(l1[5])       #nf;number of files to store
    isolver = Int64(l1[6])  #isolver:[1]ikeda, [2]arakawa
    isc = Int64(l1[7])      #isc; [0]don't write-screen, [1]write-screen
    ich = Int64(l1[8])      #ich; Check for the file
    ipr = Int64(l1[9])      #ipr; [1]TGV, [2]VM, [3]Decay 
    ndc = Int64(l1[10])     #NXC=NYC, coarse resolution
    ichkp = Int64(l1[11])   #ichkp; [0]t=0, [1]checkpoint
    istart = Int64(l1[12])  #istart; last saved file (starting point)

    freq = Int(nt/ns)

    if (ich != 19)
        print("Check input.txt file")
    end
    # assign parameters
    nx = nd
    ny = nd

    nxc = ndc
    nyc = ndc

    lx = 2.0*pi
    ly = 2.0*pi

    dx = lx/Float64(nx)
    dy = ly/Float64(ny)

    dxc = lx/Float64(nxc)
    dyc = ly/Float64(nyc)

    # compute frequencies, vorticity field in frequency domain
    kx = fftfreq(nx,nx)
    ky = fftfreq(ny,ny)

    kx = reshape(kx,(nx,1))
    ky = reshape(ky,(1,ny))

    k2 = @. kx^2 + ky^2
    k2[1,1] = 1.0e-12

    P    = plan_fft(rand(nx,ny))
    Pc   = plan_fft(rand(nxc,nyc))
    P2   = plan_fft(rand(2*nx,2*ny))
    P2c  = plan_fft(rand(2*nxc,2*nyc))
    iP   = plan_ifft(rand(nx,ny))
    iPc  = plan_ifft(rand(nxc,nyc))
    iP2  = plan_ifft(rand(2*nx,2*ny))
    iP2c = plan_ifft(rand(2*nxc,2*nyc))
    rP   = plan_rfft(rand(nx,ny))
    rPc  = plan_rfft(rand(nxc,nyc))
    rP2  = plan_rfft(rand(2*nx,2*ny))
    rP2c = plan_rfft(rand(2*nxc,2*nyc))

    wnf = zeros(Complex{Float64},nx,ny)
    w1f = zeros(Complex{Float64},nx,ny)
    w2f = zeros(Complex{Float64},nx,ny)

    jnf = zeros(Complex{Float64},nx,ny)
    j1f = zeros(Complex{Float64},nx,ny)
    j2f = zeros(Complex{Float64},nx,ny)
    
    s = zeros(nx+1,ny+1)
    j = zeros(nx+1,ny+1)    
    w = zeros(nx+1,ny+1)    #lol

    jfc         = zeros(Complex{Float64},nxc,nyc)
    jcoarsef    = zeros(Complex{Float64},nxc,nyc)
    wfc         = zeros(Complex{Float64},nxc,nyc)

    jc      = zeros(nxc+1,nyc+1)
    jcoarse = zeros(nxc+1,nyc+1)

    #%%
    # set the initial condition based on the problem selected
    if (ipr == 1)
        w0 = tgv_2D_ic(nx,ny) # taylor-green vortex problem
    elseif (ipr == 2)
        w0 = vm_ic(nx,ny) # vortex-merger problem
    elseif (ipr == 3)
        w0 = decay_ic(nx,ny,dx,dy,iP) # decaying homegeneous isotropic turbulence problem
    end
    #%%  
    ifile = 0
    tchkp = ichkp*freq*istart*dt
    folder = "data_"*string(nx)*"_v2"
    if ichkp == 0
        wnf[:,:] = P*(complex.(w0[1:end-1,1:end-1],0.0)) # fourier space forward
        s[:,:] = fps(nx,ny,dx,dy,k2,-wnf,iP)
        w[:,:] = wave2phy(nx,ny,wnf,iP)
               
        kxc = fftfreq(nxc,nxc)
        kyc = fftfreq(nyc,nyc)
        kxc = reshape(kxc,(nxc,1))
        kyc = reshape(kyc,(1,nyc))
                
        k2c = @. kxc^2 + kyc^2
        k2c[1,1] = 1.0e-12
                 
        jnf[:,:] = nonlineardealiased(nx,ny,kx,ky,k2,wnf,iP2,rP2)
        j[:,:] = wave2phy(nx,ny,jnf,iP) # jacobian for fine solution field
            
         # coarsened(jacobian field)
        jfc[:,:] = coarsen(nx,ny,nxc,nyc,jnf) # coarsened(jacobian field) in frequency domain
        jc[:,:] = wave2phy(nxc,nyc,jfc,iPc) # coarsened(jacobian field) physical space
                   
        wfc[:,:] = coarsen(nx,ny,nxc,nyc,wnf)       
        jcoarsef[:,:] = nonlineardealiased(nxc,nyc,kxc,kyc,k2c,wfc,iP2c,rP2c) # jacobian(coarsened solution field) in frequency domain
        jcoarse[:,:] = wave2phy(nxc,nyc,jcoarsef,iPc) # jacobian(coarsened solution field) physical space
                
        sgs = jc - jcoarse
        write_data(jc,jcoarse,sgs,w,s,0,folder)
    elseif ichkp == 1
        print(istart)
        file_input = "spectral/"*folder*"/04_vorticity/w_"*string(istart)*".csv"
        w = readdlm(file_input, ',', Float64)
    end
    #%%

    wnf[:,:] = P*(complex.(w[1:end-1,1:end-1],0.0)) # fourier space forward

    #%%
    # initialize variables for time integration
    a1, a2, a3 = 8.0/15.0, 2.0/15.0, 1.0/3.0
    g1, g2, g3 = 8.0/15.0, 5.0/12.0, 3.0/4.0
    r2, r3 = -17.0/60.0, -5.0/12.0

    z = 0.5*dt*k2/re
    d1 = a1*z
    d2 = a2*z
    d3 = a3*z

    #%%
    clock_time_init = time()
    # time integration using hybrid third-order Runge-Kutta implicit Crank-Nicolson scheme
    # refer to Orlandi: Fluid flow phenomenon


    for n in Int(ichkp*istart*freq)+1:nt
        looptime = time()
        t = n*dt
        # 1st step
        jnf[:,:] = nonlineardealiased(nx,ny,kx,ky,k2,wnf,iP2,rP2)    
        w1f[:,:] = @. ((1.0 - d1)/(1.0 + d1))*wnf + (g1*dt*jnf)/(1.0 + d1)
        w1f[1,1] = 0.0

        # 2nd step
        j1f[:,:] = nonlineardealiased(nx,ny,kx,ky,k2,w1f,iP2,rP2)
        w2f[:,:] = @. ((1.0 - d2)/(1.0 + d2))*w1f + (r2*dt*jnf + g2*dt*j1f)/(1.0 + d2)
        w2f[1,1] = 0.0

        # 3rd step
        j2f[:,:] = nonlineardealiased(nx,ny,kx,ky,k2,w2f,iP2,rP2)
        wnf[:,:] = @. ((1.0 - d3)/(1.0 + d3))*w2f + (r3*dt*j1f + g3*dt*j2f)/(1.0 + d3)
        wnf[1,1] = 0.0
        a = time() - looptime
        @printf("Avg. %.5fs per RK3 step\n",a/3)
        # println("Avg. "*string(round(a/3; digits=5))*"s per RK3 step")
        if any(isnan.(wnf))
            println("WARNING: NaN encountered")
            println("Code will exit")
            file_input = "spectral/"*folder*"/04_vorticity/w_"*string(Int(round(n/freq-1)))*".csv"
            wback = readdlm(file_input, ',', Float64) 
            wnf[:,:] = P*(complex.(wback[1:end-1,1:end-1],0.0))
            break
            # println("SOLVER WILL BACKTRACK TO PREVIOUS SOLUTION")
            # file_input = "spectral/"*folder*"/04_vorticity/w_"*string(Int(round(n/freq-5)))*".csv"
            # wback = readdlm(file_input, ',', Float64) 
            # wnf = P*(complex.(wback[1:end-1,1:end-1],0.0))
        end
        if (mod(n,freq) == 0)
            s[:,:] = fps(nx,ny,dx,dy,k2,-wnf,iP)
            w[:,:] = wave2phy(nx,ny,wnf,iP)
               
            kxc = fftfreq(nxc,nxc)
            kyc = fftfreq(nyc,nyc)
            kxc = reshape(kxc,(nxc,1))
            kyc = reshape(kyc,(1,nyc))
                
            k2c = @. kxc^2 + kyc^2
            k2c[1,1] = 1.0e-12
                 
            jnf[:,:] = nonlineardealiased(nx,ny,kx,ky,k2,wnf,iP2,rP2)
            j[:,:] = wave2phy(nx,ny,jnf,iP) # jacobian for fine solution field
            
            # coarsened(jacobian field)
            jfc[:,:] = coarsen(nx,ny,nxc,nyc,jnf) # coarsened(jacobian field) in frequency domain
            jc[:,:] = wave2phy(nxc,nyc,jfc,iPc) # coarsened(jacobian field) physical space
                   
            wfc[:,:] = coarsen(nx,ny,nxc,nyc,wnf)       
            jcoarsef[:,:] = nonlineardealiased(nxc,nyc,kxc,kyc,k2c,wfc,iP2c,rP2c) # jacobian(coarsened solution field) in frequency domain
            jcoarse[:,:] = wave2phy(nxc,nyc,jcoarsef,iPc) # jacobian(coarsened solution field) physical space
                
            sgs = jc - jcoarse
            write_data(jc,jcoarse,sgs,w,s,Int(round(n/freq)),folder)
            @printf("n: %3i, t = %6.4f %4ix%4i\n",n,t+tchkp,nx,ny)
            # println("n: $n, t = $(round(t+tchkp; digits=4)) $(size(wnf)[1])x$(size(wnf)[2])")
        end
        if (mod(n,50*freq) == 0)
            w_plot(nx,ny,dt,w0,w,folder,n)
        end
    end
    w = wave2phy(nx,ny,wnf,P) # final vorticity field in physical space            

    total_clock_time = time() - clock_time_init
    print("Total clock time=", total_clock_time)  

    #%%
    # compute the exact, initial and final energy spectrum for DHIT problem
    if (ipr == 3)
        en, n = energy_spectrum(nx,ny,w,P)
        en0, n = energy_spectrum(nx,ny,w0,P)
        k = LinRange(1,n,n)
        
        k0 = 10.0
        c = @. 4.0/(3.0*sqrt(pi)*(k0^5))           
        ese = @. c*(k^4)*exp(-(k/k0)^2)
        
        writedlm("spectral/energy_spectral_"*string(nd)*"_"*string(Int(re))*".csv", en, ',')
    end
    #%%

    #%%
    # energy spectrum plot for DHIT problem
    if (ipr == 3)
        # fig, ax = plt.subplots()
        # fig.set_size_inches(7,5)
        
        line = 100*k^(-3.0)
        
        # ax.loglog(k,ese[:],'k', lw = 2, label='Exact')
        # ax.loglog(k,en0[1:],'r', ls = '--', lw = 2, label='$t = 0.0$')
        # ax.loglog(k,en[1:], 'b', lw = 2, label = '$t = '+string(dt*nt)+'$')
        # #ax.loglog(k,en_a[1:], 'y', lw = 2, label = '$t = '+string(dt*nt)+'$')
        # ax.loglog(k,line, 'g--', lw = 2, label = 'k^-3')
        
        # plt.xlabel('$K$')
        # plt.ylabel('$E(K)$')
        # plt.legend(loc=0)
        # plt.ylim(1e-16,1e-0)
        

        p1 = plot(k,ese,
            lw=2,
            linecolor = :green,
            xscale = :log,
            yscale = :log,
            xlabel = "k",
            ylabel = "E(k)",
            ylims = (1e-16,1e-0),
            label="Exact",
            title = "TKE spectrum",
            legend = :bottomleft,
            size=(900,900))
            
        p1 = plot!(k,en0[2:end],
            lw=2,
            ls = :dash,
            linecolor = :red,
            xscale = :log,
            yscale = :log,
            label="t = 0.0")

        p1 = plot!(k,en[2:end],
            lw=2,
            ls = :dash,
            linecolor = :blue,
            xscale = :log,
            yscale = :log,
            label="t = 0.0"*string(dt*nt))

        p1 = plot!(k,line,
            lw=2,
            ls = :dash,
            linecolor = :black,
            xscale = :log,
            yscale = :log,
            label="k^-3")
        plot(p1)
        savefig("spectral/es_spectral.png")    

    end
    # #%%
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.gca(projection='3d', proj_type = 'ortho')

    # X, Y = np.mgrid[0:2.0*pi+dx:dx, 0:2.0*pi+dy:dy]

    # surf = ax.plot_surface(X, Y, w, cmap='coolwarm',vmin=-30, vmax=30,
    #                        linewidth=0, antialiased=False,rstride=1,
    #                         cstride=1)

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.view_init(elev=60, azim=30)
    # plt.show()

    # #%%
    # savefig("vorticity_3D1.png", dpi=30)
end

main()












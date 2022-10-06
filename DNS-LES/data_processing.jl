using Printf
println(string(Threads.nthreads())*" THREADS")
using FFTW
FFTW.set_num_threads(Threads.nthreads())
using CUDA, CUDA.CUFFT
CUDA.device()
using Plots, LaTeXStrings
using Plots.PlotMeasures
gr()
using DelimitedFiles
## TODO 
# Gaussian filter the fine DNS solution
# Coarsen filtered DNS solution to LES resolution
# Calculate Π terms

CUDA.allowscalar(false)
## THE Π TERMS ARE SIMPLY jcoarse - jc

#%% Exact solution to Taylor-Green Vortex in 2D
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

#%% 2D Taylor-Green Vortex initial condition
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

#%% 2D Vortex Merger initial condition
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

#%% Initial condition for 2D DHIT
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
    
    
    w = zeros(nx+1,ny+1)
    
    epsilon = 1.0e-6
    
    kx = zeros(nx)
    ky = zeros(ny)
    
    kx[1:Int(nx/2)] = 2*pi/(Float64(nx)*dx)*Float64.(range(0,Int(nx/2)-1,step=1))
    kx[Int(nx/2)+1:nx] = 2*pi/(Float64(nx)*dx)*Float64.(range(-Int(nx/2),-1,step=1))

    ky[1:Int(ny/2)] = 2*pi/(Float64(ny)*dy)*Float64.(range(0,Int(ny/2)-1,step=1))
    ky[Int(ny/2)+1:ny] = 2*pi/(Float64(ny)*dy)*Float64.(range(-Int(ny/2),-1,step=1))
    
    kx[1] = epsilon
    ky[1] = epsilon
    
    ξ = 2.0*pi*rand(Int(nx/2+1), Int(ny/2+1))
    η = 2.0*pi*rand(Int(nx/2+1), Int(ny/2+1))

    ind = falses(Int(nx/2+1), Int(ny/2+1))
    ind[2:Int(nx/2),2:Int(ny/2)] .= true

    phase = zeros(Complex{Float64},nx,ny)
    
    phase[2:Int(nx/2),2:Int(ny/2)]          = complex.(cos.(ξ[ind] + η[ind]), 
                                                        sin.(ξ[ind] + η[ind]))

    phase[end:-1:Int(nx/2)+2,2:Int(ny/2)]   = complex.(cos.(-ξ[ind] + η[ind]), 
                                                        sin.(-ξ[ind] + η[ind]))

    phase[2:Int(nx/2),end:-1:Int(ny/2)+2]   = complex.(cos.(ξ[ind] - η[ind]), 
                                                        sin.(ξ[ind] - η[ind]))

    phase[end:-1:Int(nx/2)+2,end:-1:Int(ny/2)+2]    = complex.(cos.(-ξ[ind] - η[ind]), 
                                                                sin.(-ξ[ind] - η[ind]))

    k0 = 10.0
    c = 4.0/(3.0*sqrt(pi)*(k0^5))           
    
    kk = @. sqrt((kx^2)' + ky^2)
    es = @. c*(kk^4)*exp(-(kk/k0)^2)

    wf = zeros(Complex{Float64},nx,ny)
    wf = @. sqrt((kk*es/pi)) * phase*(nx*ny)
            
    ut = real(iP*wf) 
    
    #periodicity
    w[1:end-1,1:end-1] = ut
    w[:,end] = w[:,1]
    w[end,:] = w[1,:]
    w[end,end] = w[1,1] 
    
    return w
end

#%% Inverse Fourier transform from freq domain
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
    
    
    u = zeros(nx+1,ny+1)

    u[1:nx,1:ny] = real(iP*uf)
    # periodic BC
    u[:,end] = u[:,1]
    u[end,:] = u[1,:]
    u[end,end] = u[1,1]

    return u
end

#%% compute the energy spectrum numerically
function energy_spectrum(w,k2,P)
    
    
    # Computation of energy spectrum and maximum wavenumber from vorticity field
    
    # Inputs
    # ------
    # w : vorticity field in physical spce (including periodic boundaries)
    # k2 : 2D wavenumber Squared
    # P : FFT matrix
    
    # Output
    # ------
    # en : energy spectrum computed from vorticity field
    # n : maximum wavenumber
    
    
    wf = P*(w[1:end-1,1:end-1])
    nx = size(wf)[1]
    ny = size(wf)[2]
    es = zeros(nx,ny)
    
    kk = @. sqrt(k2)
    es = @. pi*((abs(wf)/(nx*ny))^2)/kk
    n = Int(round(sqrt(nx^2 + ny^2)/2.0))-1
    
    en = zeros(n+1)
    ind = falses(nx-1,ny-1)
    enind = falses(nx,ny)
    for k in 1:n
        en[k+1] = 0.0
        ic = 0
        ind[:,:] = @. (kk[2:end,2:end]>(k-0.5)) & (kk[2:end,2:end]<(k+0.5))
        ic = length(kk[2:end,2:end][ind])
        enind[2:end,2:end] = ind
        en[k+1] = sum(es[enind])/ic
    end
    return en, n
end

#%% Fast Poisson solver using FFT
function fps(nx,ny,k2,f,iP)
    
    
    # FFT based fast poisson solver 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction
    # dx,dy : grid spacing in x and y direction
    # k2 : absolute squared wavenumber over 2D domain
    # f : right hand side of poisson equation in frequency domain (excluding periodic boundaries)
    # iP : IFFT matrix
    
    # Output
    # ------
    # u : solution to the Poisson eqution in physical space (including periodic boundaries)
    
    
    u = zeros(nx+1,ny+1)
       
    # the denominator is based on the scheme used for discrtetizing the Poisson equation
    soln = @. f/k2
    
    # compute the inverse fourier transform
    u[1:nx,1:ny] = real(iP*soln)
    u[:,end] = u[:,1]
    u[end,:] = u[1,:]
    u[end,end] = u[1,1]
    
    return u
end

#%% Coarsening
function coarsen(nx,ny,nxc,nyc,uf) 
    
    
    # coarsen the data along with the size of the data 
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # nxc,nyc : number of grid points in x and y direction on coarse grid
    # uf : solution field on fine grid in frequency domain (excluding periodic boundaries)
    
    # Output
    # ------
    # ufc : coarsened solution in frequency domain (excluding periodic boundaries)
    
    
    ufc = zeros(Complex{Float64},nxc,nyc)
    
    ufc[1:Int(nxc/2)-1,1:Int(nyc/2)-1] = uf[1:Int(nxc/2)-1,1:Int(nyc/2)-1]
    ufc[Int(nxc/2):end,1:Int(nyc/2)-1] = uf[Int(nx-nxc/2):end,1:Int(nyc/2)-1]    
    ufc[1:Int(nxc/2)-1,Int(nyc/2):end] = uf[1:Int(nxc/2)-1,Int(ny-nyc/2):end]
    ufc[Int(nxc/2):end,Int(nyc/2):end] = uf[Int(nx-nxc/2):end,Int(ny-nyc/2):end] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    return ufc
end 

#%% Gaussian filter
function filter_gauss(Δ,k2,uf) 

    # coarsen the data along with the size of the data 
    
    # Inputs
    # ------
    # Δ : filter width
    # k2 : absolute squared wavenumber over 2D domain
    # uf : solution field on fine grid in frequency domain (excluding periodic boundaries)
    
    # Output
    # ------
    # uf_filtered : filtered solution in frequency domain (excluding periodic boundaries)
    
    G = @. exp(-(1/24)*k2*Δ^2) # taken from Y. Guan, A. Chattopadhyay, A. Subel et al
    
    uf_filtered = @. G*uf
    
    return uf_filtered
end 

#%% Compute the Jacobian with dealiasing 
function nonlineardealiased(nx,ny,kx,ky,k2,wf,iP,P,iP2,rP2,opt)   
    
    
    # Compute the Jacobian with dealiasing (Default 3/2)
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : absolute wave number over 2D domain
    # wf : vorticity field in frequency domain (excluding periodic boundaries)
    # P : FFT matrix
    # iP2 : IFFT matrix for 2x grid
    # rP2 : FFT matrix with real coeffs for 2x grid
    # opt : Method of dealiasing (1 for 3/2 padding, 2 for FT, 3 for FS)
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    if opt == 3
        # FS dealiasing
        α = 36; m = 36

        dealias = @. (
            (exp(-α*(2*abs(kx)/nx)^m))
            *
            (exp(-α*(2*abs(ky)/ny)^m))
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    elseif opt == 2
        # FT dealiasing
        k_max_dealias = 2.0/3.0 * (nx/2 + 1)
        dealias = (
            (abs.(kx) .< k_max_dealias)
            .*
            (abs.(ky) .< k_max_dealias)
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    else
        nxe = Int(3*nx/2)
        nye = Int(3*ny/2)

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
        
        j1 = real(iP2*j1f_padded)
        j2 = real(iP2*j2f_padded)
        j3 = real(iP2*j3f_padded)
        j4 = real(iP2*j4f_padded)
        
        jacp = @. j1*j2 - j3*j4

        jacpf = rP2*jacp
        
        jf = zeros(Complex{Float64},nx,ny)
        
        jf[1:Int(nx/2),1:Int(ny/2)]             = jacpf[1:Int(nx/2),1:Int(ny/2)]
        jf[Int(nx/2)+1:end,1:Int(ny/2)]         = conj.(jacpf[Int(nx/2)+1:-1:2,[1;end:-1:end-Int(nx/2)+2]])    
        jf[1:Int(nx/2),Int(ny/2)+1:end]         = jacpf[1:Int(nx/2),Int(nye-ny/2)+1:end]    
        jf[Int(nx/2)+1:end,Int(ny/2)+1:end]     = conj.(jacpf[Int(nx/2)+1:-1:2,Int(nx/2)+1:-1:2])
        
        jf = jf*(nx*ny)/(nxe*nye)
        
        return jf
    end
end

function nonlineardealiased_GPU(nx,ny,kx,ky,k2,wf,iP,P,iP2,rP2,opt)   
    
    
    # Compute the Jacobian with dealiasing (Default 3/2)
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : absolute wave number over 2D domain
    # wf : vorticity field in frequency domain (excluding periodic boundaries)
    # P : FFT matrix
    # iP2 : IFFT matrix for 2x grid
    # rP2 : FFT matrix with real coeffs for 2x grid
    # opt : Method of dealiasing (1 for 3/2 padding, 2 for FT, 3 for FS)
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    if opt == 3
        # FS dealiasing
        α = 36; m = 36

        dealias = @. (
            (exp(-α*(2*abs(kx)/nx)^m))
            *
            (exp(-α*(2*abs(ky)/ny)^m))
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    elseif opt == 2
        # FT dealiasing
        k_max_dealias = 2.0/3.0 * (nx/2 + 1)
        dealias = (
            (abs.(kx) .< k_max_dealias)
            .*
            (abs.(ky) .< k_max_dealias)
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    else
        nxe = Int(3*nx/2)
        nye = Int(3*ny/2)

        j1f_padded = zeros(Complex{Float64},nxe,nye)
        j2f_padded = zeros(Complex{Float64},nxe,nye)
        j3f_padded = zeros(Complex{Float64},nxe,nye)
        j4f_padded = zeros(Complex{Float64},nxe,nye)
        
        zero_pad = zeros(nxe-nx,nye-ny)

        j1f_padded[1:Int(nx/2),:]           = hcat(j1f[1:Int(nx/2),1:Int(ny/2)],zero_pad,j1f[1:Int(nx/2),Int(ny/2)+1:end])
        j1f_padded[Int(nxe-nx/2)+1:end,:]   = hcat(j1f[Int(nx/2)+1:end,1:Int(ny/2)],zero_pad,j1f[Int(nx/2)+1:end,Int(ny/2)+1:end])

        j2f_padded[1:Int(nx/2),:]           = hcat(j2f[1:Int(nx/2),1:Int(ny/2)],zero_pad,j2f[1:Int(nx/2),Int(ny/2)+1:end])
        j2f_padded[Int(nxe-nx/2)+1:end,:]   = hcat(j2f[Int(nx/2)+1:end,1:Int(ny/2)],zero_pad,j2f[Int(nx/2)+1:end,Int(ny/2)+1:end])

        j3f_padded[1:Int(nx/2),:]           = hcat(j3f[1:Int(nx/2),1:Int(ny/2)],zero_pad,j3f[1:Int(nx/2),Int(ny/2)+1:end])
        j3f_padded[Int(nxe-nx/2)+1:end,:]   = hcat(j3f[Int(nx/2)+1:end,1:Int(ny/2)],zero_pad,j3f[Int(nx/2)+1:end,Int(ny/2)+1:end])

        j4f_padded[1:Int(nx/2),:]           = hcat(j4f[1:Int(nx/2),1:Int(ny/2)],zero_pad,j4f[1:Int(nx/2),Int(ny/2)+1:end])
        j4f_padded[Int(nxe-nx/2)+1:end,:]   = hcat(j4f[Int(nx/2)+1:end,1:Int(ny/2)],zero_pad,j4f[Int(nx/2)+1:end,Int(ny/2)+1:end])

        j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
        j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
        j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
        j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
        
        j1 = real(iP2*j1f_padded)
        j2 = real(iP2*j2f_padded)
        j3 = real(iP2*j3f_padded)
        j4 = real(iP2*j4f_padded)
        
        jacp = @. j1*j2 - j3*j4

        jacpf = rP2*jacp
        
        jf = zeros(Complex{Float64},nx,ny)
        
        jf[1:Int(nx/2),1:Int(ny/2)]             = jacpf[1:Int(nx/2),1:Int(ny/2)]
        jf[Int(nx/2)+1:end,1:Int(ny/2)]         = conj.(jacpf[Int(nx/2)+1:-1:2,[1;end:-1:end-Int(nx/2)+2]])    
        jf[1:Int(nx/2),Int(ny/2)+1:end]         = jacpf[1:Int(nx/2),Int(nye-ny/2)+1:end]    
        jf[Int(nx/2)+1:end,Int(ny/2)+1:end]     = conj.(jacpf[Int(nx/2)+1:-1:2,Int(nx/2)+1:-1:2])
        
        jf = jf*(nx*ny)/(nxe*nye)
        
        return jf
    end
end

#%% Compute the Jacobian with dealiasing 
function nonlineardealiased_implicit(nx,ny,kx,ky,k2,wf,P,iP2,rP2)   
    
    
    # Compute the Jacobian with dealiasing (Default 3/2)
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : absolute wave number over 2D domain
    # wf : vorticity field in frequency domain (excluding periodic boundaries)
    # P : FFT matrix
    # iP2 : IFFT matrix for 2x grid
    # rP2 : FFT matrix with real coeffs for 2x grid
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    nxe = Int(3*nx/2)
    nye = Int(3*ny/2)

    j1f_padded = zeros(Complex{Float64},nxe,nye)
    j2f_padded = zeros(Complex{Float64},nxe,nye)
    j3f_padded = zeros(Complex{Float64},nxe,nye)
    j4f_padded = zeros(Complex{Float64},nxe,nye)
    
    # j1f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j1f[1:Int(nx/2),1:Int(ny/2)]
    # j1f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j1f[Int(nx/2)+1:end,1:Int(ny/2)]    
    # j1f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j1f[1:Int(nx/2),Int(ny/2)+1:end]    
    # j1f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j1f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    # j2f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j2f[1:Int(nx/2),1:Int(ny/2)]
    # j2f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j2f[Int(nx/2)+1:end,1:Int(ny/2)]    
    # j2f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j2f[1:Int(nx/2),Int(ny/2)+1:end]    
    # j2f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j2f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    # j3f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j3f[1:Int(nx/2),1:Int(ny/2)]
    # j3f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j3f[Int(nx/2)+1:end,1:Int(ny/2)]    
    # j3f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j3f[1:Int(nx/2),Int(ny/2)+1:end]    
    # j3f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j3f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    # j4f_padded[1:Int(nx/2),1:Int(ny/2)]                     = j4f[1:Int(nx/2),1:Int(ny/2)]
    # j4f_padded[Int(nxe-nx/2)+1:end,1:Int(ny/2)]             = j4f[Int(nx/2)+1:end,1:Int(ny/2)]    
    # j4f_padded[1:Int(nx/2),Int(nye-ny/2)+1:end]             = j4f[1:Int(nx/2),Int(ny/2)+1:end]    
    # j4f_padded[Int(nxe-nx/2)+1:end,Int(nye-ny/2)+1:end]     = j4f[Int(nx/2)+1:end,Int(ny/2)+1:end] 
    
    j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
    j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
    j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
    j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
    
    j1 = real(iP2*j1f_padded)
    j2 = real(iP2*j2f_padded)
    j3 = real(iP2*j3f_padded)
    j4 = real(iP2*j4f_padded)
    
    jacp = @. j1*j2 - j3*j4

    jacpf = rP2*jacp
    
    jf = zeros(Complex{Float64},nx,ny)
    
    jf[1:Int(nx/2),1:Int(ny/2)]             = jacpf[1:Int(nx/2),1:Int(ny/2)]
    jf[Int(nx/2)+1:end,1:Int(ny/2)]         = conj.(jacpf[Int(nx/2)+1:-1:2,[1;end:-1:end-Int(nx/2)+2]])    
    jf[1:Int(nx/2),Int(ny/2)+1:end]         = jacpf[1:Int(nx/2),Int(nye-ny/2)+1:end]    
    jf[Int(nx/2)+1:end,Int(ny/2)+1:end]     = conj.(jacpf[Int(nx/2)+1:-1:2,Int(nx/2)+1:-1:2])
    
    jf = jf*(nx*ny)/(nxe*nye)
    
    return jf
end

#%% Compute the Jacobian without dealiasing
function nonlinear(nx,ny,kx,ky,k2,wf,iP,P) 
    
    
    # Compute the Jacobian without dealiasing 
    
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

#%%
function w_plot(nx,ny,dt,w0,w,folder,n)
    # Plots the vorticity field

    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid 
    # dt: Time step
    # folder : Destination folder
    # Outputs/ write
    # ------
    # w0 : Initial vorticity field
    # w : vorticity field at time t = n*dt
    c1 = heatmap(LinRange(0,2pi,nx+1),LinRange(0,2pi,ny+1),w0,
        title = "t = 0.0",
        clim=(minimum(w0),maximum(w0)),
        axis = nothing)
    c2 = heatmap(LinRange(0,2pi,nx+1),LinRange(0,2pi,ny+1),w,
        title = "t = $(round(n*dt,digits=2))",
        clim=(minimum(w0),maximum(w0)),
        axis = nothing)
    filename = "spectral/"*folder*"/field_spectral_"*string(n)*".png"
    plot(c1,c2,size = (1400,600))
    savefig(filename)
end
blank = plot(foreground_color_subplot=:white)
l = @layout [grid(2, 2) a{0.095w}]
plot(c1, c2, c3, c4, blank, layout=l, link=:all)
p_all = scatter!([0], [0], zcolor=[NaN], clims=(minimum(w0),maximum(w0)), label="", colorbar_title="Vorticity", background_color_subplot=:transparent, markerstrokecolor=:transparent, framestyle=:none, inset=bbox(0.1, 0, 1, 0.9, :center, :right), subplot=6)
# %%
function write_data(jc,jcoarse,sgs,w,s,w_LES,s_LES,n,folder)
    
    
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
    # w_LES : filtered vorticity (including periodic boundaries)
    # s : streamfunction in physical space for fine grid (including periodic boundaries)
    # s_LES : filtered streamfunction (including periodic boundaries)
    

    if !isdir("spectral/"*folder)
        mkdir("spectral/"*folder)
        mkdir("spectral/"*folder*"/01_coarsened_jacobian_field")
        mkdir("spectral/"*folder*"/02_jacobian_coarsened_field")
        mkdir("spectral/"*folder*"/03_subgrid_scale_term")
        mkdir("spectral/"*folder*"/04_DNS_vorticity")
        mkdir("spectral/"*folder*"/05_LES_vorticity")
        mkdir("spectral/"*folder*"/06_DNS_streamfunction")
        mkdir("spectral/"*folder*"/07_LES_streamfunction")
    end

    filename = "spectral/"*folder*"/01_coarsened_jacobian_field/J_fourier_"*string(n)*".csv"  
    writedlm(filename,jc,',')
    filename = "spectral/"*folder*"/02_jacobian_coarsened_field/J_coarsen_"*string(n)*".csv"
    writedlm(filename,jcoarse,',')
    filename = "spectral/"*folder*"/03_subgrid_scale_term/sgs_"*string(n)*".csv"
    writedlm(filename,sgs,',')
    filename = "spectral/"*folder*"/04_DNS_vorticity/w_"*string(n)*".csv"
    writedlm(filename,w,',')
    filename = "spectral/"*folder*"/05_LES_vorticity/w_"*string(n)*".csv"
    writedlm(filename,w_LES,',')
    filename = "spectral/"*folder*"/06_DNS_streamfunction/s_"*string(n)*".csv"
    writedlm(filename,s,',')
    filename = "spectral/"*folder*"/07_LES_streamfunction/s_"*string(n)*".csv"
    writedlm(filename,s_LES,',')
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
    opt = Int64(l1[11])     #Dealiasing algorithm:[1]3/2 padding, [2]FT, [3],FS
    ichkp = Int64(l1[12])   #ichkp; [0]t=0, [1]checkpoint
    istart = Int64(l1[13])  #istart; last saved file (starting point)

    freq = Int(nt/ns)
    @printf("DNS RESOLUTION: %ix%i\n",nd,nd)
    @printf("LES RESOLUTION: %ix%i\n",ndc,ndc)
    @printf("REYNOLDS NUMBER = %.0f\n",re)
    @printf("STORING %i FILES PER VARIABLE\n",ns)
    @printf("WILL RUN FOR %i ITERATIONS UNTIL t = %.1f\n",nt,nt*dt)
    if opt == 3
        @printf("FOURIER SMOOTHING SELECTED AS DEALIASING METHOD\n")
    elseif opt == 2
        @printf("FOURIER TRUNCATION SELECTED AS DEALIASING METHOD\n")
    else
        @printf("3/2 PADDING SELECTED AS DEALIASING METHOD\n")
    end
    
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

    kxc = fftfreq(nxc,nxc)
    kyc = fftfreq(nyc,nyc)
    kxc = reshape(kxc,(nxc,1))
    kyc = reshape(kyc,(1,nyc))
            
    k2c = @. kxc^2 + kyc^2
    k2c[1,1] = 1.0e-12

    kc = ndc/2
    Δ = 2pi/kc

    P    = plan_fft(rand(nx,ny))
    Pc   = plan_fft(rand(nxc,nyc))
    P2   = plan_fft(rand(Int(3*nx/2),Int(3*ny/2)))
    P2c  = plan_fft(rand(Int(3*nxc/2),Int(3*nyc/2)))
    iP   = plan_ifft(rand(nx,ny))
    iPc  = plan_ifft(rand(nxc,nyc))
    iP2  = plan_ifft(rand(Int(3*nx/2),Int(3*ny/2)))
    iP2c = plan_ifft(rand(Int(3*nxc/2),Int(3*nyc/2)))
    rP   = plan_rfft(rand(nx,ny))
    rPc  = plan_rfft(rand(nxc,nyc))
    rP2  = plan_rfft(rand(Int(3*nx/2),Int(3*ny/2)))
    rP2c = plan_rfft(rand(Int(3*nxc/2),Int(3*nyc/2)))

    wnf = zeros(Complex{Float64},nx,ny)
    w1f = zeros(Complex{Float64},nx,ny)
    w2f = zeros(Complex{Float64},nx,ny)

    wfc         = zeros(Complex{Float64},nxc,nyc)
    jfc         = zeros(Complex{Float64},nxc,nyc)
    jcoarsef    = zeros(Complex{Float64},nxc,nyc)

    w   = zeros(nx+1,ny+1)
    w0  = zeros(nx+1,ny+1)

    jnf = zeros(Complex{Float64},nx,ny)
    j1f = zeros(Complex{Float64},nx,ny)
    j2f = zeros(Complex{Float64},nx,ny)
    
    s = zeros(nx+1,ny+1)
    j = zeros(nx+1,ny+1)    
    w = zeros(nx+1,ny+1)    
    s_LES = zeros(nxc+1,nyc+1)
    w_LES = zeros(nxc+1,nyc+1)

    jc      = zeros(nxc+1,nyc+1)
    jcoarse = zeros(nxc+1,nyc+1)
    
    #%%
    # set the initial condition based on the problem selected
    ifile = 0
    tchkp = ichkp*freq*istart*dt
    folder = "data_"*string(nx)*"_re_"*string(Int(re))*"_v2"
    #%%
    file_input = "spectral/"*folder*"/04_DNS_vorticity/w_"*string(0)*".csv"
    w0[:,:] = readdlm(file_input, ',', Float64)
    # for i in 1:16
    #     file_input = "spectral/"*folder*"/04_DNS_vorticity/w_"*string(Int(i*2500/freq))*".csv"
    #     w[:,:] = readdlm(file_input, ',', Float64)
    #     w_plot(nx,ny,dt,w0,w,folder,i*2500)
    # end
    file_input = "spectral/"*folder*"/04_DNS_vorticity/w_"*string(ns)*".csv"
    w[:,:] = readdlm(file_input, ',', Float64)
    file_input = "spectral/"*folder*"/05_LES_vorticity/w_"*string(ns)*".csv"
    w_LES[:,:] = readdlm(file_input, ',', Float64)
    wnf[:,:] = P*(complex.(w[1:end-1,1:end-1],0.0)) # fourier space forward
    #%%
    # compute the exact, initial and final energy spectrum for DHIT problem
    if (ipr == 3)
        en, n = energy_spectrum(w,k2,P)
        en0, n = energy_spectrum(w0,k2,P)
        en_filt, nc = energy_spectrum(w_LES,k2c,Pc)
        en_LES_NM = readdlm("spectral/data_256_re_32000_v2/energy_spectral_"*string(ndc)*"_"*string(Int(re))*".csv",',', Float64)

        k = LinRange(1,n,n)
        kc = LinRange(1,nc,nc)

        k0 = 10.0
        c = @. 4.0/(3.0*sqrt(pi)*(k0^5))           
        ese = @. c*(k^4)*exp(-(k/k0)^2)
        
        writedlm("spectral/"*folder*"/energy_spectral_"*string(nd)*"_"*string(Int(re))*".csv", en, ',')
    end
    #%%

    #%%
    # energy spectrum plot for DHIT problem
    if (ipr == 3)
    
        line = @. 100*k[20:110]^(-3.0)
        
        scalefontsizes(2)
        p1 = plot(k,ese,
            lw=2,
            ls = :dash,
            linecolor = :green,
            xscale = :log,
            yscale = :log,
            xlabel = L"k",
            ylabel = L"E(k)",
            yticks = exp10.(range(-16,stop=0,length=9)),
            xticks = exp10.(range(-0,stop=3,length=4)),
            ylims = (1e-16,1e-0),
            label=L"Exact",
            title = "TKE spectrum",
            legend = :bottomleft,
            left_margin = [10mm 0mm],
            bottom_margin = [10mm 10mm],
            size=(1400,900))
            
        p1 = plot!(k,en0[2:end],
            lw=2,
            ls = :solid,
            linecolor = :red,
            xscale = :log,
            yscale = :log,
            label=L"t = 0.0")

        p1 = plot!(k,en[2:end],
            lw=2,
            ls = :solid,
            linecolor = :blue,
            xscale = :log,
            yscale = :log,
            label=latexstring("DNS"))

        p1 = plot!(kc,en_filt[2:end],
            lw=2,
            ls = :solid,
            linecolor = :cyan,
            xscale = :log,
            yscale = :log,
            label=latexstring("Filtered DNS"))

        p1 = plot!(kc,en_LES_NM[2:end],
            lw=2,
            ls = :solid,
            linecolor = :magenta,
            xscale = :log,
            yscale = :log,
            label=latexstring("LES-NM"))

        p1 = plot!(kc,en_LES_CNN[2:end],
            lw=2,
            ls = :solid,
            linecolor = :lime,
            xscale = :log,
            yscale = :log,
            label=latexstring("LES-CNN"))

        p1 = plot!(k,exp.(-(1/24)*k.^2*Δ^2),
            lw=1,
            ls = :dash,
            linecolor = :red,
            xscale = :log,
            yscale = :log,
            label=latexstring("Gaussian Filter"))

        p1 = plot!(nc*[1,1],[1e-16,1],
            lw=1,
            ls = :dash,
            linecolor = :black,
            xscale = :log,
            yscale = :log,
            label=false)

        p1 = plot!(k[20:110],line,
            lw=1,
            ls = :dash,
            linecolor = :black,
            xscale = :log,
            yscale = :log,
            label=false
            )
        annotate!([1e2],[1e-3],L"k^{-3}",font(16))
        annotate!([165],[1e-14],L"k_c",font(16))
        
        savefig("spectral/"*folder*"/es_spectral_LESNM.png")    

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

# main()








## WIP
function nonlineardealiased_GPU(nx,ny,kx,ky,k2,wf,iP,P,iP2,rP2,opt)   
    
    
    # Compute the Jacobian with dealiasing (Default 3/2) using GPU
    
    # Inputs
    # ------
    # nx,ny : number of grid points in x and y direction on fine grid
    # kx,ky : wavenumber in x and y direction
    # k2 : CUDA absolute wave number over 2D domain
    # wf : CUDA vorticity field in frequency domain (excluding periodic boundaries)
    # P : FFT matrix
    # iP2 : IFFT matrix for 2x grid
    # rP2 : FFT matrix with real coeffs for 2x grid
    # opt : Method of dealiasing (1 for 3/2 padding, 2 for FT, 3 for FS)
    
    # Output
    # ------
    # jf : jacobian in frequency domain (excluding periodic boundaries)
    #      (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    
    j1f = @. -1.0im*kx*wf/k2
    j2f = @. 1.0im*ky*wf
    j3f = @. -1.0im*ky*wf/k2
    j4f = @. 1.0im*kx*wf
    if opt == 3
        # FS dealiasing
        α = 36; m = 36

        dealias = @. (
            (exp(-α*(2*abs(kx)/nx)^m))
            *
            (exp(-α*(2*abs(ky)/ny)^m))
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    elseif opt == 2
        # FT dealiasing
        k_max_dealias = 2.0/3.0 * (nx/2 + 1)
        dealias = (
            (abs.(kx) .< k_max_dealias)
            .*
            (abs.(ky) .< k_max_dealias)
        )
        j1 = real(iP*j1f)
        j2 = real(iP*j2f)
        j3 = real(iP*j3f)
        j4 = real(iP*j4f)
        
        jac = @. j1*j2 - j3*j4
        
        jf = P*jac
        jf .*= dealias
        return jf
    else
        nxe = Int(3*nx/2)
        nye = Int(3*ny/2)

        j1f_padded = CUDA.zeros(Complex{Float64},nxe,nye)
        j2f_padded = CUDA.zeros(Complex{Float64},nxe,nye)
        j3f_padded = CUDA.zeros(Complex{Float64},nxe,nye)
        j4f_padded = CUDA.zeros(Complex{Float64},nxe,nye)
        
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
        
        j1 = real(iP2*j1f_padded)
        j2 = real(iP2*j2f_padded)
        j3 = real(iP2*j3f_padded)
        j4 = real(iP2*j4f_padded)
        
        jacp = @. j1*j2 - j3*j4

        jacpf = rP2*jacp
        
        jf = CUDA.zeros(Complex{Float64},nx,ny)
        
        jf[1:Int(nx/2),1:Int(ny/2)]             = jacpf[1:Int(nx/2),1:Int(ny/2)]
        jf[Int(nx/2)+1:end,1:Int(ny/2)]         = conj.(jacpf[Int(nx/2)+1:-1:2,[1;end:-1:end-Int(nx/2)+2]])    
        jf[1:Int(nx/2),Int(ny/2)+1:end]         = jacpf[1:Int(nx/2),Int(nye-ny/2)+1:end]    
        jf[Int(nx/2)+1:end,Int(ny/2)+1:end]     = conj.(jacpf[Int(nx/2)+1:-1:2,Int(nx/2)+1:-1:2])
        
        jf = jf*(nx*ny)/(nxe*nye)
        
        return jf
    end
end
# function var_gif(folder,var)
#     file_input = "spectral/"*folder*"/"*var*"/"*string(i)*".csv"
#     f = readdlm(file_input, ',', Float64)
# end
folder = "data_"*string(2048)*"_re_"*string(Int(32000))*"_v2"
file_input = "spectral/"*folder*"/"*"04_DNS_vorticity"*"/"*"w_0.csv"
w0 = readdlm(file_input, ',', Float64)
dt = 4/400
anim = @animate for i ∈ 0:2:400
    println(i)
    t = round(dt*i,digits=2)
    file_input = "spectral/"*folder*"/"*"04_DNS_vorticity"*"/"*"w_"*string(i)*".csv"
    w = readdlm(file_input, ',', Float64)
    p1 = heatmap(w,clim=(minimum(w0),maximum(w0)),size=(1920,1080),axis = nothing,colorbar=false,left_margin = [1mm 1mm],
    bottom_margin = [1mm 1mm])
    plot(p1)
end
gif(anim, "Vorticity_Re32000_full.gif", fps =50)
# stds = zeros(8001,3)
# for i in 0:8000
#     file_input = "spectral/Training set/"*folder*"/05_LES_vorticity/w_"*string(i)*".csv"
#     stds[i+1,1] = std(readdlm(file_input, ',', Float64))
#     file_input = "spectral/Training set/"*folder*"/07_LES_streamfunction/s_"*string(i)*".csv"
#     stds[i+1,2] = std(readdlm(file_input, ',', Float64))
#     file_input = "spectral/Training set/"*folder*"/03_subgrid_scale_term/sgs_"*string(i)*".csv"
#     stds[i+1,3] = std(readdlm(file_input, ',', Float64))
# end
for i = 1:10
    s1 = heatmap(cpu(CNN_model(gpu(input_test[:,:,:,ind[i]:ind[i]]))[:,:]),clim=(-8,8),c=:seismic,title="CNN")
    s2 = heatmap(output_test[:,:,1,ind[i]],clim=(-8,8),c=:seismic,title="True")
    #cc = corr_coeff(gpu(input_test[:,:,:,ind[i]:ind[i]]),output_test[:,:,1,ind[i]])
    plot(s1,s2,size=(1000,450),axis=nothing)
    savefig("SGS_comp_"*string(i)*".png")
end
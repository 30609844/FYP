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
nx = 2048; re = 32000
folder = "data_"*string(nx)*"_re_"*string(Int(re))*"_v2"

gif(anim, "Temperature_field_Re100.gif", fps = 50)
# function var_gif(folder,var)
#     file_input = "spectral/"*folder*"/"*var*"/"*string(i)*".csv"
#     f = readdlm(file_input, ',', Float64)
# end
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
 kx = CuArray(fftfreq(nx,nx))
 ky = CuArray(fftfreq(ny,ny))

 kx = reshape(kx,(nx,1))
 ky = reshape(ky,(1,ny))

 k2 = @. kx^2 + ky^2
 k2[1,1] = 1.0e-12

 kxc = CuArray(fftfreq(nxc,nxc))
 kyc = CuArray(fftfreq(nyc,nyc))
 kxc = reshape(kxc,(nxc,1))
 kyc = reshape(kyc,(1,nyc))
         
 k2c = @. kxc^2 + kyc^2
 k2c[1,1] = 1.0e-12

 kc = ndc/2
 Δ = pi/kc

 P    = plan_fft(CUDA.rand(nx,ny))
 Pc   = plan_fft(CUDA.rand(nxc,nyc))
 P2   = plan_fft(CUDA.rand(Int(3*nx/2),Int(3*ny/2)))
 P2c  = plan_fft(CUDA.rand(Int(3*nxc/2),Int(3*nyc/2)))
 iP   = plan_ifft(CUDA.rand(nx,ny))
 iPc  = plan_ifft(CUDA.rand(nxc,nyc))
 iP2  = plan_ifft(CUDA.rand(Int(3*nx/2),Int(3*ny/2)))
 iP2c = plan_ifft(CUDA.rand(Int(3*nxc/2),Int(3*nyc/2)))
 rP   = plan_rfft(CUDA.rand(nx,ny))
 rPc  = plan_rfft(CUDA.rand(nxc,nyc))
 rP2  = plan_rfft(CUDA.rand(Int(3*nx/2),Int(3*ny/2)))
 rP2c = plan_rfft(CUDA.rand(Int(3*nxc/2),Int(3*nyc/2)))

 w0  = CUDA.zeros(nx+1,ny+1)
 w0[:,:] = decay_ic(nx,ny,dx,dy,iP)

 wnf = CUDA.zeros(Complex{Float64},nx,ny)
 wnf[:,:] = P*(complex.(w0[1:end-1,1:end-1],0.0))
w0 = readdlm(file_input, ',', Float64)
anim = @animate for i ∈ 1:36
    println(i)
    t = round(dt*i,digits=4)
    file_input = "spectral/"*folder*"/"*"04_DNS_vorticity"*"/"*"w_"*string(i)*".csv"
    w = readdlm(file_input, ',', Float64)
    p1 = heatmap(w,clim=(minimum(w0),maximum(w0)),size=(1920,1080),axis = nothing,colorbar=false,left_margin = [1mm 1mm],
    bottom_margin = [1mm 1mm])
    plot(p1)
end
gif(anim, "Vorticity_Re32000.gif", fps = 30)
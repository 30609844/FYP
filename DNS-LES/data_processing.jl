using Printf
println(string(Threads.nthreads())*" THREADS")
using FFTW
FFTW.set_num_threads(Threads.nthreads())
using CUDA, CUDA.CUFFT
using Plots, LaTeXStrings
using Plots.PlotMeasures
gr()
using DelimitedFiles
## TODO 
# Gaussian filter the fine DNS solution
# Coarsen filtered DNS solution to LES resolution
# Calculate Π terms


## THE Π TERMS ARE SIMPLY jcoarse - jc
nx = 2048; re = 32000
folder = "data_"*string(nx)*"_re_"*string(Int(re))*"_v2"
# function var_gif(folder,var)
#     file_input = "spectral/"*folder*"/"*var*"/"*string(i)*".csv"
#     f = readdlm(file_input, ',', Float64)
# end
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
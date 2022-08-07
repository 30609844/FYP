using Plots
GR()
using DelimitedFiles
## TODO 
# Gaussian filter the fine DNS solution
# Coarsen filtered DNS solution to LES resolution
# Calculate Π terms


## THE Π TERMS ARE SIMPLY jcoarse - jc

function var_gif(folder,var)
    file_input = "spectral/"*folder*"/"*var*"/"*string(i)*".csv"
    f = readdlm(file_input, ',', Float64)
end

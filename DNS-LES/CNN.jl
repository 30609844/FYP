using Flux, CUDA, Statistics, FFTW, Random
using Flux: onehotbatch, onecold, crossentropy, throttle
FFTW.set_num_threads(Threads.nthreads())
using Base.Iterators: repeated, partition
using Printf
using Plots, LaTeXStrings
using Plots.PlotMeasures
using DelimitedFiles

gr()
println(string(Threads.nthreads())*" THREADS")
CUDA.device()
println("Start....")

function make_minibatch(X, Y, idxs)
  input_batch = Array{Float32}(undef, size(X[:,:,:,1])..., length(idxs))
  output_batch = Array{Float32}(undef, size(Y[:,:,:,1])..., length(idxs))
  for i in 1:length(idxs)
      input_batch[:, :, :, i] = Float32.(X[:,:,:,idxs[i]])
      output_batch[:, :, :, i] = Float32.(Y[:,:,:,idxs[i]])
  end
  return (input_batch, output_batch)
end

function build_model(conv_depth, kernel_size, act_func)

  @printf("CNN depth: %i\n",conv_depth)
  @printf("Kernel size: %i\n",kernel_size[1])

  model = Chain(
    Conv(kernel_size,2 => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => conv_depth,act_func,pad=SamePad()),
    Conv(kernel_size,conv_depth => 1,identity,pad=SamePad()),
    ) #|> gpu
  return model

end

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

  ## Data set has 7000 data points
  trainN=7000
  testN=350
  lead=1;
  minibatch_size = 35
  num_epochs = Int64(trainN/minibatch_size)
  pool_size = 2
  drop_prob=0.0
  Nlat=257
  Nlon=257
  n_channels=2
  NT = 7000 # Numer of snapshots per file
  numDataset = 5 # number of dataset / 2

  idxs = shuffle(1:7000).+1000
  mb_idxs = partition(1:trainN, minibatch_size)

  

  input_train = zeros(Float32,Nlon,Nlat,n_channels,trainN)
  output_train = zeros(Float32,Nlon,Nlat,1,trainN)
  input_test = zeros(Float32,Nlon,Nlat,n_channels,testN)
  output_test = zeros(Float32,Nlon,Nlat,1,testN)

  # Load training data
  folder = "data_"*string(nd)*"_re_"*string(Int(re))*"_v2"
  for i in 1:trainN
    file_input_w = "spectral/Training set/"*folder*"/05_LES_vorticity/w_"*string(i+1000)*".csv"
    file_input_s = "spectral/Training set/"*folder*"/07_LES_streamfunction/s_"*string(i+1000)*".csv"
    input_train[:,:,1,i] = readdlm(file_input_w, ',', Float32)/std(readdlm(file_input_w, ',', Float32))
    input_train[:,:,2,i] = readdlm(file_input_s, ',', Float32)/std(readdlm(file_input_s, ',', Float32))
    file_input_sgs = "spectral/Training set/"*folder*"/03_subgrid_scale_term/sgs_"*string(i+1000)*".csv"
    output_train[:,:,1,i] = readdlm(file_input_sgs, ',', Float32)/std(readdlm(file_input_sgs, ',', Float32))
  end

  # Load testing data
  folder = "data_"*string(nd)*"_re_"*string(Int(re))*"_v2"
  for i in 1:testN
    file_input_w = "spectral/Testing set/"*folder*"/05_LES_vorticity/w_"*string(i+50)*".csv"
    file_input_s = "spectral/Testing set/"*folder*"/07_LES_streamfunction/s_"*string(i+50)*".csv"
    input_test[:,:,1,i] = readdlm(file_input_w, ',', Float32)/std(readdlm(file_input_w, ',', Float32))
    input_test[:,:,2,i] = readdlm(file_input_s, ',', Float32)/std(readdlm(file_input_s, ',', Float32))
    file_input_sgs = "spectral/Testing set/"*folder*"/03_subgrid_scale_term/sgs_"*string(i+50)*".csv"
    output_test[:,:,1,i] = readdlm(file_input_s, ',', Float32)/std(readdlm(file_input_sgs, ',', Float32))
   end

  # Bundle snapshots together into minibatches
  mb_idxs = partition(1:trainN, minibatch_size)
  train_set = [make_minibatch(input_train, output_train, i) for i in mb_idxs]

  # Prepare test set as one giant minibatch:
  test_set = make_minibatch(input_test, output_test, 1:testN)

  # Load model and datasets onto GPU, if enabled
  train_set |> gpu
  test_set |> gpu

  # CNN Parameters
  params = (conv_depth = 64, kernel_size = (5,5), act_func=relu)
  CNN_model = build_model(params...)

  # Make sure our model is nicely precompiled before starting our training loop
  CNN_model(train_set[1][1])

  loss(x,y) = Flux.Losses.mse(CNN_model(x),y)

  opt = ADAM(1e-5)
  @info("Beginning training loop...")

  println("TRAINING BEGIN")
  for i in 1:4000
    Flux.train!(loss, Flux.params(CNN_model), train_set, opt)
    if mod(i,100) == 0
      @printf("Epoch: %i, MSE = %6.4e\n",i,loss)
    end
  end
end

main()
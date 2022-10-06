using Flux, CUDA, Statistics, FFTW, Random, Zygote
using Flux: onehotbatch, onecold, crossentropy, throttle, loadparams!
using BSON: @load, @save
FFTW.set_num_threads(Threads.nthreads())
using Base.Iterators: repeated, partition
using Printf
using Plots, LaTeXStrings
using Plots.PlotMeasures
using DelimitedFiles

gr()
println(string(Threads.nthreads())*" THREADS")
CUDA.device!(1)
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
  folder = "data_"*string(nd)*"_re_"*string(Int(re))*"_v2"

  ## Data set has 6000 data points
  trainN=6000
  testN=300
  lead=1;
  minibatch_size = 30
  num_epochs = 20
  pool_size = 2
  drop_prob=0.0
  Nlat=256
  Nlon=256
  n_channels=2
  NT = 6000 # Numer of snapshots per file
  numDataset = 3 # number of dataset / 2

  # Load testing data
  input_test = zeros(Float32,Nlon,Nlat,n_channels,testN)
  output_test = zeros(Float32,Nlon,Nlat,1,testN)
  for i in 1:testN
    file_input_w = "spectral/Testing set/"*folder*"/05_LES_vorticity/w_"*string(i+100)*".csv"
    file_input_s = "spectral/Testing set/"*folder*"/07_LES_streamfunction/s_"*string(i+100)*".csv"
    input_test[:,:,1,i] = Flux.normalise(readdlm(file_input_w, ',', Float32)[1:end-1,1:end-1])
    input_test[:,:,2,i] = Flux.normalise(readdlm(file_input_s, ',', Float32)[1:end-1,1:end-1])
    file_input_sgs = "spectral/Testing set/"*folder*"/03_subgrid_scale_term/sgs_"*string(i+100)*".csv"
    output_test[:,:,1,i] = Flux.normalise(readdlm(file_input_sgs, ',', Float32)[1:end-1,1:end-1])
  end
  # input_test[:,:,1,:] = Flux.normalise(input_test[:,:,1,:])
  # input_test[:,:,2,:] = Flux.normalise(input_test[:,:,2,:])
  # output_test[:,:,1,:] = Flux.normalise(output_test[:,:,1,:]) # stdSGS = 11.5


  # Prepare test set as one giant minibatch:
  test_set = make_minibatch(input_test, output_test, 1:testN)
  test_set = test_set |> gpu

  # CNN Parameters
  params = (conv_depth = 64, kernel_size = (5,5), act_func=relu)
  CNN_model = build_model(params...)
  if isfile("CNN_model.bson")
    @load "CNN_model.bson" CNN_model
  end
  CNN_model = CNN_model |> gpu

  loss(x,y) = Flux.Losses.mse(CNN_model(x),y)
  corr_coeff(x,y) = cov(vec(CNN_model(x)),vec(y))/(std(CNN_model(x))*std(y))
  opt = ADAM(1e-5)

  # Only training data will be batched
  input_train = zeros(Float32,Nlon,Nlat,n_channels,minibatch_size)
  output_train = zeros(Float32,Nlon,Nlat,1,minibatch_size)

  # Bundle snapshots together into minibatches
  train_set = [make_minibatch(input_train, output_train, 1:minibatch_size)]

  # Load model and datasets onto GPU, if enabled
  train_set = train_set |> gpu

  # Make sure our model is nicely precompiled before starting our training loop
  CNN_model(train_set[1][1])
  @info("Beginning training loop...")
  for epoch in 1:num_epochs
    @printf("EPOCH: %2i\n",epoch)
    # Permuted training data
    train_idxs = shuffle(0:trainN*numDataset-1)
    mb_idxs = partition(1:trainN*numDataset, minibatch_size)

    # Iterating over every batch
    batch_no = 1
    for index in mb_idxs
      # Load training data
      for i in 1:minibatch_size
        file_input_w = "spectral/Training set "*string(Int(floor(1+train_idxs[index[i]]/trainN)))*"/"*folder*"/05_LES_vorticity/w_"*string(mod(train_idxs[index[i]],trainN)+2000)*".csv"
        file_input_s = "spectral/Training set "*string(Int(floor(1+train_idxs[index[i]]/trainN)))*"/"*folder*"/07_LES_streamfunction/s_"*string(mod(train_idxs[index[i]],trainN)+2000)*".csv"
        input_train[:,:,1,i] = Flux.normalise(readdlm(file_input_w, ',', Float32)[1:end-1,1:end-1])
        input_train[:,:,2,i] = Flux.normalise(readdlm(file_input_s, ',', Float32)[1:end-1,1:end-1])
        file_input_sgs = "spectral/Training set "*string(Int(floor(1+train_idxs[index[i]]/trainN)))*"/"*folder*"/03_subgrid_scale_term/sgs_"*string(mod(train_idxs[index[i]],trainN)+2000)*".csv"
        output_train[:,:,1,i] = Flux.normalise(readdlm(file_input_sgs, ',', Float32)[1:end-1,1:end-1])
      end
      # input_train[:,:,1,:] = Flux.normalise(input_train[:,:,1,:])
      # input_train[:,:,2,:] = Flux.normalise(input_train[:,:,2,:])
      # output_train[:,:,1,:] = Flux.normalise(output_train[:,:,1,:])
      # Find the std of raw sgs from t=1 to t=4, this will normalise the sgs term during LES

      # Bundle snapshots together into minibatches
      train_set = [make_minibatch(input_train, output_train, 1:minibatch_size)]

      # Load model and datasets onto GPU, if enabled
      train_set = train_set |> gpu

      Flux.train!(loss, Flux.params(CNN_model), train_set, opt)
      if mod(batch_no,50)==0
        @printf("Iteration: %3i, MSE = %6.4e\n",batch_no,loss(train_set[1][1],train_set[1][2]))
      end
      batch_no += 1
      # if mod(batch_no,50)==0
      #   CNN_model = cpu(CNN_model)
      #   @save "CNN_model.bson" CNN_model
      #   CNN_model = gpu(CNN_model)
      # end
    end
    CNN_model = cpu(CNN_model)
    @save "CNN_model.bson" CNN_model
    CNN_model = gpu(CNN_model)
    avg_c = 0
    Random.seed!(2)
    ind = shuffle(1:300)
    for i in 1:100
      avg_c = avg_c + 0.01*corr_coeff(test_set[1][:,:,:,ind[i]:ind[i]],test_set[2][:,:,:,ind[i]:ind[i]])
    end
    @printf("Correlation coefficient = %5.4f\n",avg_c)
  end
end

main()
# heatmap(aa,clim=(-7.7,7.7),size=(450,450),axis=nothing,cbar=false,c=:seismic)
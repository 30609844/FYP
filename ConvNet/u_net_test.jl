using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf

C1 = Chain(Conv((3,3), 3=>23, pad=1, relu),Conv((3, 3), 23=>50, pad=1, relu),Conv((3, 3), 50=>50, pad=1, relu))
Down1 = Chain(C1,MaxPool((2,2)))

C2 = Chain(Down1,Conv((3, 3), 50=>100, pad=1, relu),Conv((3, 3), 100=>100, pad=1, relu))
Down2 = Chain(C2,MaxPool((2,2)))

C3 = Chain(Down2,Conv((3, 3), 100=>200, pad=1, relu),Conv((3, 3), 200=>200, pad=1, relu))
Down3 = Chain(C3,MaxPool((2,2)))

C4 = Chain(Down3,Conv((3, 3), 200=>400, pad=1, relu),Conv((3, 3), 400=>400, pad=1, relu))
Down4 = Chain(C4,MaxPool((2,2)))

C5 = Chain(Down4,Conv((3, 3), 400=>800, pad=1, relu),Conv((3, 3), 800=>800, pad=1, relu))
UpConv1 = Chain(C5,Upsample(scale = (2, 2)),ConvTranspose((2,2), 800=>400, stride=2))

C6 = Chain(UpConv1+C4)
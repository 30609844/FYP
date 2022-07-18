using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf

model = Chain(Conv((3,3), 2=>23, pad=1, relu),Conv((3, 3), 23=>50, pad=1, relu),Conv((3, 3), 50=>50, pad=1, relu),Conv((3, 3), 50=>18, pad=1, relu))

## Splines
function sgn(x)
    s = sign.(x)
    s[s.==0] .= 1
    return s
end
function heaviside(x)
    return (sgn(x).+1)./2
end

# 1st order splines
function p1_1(offsets)
    offsets = offsets.*sgn(offsets)
    return (1 .-offsets)
end

p1 = [p1_1]

# 2nd order splines
function p2_1(offsets)
    offsets = offsets.*sgn(offsets)
    return (1 .-offsets).^2 .* (1 .+2*offsets)
end

function p2_2(offsets)
    abs_offsets = offsets.*sgn(offsets)
    return sgn(offsets).*(1 .-abs_offsets).^2 .* (abs_offsets)
end

# derivatives
function dp2_1(offsets)#first derivative (needs to be devided by dt)
	abs_offsets = offsets.*sgn(offsets)
	return sgn(offsets).*(6*abs_offsets.^2 .-6*abs_offsets)
end

function dp2_2(offsets)
	abs_offsets = offsets.*sgn(offsets)
	return 3*abs_offsets.^2 .- 4*abs_offsets .+ 1
end

function d2p2_1(offsets)#2nd derivative (needs to be devided by dt^2)
	abs_offsets = offsets.*sgn(offsets)
	return 12*abs_offsets .- 6
end

function d2p2_2(offsets)
	abs_offsets = offsets.*sgn(offsets)
	return sgn(offsets).*(6*abs_offsets .-4)
end

p2 = [p2_1,p2_2] # list of 2nd order basis splines

# 3rd order splines
function p3_1(offsets)
	offsets = offsets.*sgn(offsets)
	return (1 .-offsets).^3 .*(1 .+ 3*offsets .+ 6*offsets.^2)
end

function p3_2(offsets)
	abs_offsets = offsets.*sgn(offsets)
	return sgn(offsets).*(1 .- abs_offsets).^3 .*(abs_offsets .+ 3*abs_offsets.^2)*2
end

function p3_3(offsets)
	offsets = offsets.*sgn(offsets)
	return (1 .-offsets).^3 .*(0.5*offsets.^2)*16
end

p3 = [p3_1,p3_2,p3_3] # list of 3rd order basis splines

# 4th order splines
function p4_1(offsets)
	return (offsets .- 1).^4 .*(1 .+ 4*offsets .+ 10*offsets.^2 .+ 20*offsets.^3).*heaviside(offsets) .+(-offsets .- 1).^4 .*(1 .-4*offsets .+ 10*offsets.^2 .- 20*offsets.^3) .*heaviside(-offsets)
end

function p4_2(offsets)
	return ((offsets .- 1).^4 .*(1*offsets .+ 4*offsets.^2 .+ 10*offsets.^3).*heaviside(offsets) .+ (-offsets .- 1).^4 .*(1*offsets .- 4*offsets.^2 .+ 10*offsets.^3) .*heaviside(-offsets))*4
end

function p4_3(offsets)
	return ((offsets .- 1).^4 .*(0.5*offsets.^2 .+ 2*offsets.^3).*heaviside(offsets) .+ (-offsets .- 1).^4 .*(0.5*offsets.^2 .- 2*offsets.^3) .*heaviside(-offsets))*32
end

function p4_4(offsets)
	return ((offsets .- 1).^4 .*((1.0/6.0).*offsets.^3).*heaviside(offsets) .+ (-offsets .- 1).^4 .*((1.0/6.0) .*offsets.^3) .*heaviside(-offsets))*512
end

p4 = [p4_1,p4_2,p4_3,p4_4]

# 5th order splines
function p5_1(offsets)
	return ((offsets .- 1).^5 .*(-1 .- 5*offsets .- 15*offsets.^2 .- 35*offsets.^3 .- 70*offsets.^4) .*heaviside(offsets) .+ (-offsets .- 1).^5 .*(-1 .+ 5*offsets .- 15*offsets.^2 .+ 35*offsets.^3 .- 70*offsets.^4) .*heaviside(-offsets))
end

function p5_2(offsets)
	return ((offsets .- 1).^5 .*(-1*offsets .- 5*offsets.^2 .-15*offsets.^3 .- 35*offsets.^4) .*heaviside(offsets) .+ (-offsets .- 1).^5 .*(-1*offsets .+ 5*offsets.^2 .- 15*offsets.^3 .+ 35*offsets.^4) .*heaviside(-offsets))*4
end

function p5_3(offsets)
	return ((offsets .- 1).^5 .*(-0.5*offsets.^2 .- 2.5*offsets.^3 .- 7.5*offsets.^4) .*heaviside(offsets) .+ (-offsets .- 1).^5 .*(-0.5*offsets.^2 .+ 2.5*offsets.^3 .- 7.5*offsets.^4) .*heaviside(-offsets))*32
end

function p5_4(offsets)
	return ((offsets .- 1).^5 .*((-0.5/3.0) .*offsets.^3 .- 2.5/3.0*offsets.^4) .*heaviside(offsets) .+ (-offsets .- 1).^5 .*((-0.5/3.0) .*offsets.^3 .+ (2.5/3.0)*offsets.^4) .*heaviside(-offsets))*512
end

function p5_5(offsets)
	return ((offsets .- 1).^5 .*((-2.5/6.0) .*offsets.^4) .*heaviside(offsets) .+ (-offsets .- 1).^5 .*((-2.5/6.0) .*offsets.^4) .*heaviside(-offsets))*1024
end

p5 = [p5_1,p5_2,p5_3,p5_4,p5_5]

p_i = [p1,p2,p3,p4,p5] # list of lists of basis splines for different orders
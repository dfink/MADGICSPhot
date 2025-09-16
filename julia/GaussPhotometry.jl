using BenchmarkTools
using DataFrames
using Profile
using Random
using ImageTransformations
using Interpolations
using Polynomials

#using Makie
#using CairoMakie


#using ForwardDiff
include("MADGICSPhot-utils.jl")

#include("AppleAccelerateLinAlgWrapper.jl")
include("GaussPhotometryDone.jl")

# test PSF_photometry
#data = Gaussian_model([14.0,14.0], Np=Np, fwhm=3.0)+randn(Np,Np)*0.01
#heatmap(data, aspect_ratio=1, color=:viridis)
#fit = PSF_photometry(data, [10.0,10.0], 3.0, Np=27)
#fit.param



# psf is always centered
# generate Gaussians for now ???
"""
    psf_covariance(Np::Int=33, Nsam::Int=10000; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.1, μ_β=4.8, σ_β=0.1, objtype="Moffat")
Generate covariance matrix for PSF model.

Parameters
----------
Np : Int
	Size of image in pixels.
Nsam : Int
	Number of samples to use for covariance matrix.
σ_cen : Float64
	Standard deviation of centroid position in pixels.
μ_fwhm : Float64
	Mean FWHM in pixels.
σ_fwhm : Float64
	Standard deviation of FWHM in pixels.
μ_β : Float64
	Mean β parameter.
σ_β : Float64
	Standard deviation of β parameter.
objtype : String
	type of object: "Moffat", "Gaussian", or "Both".

Returns

Cstar : Array{Float64,2}
	Covariance matrix for PSF model.	
Notes
"""
function psf_covariance(Np::Int=33, Nsam::Int=10000; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.1, μ_β=4.8, σ_β=0.1, objtype="Moffat")

	# Moffat samples are normalized to sum to 1.0 (roughly)
	if objtype=="Moffat"
	 	psfs  = Moffat_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, smear=false)
	elseif objtype=="Gaussian"
	 	psfs  = Gaussian_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, smear=false)
	elseif objtype=="Both"
		psfs1 = Moffat_model_samples(Np, Nsam÷2; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, smear=false)
		psfs2 = Gaussian_model_samples(Np, Nsam÷2; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, smear=false)
		psfs  = cat(psfs1, psfs2, dims=3)
	end
	
	Dstar = reshape(psfs,Np*Np,:)
	Cstar = (Dstar*Dstar') .* (1.0 / Nsam)

	return Cstar
end


# eigenfactors for a Moffat psf
function psf_efacs_slow(Nvec=20; Np=33, Nsam=20000, σ_cen=0.25, μ_fwhm=nothing, σ_fwhm=0.1, objtype=nothing, β=nothing)

	# padded grid size
	Np2 = Np*2-1

	# generate covariance matrix for padded region (to make shifts easy)
	bigcov = psf_covariance(Np2, Nsam, σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, objtype=objtype, μ_β=β)
	
	# SHOULD pass Dstar to Krylov instead of bigcov!!!!!
	# compute Nvec eigenvectors
	efacs = cov_efacs(bigcov, Nvec=Nvec)
	return efacs
end





# get times for psf_efacs as a function of Np
# This is linear in Nsam
function plot_efac_times()
	Nps = [17, 23, 27, 33, 49]
	efacstimes = zeros(length(Nps))
	efacstimes2 = zeros(length(Nps))
	efacsmof = zeros(length(Nps))
	Nsam = 10000
	for i=1:length(Nps)
		Np = Nps[i]
		efacsmof[i] = @elapsed psf_efacs(20, Np=Np, Nsam=Nsam, σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.1, objtype="Moffat", μ_β=4.8)
		efacstimes[i] = @elapsed psf_efacs(20, Np=Np, Nsam=Nsam, σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.1, objtype="Gaussian")
		efacstimes2[i] = @elapsed psf_efacs_slow(20, Np=Np, Nsam=Nsam, σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.1, objtype="Gaussian")
	end
	# make plot
	p1 = plot(Nps, efacsmof, label="Moffat", line=(2), xlabel="Np", ylabel="time (s)")
	plot!(p1, Nps, efacstimes, label="Gaussian", line=(:dash,2), title="PSF prior, 10,000 samples")
	#plot!(p1, Nps, efacstimes2, label="psf_efacs_slow", xlabel="Np", ylabel="time (s)", title="PSF eigenfactor generation")
	
	return p1
end

#p1 = plot_efac_times()
#savefig(p1, "efactime.pdf")

function plot_psf_model_times()
	Nps = [17, 23, 27, 33, 49, 65]
	Gausstimes = zeros(length(Nps))
	Moftimes = zeros(length(Nps))
	for i=1:length(Nps)
		Np  = Nps[i]
		cen = [Np÷2+1.0,Np÷2+1.0]
		println("Np: ",Np)
		info = @benchmark imgs = Gaussian_model_samples($Np, 1000; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.05, smear=false)
		Gausstimes[i] = median(info.times)
		info = @benchmark imgs = Moffat_model_samples($Np, 1000; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.05, μ_β=4.8, σ_β=0.1, smear=false)
		Moftimes[i] = median(info.times)
	end

	# make plot with Moffat first
	p1 = plot(Nps, Moftimes/1e6, label="Moffat", line=(2))
	plot!(p1, Nps, Gausstimes/1e6, label="Gaussian", xlabel="Np", line=(:dash,2), ylabel="time (μs)", title="PSF model generation")

	return p1
end

#p1 = plot_psf_model_times()
#savefig(p1, "psfmodeltime.pdf")

# actually can do more than one star
function onestar_deleteme(V, B, Binv, starcens, cens; Np = 33, σ_pix = 0.003, fwhm = 2.5, seed=nothing)
	Random.seed!(seed)     # seed=nothing is a random seed
	Nobj  = length(cens)
	Ncomp = Nobj+1

	# mock image is noise plus star
	#	comp = zeros(Np,Np,Ncomp)   # components (including noise)


	img = randn(Np,Np) .* (σ_pix /1)
	#	comp[:,:,end] = img
	for k=1:length(starcens)
		#comp[:,:,k] = Moffat_model(starcens[k]; Np=Np, fwhm=fwhm, β=4.8)
		img += Gaussian_model(starcens[k]; Np=Np, fwhm=fwhm)
	end

	# build inverse covariance
	#Cinv = copy(Binv)
	W = []
	#igrid = falses(Np,Np)
	#Nh= Np÷2+1
	#igrid[Nh-7:Nh+7,Nh-7:Nh+7] .= true
	#ind = findall(reshape(igrid,length(igrid)))


	x_d = vec(img)
	Ndim = size(Binv,1)
	CinvXd = Binv*x_d
	CinvV  = Binv*V

	M = (I + (V'*CinvV))
	CinvXd .-= CinvV*(M\(CinvV'*x_d))

	μ_comp = Array{Float64, 3}(undef,Np,Np,Ncomp)
	μ_comp[:,:,1] = V*(V' * CinvXd)

	return μ_comp
end


function testsomething()
	# set BLAS threads to 4
	BLAS.set_num_threads(4)

	μ_comp = Array{Float64, 3}(undef,Np,Np,Nimg)
	@isdefined(μ_comp) 
	foo = MADGICS_single_star(Binv, V, img)        # 370 usec for 100 stars
	bar = MADGICS_single_star2(Binv, V, img)       # 220 usec for 100 stars
	MADGICS_single_star!(μ_comp, Binv, V, img)     # 110 usec for 100 stars
	MADGICS_single_star_acc!(μ_comp, Binv, V, img) # 80 usec for 100 stars


	# set BLAS threads to 1  (it is actually faster??)
	BLAS.set_num_threads(1)

	foo = MADGICS_single_star(Binv, V, img)  # 350 usec for 100 stars
	bar = MADGICS_single_star2(Binv, V, img) # 200 usec for 100 stars
	MADGICS_single_star!(μ_comp, Binv, V, img) # 105 usec for 100 stars
	MADGICS_single_star_acc!(μ_comp, Binv, V, img) # 72 usec for 100 stars
	return
end


function precision_x_plot(efacs; σ_pix=0.003, σ_back=0.01, fwhm=fwhm, seed=nothing)
	Np = round(Int,sqrt(size(efacs,1)))÷2+1  # assume square
	Ndim = Np*Np
	Nphalf = Np÷2+1

	# variance of dx, dy efacs
	var_dx = mean(sum(efacs[:,2:3].^2,dims=1))

	# background covariance
	#B    = diagm(ones(Ndim).*σ_pix^2) .+ σ_back^2
	fudge = 1
	#B    = diagm(ones(Ndim).*(fudge*σ_pix)^2) .+ σ_back^2
	#Binv = inv(cholesky(B))
    B    = Diagonal(ones(Ndim).*(fudge*σ_pix)^2)
	Binv = inv(B)
	Δx   = collect(-2:0.0025:2)
	Ntrial = length(Δx)
	κ    = [1.0]
	labels = ["κ = 0.1", "κ = 1.0","κ = 10"]
	Nκ   = length(κ)
	flux = zeros(Ntrial, Nκ)
	dx   = zeros(Ntrial, Nκ)
	for j=1:Nκ
		for i=1:Ntrial
			fcomp = onestar(efacs.*κ[j], B, Binv, [[Nphalf+Δx[i],Nphalf]], [[Nphalf,Nphalf]], Np=Np, σ_pix=σ_pix, fwhm=fwhm, seed=seed)
			fpar  = comp_params(fcomp)
			flux[i,j] = fpar[1].flux
			dx[i,j]   = fpar[1].x - Nphalf
		end
	end
	mcolor = [:red2,:green3,:blue2]
	var_dx = mean(sum(efacs[:,2:3].^2,dims=1))
	fac = (fudge*σ_pix)^2/(var_dx + (fudge*σ_pix)^2)
	p = plot([-1.5,1.5],[-1.5,1.5].*(-fac),linecolor=:blue,linewidth=1,label=false,legend=:topleft,
			xlabel="True Δx [pix]", ylabel="Recovered x error [pix]")

	for j=Nκ:-1:1
		#scatter!(p, Δx, dx[:,j], m=(mcolor[j],2,0.65),label=labels[j])
		xerr = dx[:,j]-Δx
		scatter!(p, Δx, xerr, m=(mcolor[j],2,0.65),label=nothing)
		wh = findall(abs.(Δx) .< 0.5)
		sigerr = std(xerr[wh])
		println(sigerr)
	end
	# p = plot([-2,2],[1.0,1.0],linecolor=:blue,linewidth=4,label=false,
	# 	xlabel="Δx [pixels]",ylabel="flux (recovered/true)")
	# for j=Nκ:-1:1
	# 	scatter!(p, Δx, flux[:,j], m=(mcolor[j],5,0.65),ylims=(.6,1.15),label=labels[j])
	# end

	return p
end


function precision_flux_plot_old(efacs; σ_pix=0.003, σ_back=0.01, fwhm=fwhm, seed=nothing)
	Np = round(Int,sqrt(size(efacs,1)))÷2+1  # assume square
	Ndim = Np*Np
	Nphalf = Np÷2+1

	covfaclist = covfactor_list(cens, efacs, Ndim=Ndim)
	V          = reshape(covfaclist,Ndim,size(covfaclist,2))

	# variance of dx, dy efacs
	#var_dx = mean(sum(efacs[:,2:3].^2,dims=1))

	# background covariance
	#B    = diagm(ones(Ndim).*σ_pix^2) .+ σ_back^2
	fudge = 1.0
	B    = diagm(ones(Ndim).*(fudge*σ_pix)^2) .+ σ_back^2
	Binv = inv(cholesky(B))
	Δx   = collect(-1:0.001:1)
	Ntrial = length(Δx)
	κ    = [1.0]
	labels = ["κ = 0.1", "κ = 1.0","κ = 10"]
	Nκ   = length(κ)
	flux = zeros(Ntrial, Nκ)
	dx   = zeros(Ntrial, Nκ)
	for j=1:Nκ
		for i=1:Ntrial
			fcomp = onestar(V.*κ[j], B, Binv, [[Nphalf+Δx[i],Nphalf]], [[Nphalf,Nphalf]], Np=Np, σ_pix=σ_pix, fwhm=fwhm, seed=seed)
			fpar  = comp_params(fcomp)
			#pp=heatmap(fcomp[:,:,1],clims=(-0.1,0.3))
			#display(pp)
			flux[i,j] = fpar[1].flux
			dx[i,j]   = fpar[1].x - Nphalf
		end
	end
	mcolor = [:red2,:green3,:blue2]
	var_flux = sum(efacs[:,1].^2)
	fac = (fudge*σ_pix)^2/(var_flux + (fudge*σ_pix)^2)
	println("flux fac: ", fac)
	p = plot([-1.5,1.5],[1.0,1.0],linecolor=:blue,linewidth=1,label=false,legend=:topleft,
			xlabel="True Δx [pix]", ylabel="Recovered flux error")

	for j=Nκ:-1:1
		#scatter!(p, Δx, dx[:,j], m=(mcolor[j],2,0.65),label=labels[j])
		xerr = dx[:,j]-Δx
		scatter!(p, Δx, flux[:,j], m=(mcolor[j],2,0.65),label=nothing)
		#wh = findall(abs.(Δx) .< 0.5)
		#sigerr = std(xerr[wh])
		#println(sigerr)
		goo = reshape(efacs[:,1],Np*2-1,Np*2-1)
		rat = sum(circshift(goo,(1,0)) .* goo) / sum(goo.*goo)
		plot!(p, Δx, 1.0 .- (1-rat).*Δx.^2, label=nothing)
	end
	# p = plot([-2,2],[1.0,1.0],linecolor=:blue,linewidth=4,label=false,
	# 	xlabel="Δx [pixels]",ylabel="flux (recovered/true)")
	# for j=Nκ:-1:1
	# 	scatter!(p, Δx, flux[:,j], m=(mcolor[j],5,0.65),ylims=(.6,1.15),label=labels[j])
	# end

	return p
end



function precision_flux_plot(efacs; σ_pix=0.003, σ_cen=nothing, σ_back=0.01, 
	fwhm=nothing, Ntrial=10000, seed=nothing, objtype=nothing)

	Np = round(Int,sqrt(size(efacs,1)))÷2+1  # assume square
	Ndim = Np*Np
	Nphalf = Np÷2+1

	fudge = 1.0

	B    = Diagonal(ones(Ndim).*(fudge*σ_pix)^2) # .+ σ_back^2
	Binv = inv(cholesky(B))
	Δx   = [0.0,0.5]
	Nx   = length(Δx)
	flux = zeros(Ntrial, Nx)
	dx   = zeros(Ntrial, Nx)
	ddx  = randn(Ntrial).*σ_cen
	ddy  = randn(Ntrial).*σ_cen
	# there are 2 offsets here.  
	# Δx tests response to systematic offset from prior.  ddx expresses the prior uncertainty.
	# loop over x offsets Δx

	# allocate undefined array
	fcomp = Array{Float64, 2}(undef,Np*Np,Ntrial)
	apflux = zeros(Ntrial, Nx)
	apfluxerr = zeros(Ntrial, Nx)

	for j=1:Nx
		# V is psf prior centered at center pixel
		covfaclist = covfactor_list([[Nphalf,Nphalf]], efacs, Ndim=Ndim)
		V          = reshape(covfaclist,Ndim,size(covfaclist,2))

		# generate image cube with offsets applied
		objxy = [[Nphalf+ddx[i]+Δx[j],Nphalf+ddy[i]+Δx[j]] for i=1:Ntrial]
		img = generate_obj_cube(objxy; Np=Np, σ_pix=σ_pix, fwhm=fwhm, seed=seed, objtype=objtype)

		# compute MADGICS posterior for each star in img cube
		#fcomp = MADGICS_single_star2(Binv, V, img)
		MADGICS_single_star!(fcomp, Binv, V, img)

		# and extract flux and position
		fpar = comp_x_and_flux(reshape(fcomp,Np,Np,Ntrial))
		flux[:,j] = fpar.flux
		dx[:,j]   = fpar.x .- Nphalf

		# compute aperture photometry for each star in img cube
		apflux[:,j], apfluxerr[:,j] = simple_aperture_photometry(img, 5, (6,8))

		for k=1:4 
			fc = reshape(fcomp[:,k],Np,Np)
			outarr = [img[:,:,k]  fc  img[:,:,k]-fc]
			q=heatmap(outarr,aspect_ratio=1.0,size=(900,300), clims=(-0.05,0.1))
			display(q)
		end
		fc = reshape(mean(fcomp,dims=2),Np,Np) 
		si = mean(img,dims=3)[:,:,1]
		outarr = [si  fc  10 .*(si-fc)]
		q = heatmap(outarr,aspect_ratio=1.0,size=(900,300), clims=(-0.05,0.1))
		display(q)

	end
	#mcolor = [:red2,:green3,:blue2]
	var_flux = sum(efacs[:,1].^2)
	fac = (fudge*σ_pix)^2/(var_flux + (fudge*σ_pix)^2)
	println("flux fac: ", fac)
	labels = [ @sprintf("Ma med:%6.4f, μ:%6.4f, σ:%6.4f",median(flux[:,k]),mean(flux[:,k]),std(flux[:,k])) for k=1:Nx ]

	histbins=0.0:0.02:2.0
	p = stephist(flux[:,1],label=labels[1], bins=histbins, normalize=true, xlabel="flux",legend=:topleft)
	for k=2:Nx
		stephist!(p,flux[:,k],label=labels[k], bins=histbins, normalize=true)
	end

	# add aperture phometry histogram
	labels = [ @sprintf("Ap med:%6.4f, μ:%6.4f, σ:%6.4f",median(apflux[:,k]),mean(apflux[:,k]),std(apflux[:,k])) for k=1:Nx ]
	for k=1:Nx
		stephist!(p,apflux[:,k],label=labels[k], bins=histbins, normalize=true)
	end

	#p = plot([-1.5,1.5],[1.0,1.0],linecolor=:blue,linewidth=1,label=false,legend=:topleft,
				#xlabel="True Δx [pix]", ylabel="Recovered flux error")

	result = (Ntrial=Ntrial, Δx=Δx, flux=flux, ddx=ddx, dx=dx, apflux=apflux, apfluxerr=apfluxerr)
	return p, result
end


# This takes ~ 1 ms, 6 ms for Lanczos
function V_subpixel_shift(V, Δx, Δy; interp_type="Cubic")
	# set up cubic interpolation function 
	Ndim,Nvec = size(V)
	Np     = round(Int,sqrt(Ndim))
	Vcub   = reshape(V,Np,Np,Nvec)
	Vout   = copy(V)

	# set up cubic interpolation function   (Should we use Lanczos3 instead?)
	for i=1:Nvec
		if interp_type == "Cubic"
			itp = cubic_spline_interpolation((1:Np, 1:Np), Vcub[:,:,i], extrapolation_bc = Line())
		elseif interp_type == "Lanczos"
			itp = Interpolations.interpolate(Vcub[:,:,i], Interpolations.Lanczos())
		end
		yr = collect((1-Δy):(Np-Δy))
		yind = findall(yr .>= 1 .&& yr .<= Np)

		xr = collect((1-Δx):(Np-Δx))
		xind = findall(xr .>= 1 .&& xr .<= Np)

		v2d = zeros(Np,Np)
		for yy in yind
			for xx in xind
				v2d[yy,xx] = itp(yr[yy],xr[xx])
			end
		end
#		v2d[yind,xind] = itp(yr[yind],xr[xind])[:]

		Vout[:,i] = v2d[:]
	end
	return Vout
end




# this takes ?? ms
function position_bias_model(V)

	function comp_diff(Np, Vpsf, dp)
		fparm = comp_params(reshape(Vpsf-dp,Np,Np,1))
		fparp = comp_params(reshape(Vpsf+dp,Np,Np,1))
		Δpos = sqrt((fparm.x[1] - fparp.x[1])^2 + (fparm.y[1] - fparp.y[1])^2)/2
		return Δpos
	end

	# assume position shift corresponds to eigenvectors 2 and 3
	Vpsf = V[:,1]

	eps = 0.1
	Δ1 = comp_diff(Np, Vpsf, V[:,2].*eps) / eps
	Δ2 = comp_diff(Np, Vpsf, V[:,3].*eps) / eps
	Δpos = (Δ1 + Δ2)/2.0

	Vec2 = (sum(V[:,2].^2) + sum(V[:,3].^2))/2.0
	return Vec2,Δpos
end

position_bias_model(V)


# this is 0.5 msec
function flux_bias_model(V)

	Vpsf = V[:,1]
	Vpsf ./= (Vpsf ⋅ Vpsf)
	dx = -0.5:0.1:0.5
	Vdot = zeros(length(dx))
	for i = 1:length(dx)
		Vshift = V_subpixel_shift(V[:,1:1], dx[i], 0.0)
		Vdot[i] = Vpsf ⋅ Vshift
	end

	# fit parabola using package Polynomials

	parab = fit(dx, Vdot, 2)

	# plot
	# p = plot(dx, Vdot, label="V⋅Vshift", xlabel="Δx [pixels]", ylabel="V⋅Vshift", title="V⋅Vshift vs Δx")
	# plot!(p, dx, parab.(dx), label="parabola", linecolor=:black, linestyle=:dash)
	# display(p)
	return parab
end

# Np=27
# Ndim = Np*Np
# Nphalf = Np÷2+1
# efacs = psf_efacs(8; Np=27, σ_cen=0.25, μ_fwhm=2.5, μ_β=4.8, objtype="Gaussian")
# covfaclist = covfactor_list([[Nphalf,Nphalf]], efacs, Ndim=Ndim)
# V         = reshape(covfaclist,Ndim,size(covfaclist,2))
# parab = flux_bias_model(V)




function test_V_subpixel_shift()
	Np = 27
	Ndim = Np*Np
	Nphalf = Np÷2+1
	covfaclist = covfactor_list([[Nphalf,Nphalf]], efacs, Ndim=Ndim)
	V1         = reshape(covfaclist,Ndim,size(covfaclist,2))

	Vout = V_subpixel_shift(V1, 0.25,0)

	eigenimage_figure(V1*V1' + 1e-6*I; panels=[4,3], binfac=4, stretch=0.3)
	eigenimage_figure(Vout*Vout' + 1e-6*I; panels=[4,3], binfac=4, stretch=0.3)
end



function reweight_eigenvalues!(fcomp, V, σ_pix)
	Vsq = sum(V.*V,dims=1)
	# clamp the reweighting factor fac to 0.25 to 1.0
	fac = max.(0.1, Vsq ./ (Vsq .+ σ_pix^2))
	alpha = inv(V'*V)*(V'*fcomp)
	alpha ./= fac'
	mul!(fcomp, V, alpha)
	#println("Reweighting eigenvalues with σ_pix: ", σ_pix, " fac: ", fac)
	return
end




function try2again(fwhm; Nvec=8, Np=33, frac_σ_cen=1/128, efacsNsam=10000, σ_fwhm = 0.02, σ_pix = 0.1, Nsam=100, Δx=0.0, xrange = -1.0:0.1:1.0, fluxvals=[1.0,2.0],
		objtype="Gaussian", objtype_prior="Gaussian", uniform=false, Niter=1, reweight=true)

	V1fac = 80.0 #fluxvals[1]
	V2fac = 80.0 #fluxvals[2]
	σ_cen = (fwhm/2.355) * frac_σ_cen
	
	# get efacs for a single star prior
	efacs = psf_efacs(Nvec; Np=Np, Nsam=efacsNsam, σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, objtype=objtype_prior, uniform=uniform)
	Neff = 1.0/(sum(efacs[:,1].^2))

	Ndim = Np*Np
	Nphalf = Np÷2+1
	covfaclist = covfactor_list([[Nphalf+Δx,Nphalf]], efacs, Ndim=Ndim)
	V1         = reshape(covfaclist,Ndim,size(covfaclist,2))

	# overwrite σ_cen for the mock data
	σ_cen = 0.0

	# determine polynomial for flux bias correction (as a function of subpixel offset)
	fbpoly = flux_bias_model(V1)

	# obtain Nsam PSF samples for mock data
	if objtype=="Moffat"
		psf1  = Moffat_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, Δx=Δx, smear=false)
	elseif objtype=="Gaussian"
		psf1  = Gaussian_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, Δx=Δx, smear=false)
	end
	

	# allocate undefined array for components
	fcomp1 = Array{Float64, 2}(undef,Np*Np,Nsam)
	fcomp2 = Array{Float64, 2}(undef,Np*Np,Nsam)
	#apflux = zeros(Nsam, Nx)
	#apfluxerr = zeros(Nsam, Nx)
	
	# allow background float or not?
	Binv = inv(cholesky(Diagonal(ones(Ndim).*(σ_pix)^2))) 
	#B    = I*(σ_pix)^2 
	#Binv = inv(cholesky(B))
#	println("B: ", size(B))

	SNR = fluxvals[2]/σ_pix/sqrt(Neff)
	println("SNR: ", SNR)

	nx = length(xrange)
	meanflux = zeros(nx)
	stdflux  = zeros(nx)
	meanx    = zeros(nx)
	stdx     = zeros(nx)
	meanΔχ2  = zeros(nx)
	stdΔχ2   = zeros(nx)

	# loop over subpixel offsets
	for i=1:nx
		if objtype=="Moffat"
			psf2 = Moffat_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, Δx=xrange[i], smear=false)
		elseif objtype=="Gaussian"
			psf2 = Gaussian_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, Δx=xrange[i], Δy=0.0, smear=false)
		end
		covfaclist = covfactor_list([[Nphalf+xrange[i],Nphalf]], efacs, Ndim=Ndim)
		V2         = reshape(covfaclist,Ndim,size(covfaclist,2))
		img        = fluxvals[1]*psf1 .+ fluxvals[2] .*psf2 .+ σ_pix .* randn(Np,Np,Nsam)

		#display(heatmap(reshape(V2[:,1],Np,Np),title="V2 i=$i"))
		#display(heatmap(img[:,:,1]))
		# compute MADGICS posterior for each star in img cube
		xint = round(Int, xrange[i])
#		println("Scaling V1 and V2")
		Δχ2 = MADGICS_double_star!(fcomp1, fcomp2, Binv, V1fac .*V1, V2fac .*V2, 0, 0, xint, 0, img, σ_pix=σ_pix, Niter=Niter, doPCA=false, fbpoly=fbpoly)
		# and extract flux and position
		#fpar1 = comp_params(fcomp1)

		#display(heatmap(img[:,:,1] - reshape(fcomp2[:,1],Np,Np),title="fcomp2 i=$i"))

		if reweight
			reweight_eigenvalues!(fcomp2, V2fac .*V2, σ_pix)
		end
		fpar2 = comp_params(reshape(fcomp2, Np, Np, Nsam))

		meanx[i]    = mean(fpar2.x) .- Nphalf
		stdx[i]     = std(fpar2.x)
	
		corrected_flux = fpar2.flux  # ./ fbpoly.(fpar2.x .- Nphalf .-xint)
		meanflux[i] = mean(corrected_flux)
		stdflux[i]  = std(corrected_flux)

		meanΔχ2[i] = mean(Δχ2)
		stdΔχ2[i]  = std(Δχ2)

	end

	return meanflux, stdflux, meanx, stdx, meanΔχ2, stdΔχ2
end



function twostar_flux_bias_plot(;Np=27, σ_pix=0.1, fwhm=2.5, flux1=10, flux2=[10.0,20.0], Niter=1, reweight=true, seed=nothing)

	# set up BLAS threads
	BLAS.set_num_threads(1)
	
	# set up plot
	p1 = plot(xlabel="Δx [pixels]", ylabel="flux", title="Recovered flux for star 2", size=(800,500), legend=:topleft)
	p2 = plot(xlabel="Δx [pixels]", ylabel="dx [pixels]", title="Recovered offset for star 2", size=(800,500), legend=:topleft, ylims=(-0.6,0.6))
	p3 = plot(xlabel="Δx [pixels]", ylabel="Δχ2", title="Δχ2 for star 2", size=(800,500), legend=:topleft)
	# run sims for pairs of stars with varying SNR
	#xr = -5.0:0.1:5.0
	xr =  -1.55:0.1:5.0
	for i = 1:length(flux2)
		println("flux2: ", flux2[i])
		meanflux, stdflux, meanx, stdx, meanΔχ2, stdΔχ2 = try2again(fwhm; Nvec=8, Np=Np, frac_σ_cen=1/4, efacsNsam=10000, 
   			σ_fwhm = 0.02, σ_pix = σ_pix, Nsam=400, Δx=0.0, fluxvals=[flux1,flux2[i]], xrange = xr, uniform=true, Niter=Niter, reweight=reweight, objtype_prior="Gaussian")
		plot!(p1, xr, meanflux, ribbon=stdflux, fillalpha=0.3, label="flux $(flux2[i])", legend=:topright)
		plot!(p1, xr, xr.*0 .+flux2[i], linecolor=:black, linestyle=:dash, label=false)
		plot!(p2, xr, meanx.-xr, ribbon=stdx, fillalpha=0.3, label="f= $(flux2[i])", legend=:topright)
		plot!(p2, xr, xr.*0, linecolor=:black, linestyle=:dash, label=false)
		plot!(p3, xr, meanΔχ2, ribbon=stdΔχ2, fillalpha=0.3, label="f= $(flux2[i])", legend=:topleft)
		plot!(p3, xr, xr.*0 .+25, linecolor=:black, linestyle=:dash, label=false)
		
	end


	return p1, p2, p3
end
# run sims for pairs of stars with varying SNR




p1,p2,p3 = twostar_flux_bias_plot(Np=27, σ_pix=0.51, fwhm=2.5, flux1=10, flux2=[20.0,10.0], Niter=1, reweight=true)
pgrid = plot(p1,p2,p3,layout=(3,1),size=(600,800))

savefig(pgrid, "twostar_flux_bias_plot_2.5_10_20_10.pdf")

p1,p2,p3 = twostar_flux_bias_plot(Np=27, σ_pix=0.351, fwhm=4.0, flux1=10, flux2=[20.0,10.0], Niter=1)
pgrid = plot(p1,p2,p3,layout=(3,1),size=(800,1000))
savefig(pgrid, "twostar_flux_bias_plot_4.0_10_20_10.pdf")

p1,p2,p3 = twostar_flux_bias_plot(Np=27, σ_pix=0.8, fwhm=1.5, flux1=10, flux2=[20.0,10.0], Niter=1)
pgrid = plot(p1,p2,p3,layout=(3,1),size=(800,1000))
savefig(pgrid, "twostar_flux_bias_plot_1.5_10_20_10.pdf")


function two_star_SNR1vs2(SNRs, fwhm; Nvec=8, Np=33, frac_σ_cen=1/128, efacsNsam=10000, σ_fwhm = 0.02, σ_pix = 0.1, Nsam=1000, Δx=1.0,
	objtype="Gaussian", objtype_prior="Gaussian", scale_covar=false)
	
	nSNR = length(SNRs)

	# Initialize output data structure
	res = nothing
	Δx_1 = -0.25
	Δx_2 = Δx
	σ_cen = (fwhm/2.355) * frac_σ_cen
	
	# get efacs for a single star prior
	efacs = psf_efacs(Nvec; Np=Np, Nsam=efacsNsam, σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, objtype=objtype_prior)
	Neff = 1.0/(sum(efacs[:,1].^2))

	Ndim = Np*Np
	Nphalf = Np÷2+1
	covfaclist = covfactor_list([[Nphalf+Δx_1,Nphalf]], efacs, Ndim=Ndim)
	V1         = reshape(covfaclist,Ndim,size(covfaclist,2))
	covfaclist = covfactor_list([[Nphalf+Δx_2,Nphalf]], efacs, Ndim=Ndim)
	V2         = reshape(covfaclist,Ndim,size(covfaclist,2))

	println("V1: ", size(V1))
	println("V2: ", size(V2))

	# obtain Nsam PSF samples for mock data
	if objtype=="Moffat"
		psf1  = Moffat_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, Δx=Δx_1, smear=false)
		psf2  = Moffat_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, Δx=Δx_2, smear=false)
	elseif objtype=="Gaussian"
		psf1  = Gaussian_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, Δx=Δx_1, smear=false)
		psf2  = Gaussian_model_samples(Np, Nsam; σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, Δx=Δx_2, smear=false)
	end
	




	# allocate undefined array for components
	fcomp = Array{Float64, 2}(undef,Np*Np,Nsam)
	#apflux = zeros(Nsam, Nx)
	#apfluxerr = zeros(Nsam, Nx)
	
	#Binv = inv(cholesky(Diagonal(ones(Ndim).*(σ_pix)^2))) 
	B    = V2*V2' + I*(σ_pix)^2 
	Binv = inv(cholesky(B))
#	println("B: ", size(B))
	flux = zeros(nSNR, nSNR, Nsam)
	dx   = zeros(nSNR, nSNR, Nsam)
	V1_fac = 1.0
	V2_fac = 1.0
	fluxtrue = zeros(nSNR, nSNR)
	# Loop over SNR
	for j = 1:nSNR	
		SNR2 = SNRs[j]
		flux2 = σ_pix * sqrt(Neff) * SNR2
		if scale_covar
			V2_fac = flux2
		end
		B    = (V2_fac^2).*(V2*V2') + I*(σ_pix)^2 
		Binv = inv(cholesky(B))
		for i = 1:nSNR
			SNR1 = SNRs[i]
			# compute fluxes
			flux1 = σ_pix * sqrt(Neff) * SNR1
			if scale_covar
				V1_fac = flux1
			end
			
			fluxtrue[j,i] = flux1
			img = flux1 .* psf1 + flux2 .* psf2 + σ_pix .* randn(Np,Np,Nsam)

			# compute MADGICS posterior for each star pair in img cube
			MADGICS_single_star!(fcomp, Binv, V1_fac.*V1, img)

			# and extract flux and position
			fpar = comp_x_and_flux(reshape(fcomp,Np,Np,Nsam))
			flux[j,i,:] = fpar.flux
			dx[j,i,:]   = fpar.x .- Nphalf

			# compute aperture photometry for each star in img cube
			#apflux[:,j], apfluxerr[:,j] = simple_aperture_photometry(img, 5, (6,8))

		end	
	end



	return flux, dx, fluxtrue
end


Nsam = 100
Np = 27
fac = 0.875
SNRs = 5:30
Δx = 1.4
flux, dx, fluxtrue = two_star_SNR1vs2(SNRs, 2.5, Nvec=8, Np=Np,Nsam=Nsam, frac_σ_cen=0.25, Δx=Δx, scale_covar=true)

#heat map with x and y axis labeled with xrange and yrange
p=heatmap(SNRs, SNRs, sum(flux,dims=3)[:,:,1]./Nsam);
plot!(p,aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="Flux  Δx=$Δx")

heatmap(SNRs, SNRs, fluxtrue);
plot!(aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="True Flux")

#heatmap(SNRs, SNRs, sum(flux,dims=3)[:,:,1]./(Nsam*fac) ./ fluxtrue)
# overplot contour of [0.9, 1.0, 1.1]
p4=contourf(SNRs, SNRs, sum(flux,dims=3)[:,:,1]./(Nsam*fac) ./ fluxtrue, levels=collect(0.9:0.05:1.8), clabels=true, line_z=true, linecolor=[:blue],linewidth=2);
plot!(aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="Flux ratio  Δx=$Δx")

p5=contourf(SNRs, SNRs, mean(dx,dims=3)[:,:,1], levels=10, clabels=true, line_z=true, linecolor=[:blue],linewidth=2)


p5=heatmap(SNRs, SNRs, mean(dx,dims=3)[:,:,1])




for Δx in [2.0]
	flux, dx, fluxtrue = two_star_SNR1vs2(SNRs, 2.5, Nvec=8, Np=Np,Nsam=Nsam, Δx=Δx)
	p4=contourf(SNRs, SNRs, sum(flux,dims=3)[:,:,1]./(Nsam*fac) ./ fluxtrue, levels=collect(0.9:0.05:1.3), clabels=true, line_z=true, linecolor=[:blue],linewidth=2);
	plot!(p4,aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="Flux ratio  Δx=$Δx")
	display(p4)
end


heatmap(std(flux,dims=3)[:,:,1])

plot!(aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="Flux std")


heatmap(sum(dx,dims=3)[:,:,1]./Nsam)

heatmap(std(dx,dims=3)[:,:,1].*1000)
plot!(aspect_ratio=1.0, xlabel="SNR1", ylabel="SNR2", title="dx std [mpix]")

Np = 27
Nvec = 20
fwhm=2.5
efacsNsam = 10000
σ_cen = 0.25

efacs = psf_efacs(Nvec; Np=Np, Nsam=efacsNsam, σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=0.1, objtype="Gaussian")


#Profile.clear()

function runsims(SNRs, FWHMs; Nvec=8, Np=33, frac_σ_cen=1/128, efacsNsam=40000, σ_fwhm = 0.02, σ_back=0.01, Ntrial=10000,
	objtype="Gaussian", objtype_prior="Gaussian")

	# Initialize plot list
	ps = []
	# Initialize output data structure
	res = nothing
	flux = 1.0
	
	# Loop over fwhms
	for fwhm in FWHMs
		σ_cen = (fwhm/2.355) * frac_σ_cen
		efacs = psf_efacs(Nvec; Np=Np, Nsam=efacsNsam, σ_cen=σ_cen, μ_fwhm=fwhm, σ_fwhm=σ_fwhm, objtype=objtype_prior)
#		gg = efacs .- mean(efacs,dims=1)
#		gg[:,1] = efacs[:,1]
# do not mean subtract
#		efacs = copy(gg)
		mylam = sum(efacs.^2,dims=1)
		Neff = 1.0/(sum(efacs[:,1].^2))  # the first eigenfactor is essentially the mean PSF, so this approximates the effective number of pixels.

		for SNR in SNRs  # loop over signal-to-noise ratios
			σ_pix = flux / (SNR*sqrt(Neff))   # noise per pixel based on Neff and SNR
			dof_wt = mylam ./ (σ_pix^2 .+ mylam)
			println("FWHM: ",fwhm,"   SNR:",SNR,"   σ_pix:",σ_pix)
			dof_active = sum(dof_wt[2:end])
			println("dof_wt: ",dof_wt, "        active:",dof_active)
			p, stats = precision_flux_plot(efacs, σ_pix=σ_pix, σ_cen=σ_cen, σ_back=σ_back, fwhm=fwhm, Ntrial=Ntrial, objtype=objtype)
			df = DataFrame(μ_fwhm  =[fwhm],
					   	   σ_fwhm  =[σ_fwhm],
						   σ_cen   =[σ_cen],
					       Np      =[Np],
						   SNR     =[SNR],
						   σ_pix   =[σ_pix],
						   Δx      =[stats.Δx],
						   dx      =[stats.dx],
						   ddx     =[stats.ddx],
						   Ntrial  =[stats.Ntrial],
						   Neff    =[Neff],
						   dofwt   =[(dof_wt[2]+dof_wt[3])/2.0],            #????
						   fmean   =[mean(stats.flux[:,1])],
						   fmed    =[median(stats.flux[:,1])],
						   fstd    =[std(stats.flux[:,1])],
						   apmean  =[mean(stats.apflux[:,1])],
						   apmed   =[median(stats.apflux[:,1])],
						   apstd   =[std(stats.apflux[:,1])],
						   objtype =[objtype],
						   objtype_prior=[objtype_prior]
			              )
						  # should we add a field for overlap integral?
			if isnothing(res)
				res = df
			else
				push!(res,df[1,:])
			end
			ymax = maximum(p.series_list[2][:y])
			plot!(p,xlims=(0,2),ylims=(0,ymax*1.8))
			push!(ps, p)
		end
	end
	return res,ps
end


#Juno.profiler()
#myplot = plot(ps..., layout=(3,3), size=(1000,700))




# SNRs = 5:20
# FWHMs = [2.0,3.0]
# res, ps = runsims(SNRs, FWHMs, Nvec=8, frac_σ_cen=1/128)

 xx = 5:20
# p1=fluxbias_SNR_plot(res);
# p2=fluxstd_SNR_plot(res);
# myplot = plot([p1,p2]..., layout=(1,2), size=(800,500))
# savefig(myplot,"fluxbias.png")

SNRs = [5,10,20]
FWHMs = [2.0,3.0]
Ntrial = 10000
res1, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=Ntrial, σ_fwhm=0.2, objtype="Gaussian", objtype_prior="Gaussian")
myplot = plot(ps..., layout=(3,3), size=(1000,1200))

res2, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=Ntrial, objtype="Gaussian", objtype_prior="Moffat")
myplot = plot(ps..., layout=(3,3), size=(1000,1200))

res3, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=Ntrial, σ_fwhm=0.2, objtype="Moffat", objtype_prior="Gaussian")
myplot = plot(ps..., layout=(3,3), size=(1000,1200))

res4, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=Ntrial, objtype="Moffat", objtype_prior="Moffat")
myplot = plot(ps..., layout=(3,3), size=(1000,1200))

frac_σ_cen

# 14.2 for 10000, 19.2 for 20000 29.8 for 40000
# 12.5 for 10000, 15.4 for 20000 21.3 for 40000
# 12.2 for 10000                  20 for 40000

myplot = plot(ps..., layout=(3,3), size=(1000,1200))


SNRs = [5,6,7,8,10,15,20,30,50,100]
FWHMs = [1.3,1.5]
res, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=100000)


myplot = plot(ps..., layout=(6,3), size=(800,1100))

Plots.scalefontsizes()
p1=fluxbias_SNR_plot(res)
p2=fluxstd_SNR_plot(res)
myplot = plot([p1,p2]..., layout=(1,2), size=(800,500))



savefig(myplot,"fluxbias_undersample.png")


objxy = [[14.0,14.0], [14.0,14.5], [14.0,15.0]]



covfaclist = covfactor_list([[14,14]], efacs, Ndim=Ndim)
V          = reshape(covfaclist,Ndim,size(covfaclist,2))


# Oct 24 version for Tom Prince
SNRs = [5,10,20]
FWHMs = [2.5]
res, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=10000, objtype="Gaussian", objtype_prior="Gaussian")

rep1=fluxbias_SNR_plot(res)
p2=fluxstd_SNR_plot(res)
myplot = plot([p1,p2]..., layout=(1,2), size=(800,500))
savefig(myplot,"fluxhistograms.png")
# Now do the same but with position

function dxplot(res, SNRind=3)

	SNR = res.SNR[SNRind]
	# x offset of mock data
	rddx = res.ddx[SNRind]
	# recovered x offset
	rdx = res.dx[SNRind]
	Plots.scalefontsizes()
	Plots.scalefontsizes(2)
	# scatter plot with transparency = 0.25
	p1 = scatter(rddx, rdx[:,1],legend=false, m=(0.5,0.05), 
		xlabel="True offset [pixels]", ylabel="recovered offset [pixels]", title="SNR = $SNR")

	xdiff = rdx[:,1] - (rddx.*res.dofwt[SNRind])
	plot!(p1, [-1,1],[-1,1].*res.dofwt[SNRind],linecolor=:gray,linewidth=3,label=false)
	xerr = std(xdiff)
	# make a string rounding xerr to 3 digits
	xerrstr = @sprintf("%.3f",xerr)
	
	p2 = scatter(rddx, xdiff,legend=false, m=(0.5,0.05), 
		xlabel="True offset [pixels]", ylabel="recovered offset - true offset [pixels]", title="RMS = $xerrstr")
	#plot!(p2, [-1,1],[-1,1].*(res.dofwt[SNRind] -1),linecolor=:gray,linewidth=3,label=false)

	p  = plot([p1,p2]..., layout=(1,2), size=(1600,850), margin=10Plots.mm)
	return p
end


# makes plots for paper
p=dxplot(res,1)
savefig(p,"dxplot5.png")
p=dxplot(res,2)
savefig(p,"dxplot10.png")
p=dxplot(res,3)
savefig(p,"dxplot20.png")


SNRs = [5,10,20]
FWHMs = [2.5]
res, ps = runsims(SNRs, FWHMs, Np=Np, Nvec=8, frac_σ_cen=1/8, Ntrial=10000, objtype="Moffat", objtype_prior="Gaussian")



using StructArrays
struct Foof{T}
           μ_fwhm::T
		   σ_fwhm::T
		   μ_cen::T
		   σ_cen::T
		   μ_β::T
		   σ_β::T
		   Np::Int64
		   Nsam_cov::Int64
		   Nsam::Int64
		   SNR::T
       end


fm = [res[i,:].fmean[1] for i=1:9]





# Simple trial
Np = 27
fwhm=4.0
foo = Gaussian_model([14.0,14.5]; Np=Np,fwhm=fwhm)
dat = reshape(foo,Np*Np)
foo = Gaussian_model([14.0,14.0]; Np=Np,fwhm=fwhm)
dat0 = reshape(foo,Np*Np)
foo = Gaussian_model([14.0,14.0]; Np=Np,fwhm=sqrt(fwhm^2 + (2.355*0.25)^2))
datfat = reshape(foo,Np*Np)

σ_fwhm = 0.25
efacs = psf_efacs(Nvec; Np=Np, Nsam=20000, σ_cen=fwhm/12, fwhm=fwhm, σ_fwhm=σ_fwhm)
covfaclist = covfactor_list([[Nphalf,Nphalf]], efacs, Ndim=Ndim)
VV = covfaclist[:,:,1]
II=inv(VV*VV' + B .* 1)

recon = VV*VV'*II*dat
recon0 = VV*VV'*II*dat0


# when α_cen is small, eval1 is a bit bigger.  Also, FWHM of evec1 is right.
# when σ_cen is large, eval1 is smaller, FWHM of evec1 is quadrature sum of σ_cen and fwhm.
# this means that overlap integral is smaller than it would be.
# but it allows more noise chasing?


# Implement square root algorithm
# input x
# output y
# y=sqrt(x)

# return indices of listA that match elements of listB
"""
    match_list(listA, listB)
	return indices of listA that match elements of listB
TBW
"""
function match_list(listA, listB)
	w = findall(x->x in listB, listA)
	return w
end


# create a dataframe to hold a, b, and c
function make_df()
	df = DataFrame(a=Float64[], b=Float64[], c=Float64[])
	return df
end


# looping over each column of efacs, display efacs as Np x Np images using heatmap
function plotmyefacs(efacs)
	Np = Int(sqrt(size(efacs,1)))
	for i=1:size(efacs,2)
		heatmap(reshape(efacs[:,i],Np,Np))
		sleep(0.5)
	end
end


function plot_efacs(efacs)
	Np = Int(sqrt(size(efacs,1)))
	heatmap(reshape(efacs,Np,Np))
end



# return product of the inverse of matrix A times a column vector x 
function invA_times_x(A,x)
	return A\x
end

# return product of the inverse of matrix (A + V*V') times a column vector x 
function invAplusVVT_times_x(A,V,x)
	return (A + V*V')\x
end

# return product of the inverse of matrix (A + V*V') times a column vector x, where V is low rank, using Sherman-Morrison formula




# read HDU 5 of the FITS file named fname, and return the result as a dataframe
function read_HDU5(fname)
	# read HDU 5 of the FITS file named fname
	hdu = FITS(fname)[5]
	# convert the HDU to a dataframe
	df = DataFrame(hdu)
	return df
end

# make a dataframe that holds integer x, string foo, and float yaxis
function make_df()
	df = DataFrame(x=Int64[], foo=String[], yaxis=Float64[])
	return df
end

# import packages needed for all the code in this file
using FITSIO
using DataFrames
using Plots
using Statistics
using LinearAlgebra
using Random
using Distributions
using DelimitedFiles
using StructArrays
using SparseArrays
using FFTW
using FFTViews
using FFTW


# use the Google Calendar API to add an event to my Calendar
# import the Google Calendar API
using GoogleCalendar
function add_event()
	# create a Google Calendar object
	cal = GoogleCalendar.Calendar("mycal")
	# create an event
	event = GoogleCalendar.Event("myevent", "2020-01-01T12:00:00", "2020-01-01T13:00:00")
	# add the event to the calendar
	add_event!(cal, event)
end


# benchmark execution time of BLISBLAS versus normal Julia multiplication
using BenchmarkTools
using LinearAlgebra
using BLIS

function benchmark_BLISBLAS()
	# create a 1000 x 1000 matrix
	A = rand(1000,1000)
	# create a 1000 x 1000 matrix
	B = rand(1000,1000)
	C = zeros(1000,1000)
	# benchmark BLISBLAS
	@btime BLIS.gemm!('N', 'N', 1.0, $A, $B, 0.0, $C)
	# benchmark normal Julia multiplication
	@btime $A * $B
end

# Load Apple accelerate wrapper package
using Accelerate

using Metal
# Demonstrate matrix multiplication with Metal 
function benchmark_Metal()
	# create a 1000 x 1000 matrix
	A = MtlArray(rand(Float32,1000,1000))
	# create a 1000 x 1000 matrix
	B = MtlArray(rand(Float32,1000,1000))
	C = MtlArray(zeros(Float32,1000,1000))
	# benchmark Metal
	@btime mul!(C, A, B)
end

using BenchmarkTools
using AppleAccelerate
using LinearAlgebra
include("AppleAccelerateLinAlgWrapper.jl")

@benchmark AccelerateLinAlgWrapper.gemm($A,$B)
A = rand(1000000)

@benchmark foo = AppleAccelerate.exp(A)
@benchmark foo = exp.(A)
@benchmark foo = AppleAccelerate.log!(A,A)

A = rand(Float32,1000, 1000)
B = rand(Float32,1000, 1000)
C = zeros(Float32,1000, 1000)
@benchmark C=A*B
@benchmark AppleAccelerate.mul!(C,A,B)

@benchmark C = 2.0 .^ A
@benchmark AppleAccelerate.exp2(A)
@benchmark exp2.(A)
@benchmark sin.(A)  # 5.4ns
@benchmark cos.(A)  # 5.7ns
@benchmark sincos.(A)  # 8.1ns

@benchmark AppleAccelerate.sin(A)  # 1.7ns
@benchmark AppleAccelerate.cos(A)  # 1.65ns
@benchmark AppleAccelerate.sincos(A)  # 2.4ns

# set the number of BLAS threads to 1 
using LinearAlgebra
BLAS.set_num_threads(1)

# benchmark AppleAcceleratedLinAlgWrapper.gemm versus normal Julia multiplication for type T 
function benchmark_gemm(T)
	# create a 1000 x 1000 matrix
	A = rand(T,1000,1000)
	# create a 1000 x 1000 matrix
	B = rand(T,1000,1000)
	C = zeros(T,1000,1000)
	# benchmark AppleAcceleratedLinAlgWrapper.gemm
	print(@benchmark AppleAccelerateLinAlgWrapper.gemm($A, $B))
	# benchmark normal Julia multiplication
	print(@benchmark $A * $B)
end



# add an event to my Google calendar using the web API 
# import the Google Calendar API
using GoogleCalendar
function add_event()
	# create a Google Calendar object
	cal = GoogleCalendar.Calendar("mycal")
	# create an event
	event = GoogleCalendar.Event("myevent", "2020-01-01T12:00:00", "2020-01-01T13:00:00")
	# add the event to the calendar
	add_event!(cal, event)
end

# looping over columns of efacs, display an Np x Np image using heatmap 
function plotmyefacs(efacs)
	Np = Int(sqrt(size(efacs,1)))
	for i=1:size(efacs,2)
		heatmap(reshape(efacs[:,i],Np,Np))
		sleep(0.5)
	end
end

# load the apple accelerate package

# put elements of vector V on the diagonal of a matrix



using Statistics
"""
    rebin(img, newDims; trim=false, sample=false)
	Resize an image to new dimensions

	Parameters
	----------
	img : Array
		Image to be resized
	newDims : Tuple
		New dimensions of image
	trim : Bool
		If true, trim image to nearest integer multiple of factor
	sample : Bool
		If true, sample image instead of averaging (only for down-sizing)
		
	Returns
	-------
	newImg : Array
		Resized image

	Notes
	-----
	Inspired by IDL rebin()
	Here "x" means row and "y" means column, so (x,y)
	Author: Douglas Finkbeiner, Harvard University
	  with help from GPT-4
TBW
"""
function rebin(img, newDims; trim=false, sample=false)
    sizeX, sizeY = size(img)
    newSizeX, newSizeY = newDims

    # New factors for resizing
    factorX = newSizeX // sizeX
    factorY = newSizeY // sizeY

    # If trim is true, adjust size to nearest integer multiple of factor
    if trim
        sizeX = floor(Int, sizeX * factorX) ÷ round(Int, factorX)
        sizeY = floor(Int, sizeY * factorY) ÷ round(Int, factorY)
    end

    # Resize in X dimension
    newImgX = zeros(newSizeX, sizeY)
    if factorX > 1
        # Up-sizing in X dimension
        for x in 1:sizeX
            newImgX[((x-1)*round(Int, factorX)+1):x*round(Int, factorX), :] .= img[x, :]'
        end
    else
        # Down-sizing in X dimension
        for x in 1:newSizeX
            if sample
                newImgX[x, :] = img[round(Int, (x-1)÷factorX)+1, :]
            else
                newImgX[x, :] = mean(img[(Int((x-1)/factorX+1)):Int(x/factorX), :],dims=1)
            end
        end
    end

    # Resize in Y dimension
    newImg = zeros(newSizeX, newSizeY)
    if factorY > 1
        # Up-sizing in Y dimension
        for y in 1:sizeY
            newImg[:, ((y-1)*round(Int, factorY)+1):y*round(Int, factorY)] .= newImgX[:, y]
        end
    else
        # Down-sizing in Y dimension
        for y in 1:newSizeY
            if sample
                newImg[:, y] = newImgX[:, round(Int, (y-1)÷factorY)+1]
            else
                newImg[:, y] = mean(newImgX[:, Int((y-1)/factorY+1):Int(y/factorY)],dims=2)
            end
        end
    end

    return newImg
end


img = rand(3,4)
imgbig = rebin(img, (6,8))
imgsmall = rebin(imgbig, (3,4))

img = rand(1000,1000)
imgbig = rebin(img, (2000,2000))
imgsmall = rebin(imgbig, (1000,1000))



# test speed of component separation
function comp_speed_test(efacs; σ_pix=0.003, σ_back=0.01, fwhm=fwhm, seed=nothing)

	Np = round(Int,sqrt(size(efacs,1)))÷2+1  # assume square
	Ndim = Np*Np
	Nphalf = Np÷2+1

	B = Diagonal(ones(Ndim).*(σ_pix)^2)
	Binv = inv(B)
	covfaclist = covfactor_list([[Nphalf,Nphalf]], efacs, Ndim=Ndim)
	V          = reshape(covfaclist,Ndim,size(covfaclist,2))
	
	Ntrial = 1000
	for i=1:Ntrial
		fcomp = onestar(V, B, Binv, [[Nphalf,Nphalf]], [[Nphalf,Nphalf]], Np=Np, σ_pix=σ_pix, fwhm=fwhm, seed=seed)
	end
end

Nvals = [33, 65, 97, 129, 193, 257]
btimes = []
for Np in Nvals
	Nvec = 20
	efacs = psf_efacs(Nvec; Np=Np, Nsam=20000, σ_cen=fwhm/12, fwhm=fwhm, σ_fwhm=σ_fwhm, objtype="Moffat")
	b = @benchmark comp_speed_test(efacs, σ_pix=0.003, σ_back=0.01, fwhm=2.0, seed=nothing)
	push!(btimes, median(b.times) / 1e9)
	println(btimes)
end







function precision_x_plot(efacs; σ_pix=0.003, σ_back=0.01, fwhm=fwhm, seed=nothing)
	Np = round(Int,sqrt(size(efacs,1)))÷2+1  # assume square
	Ndim = Np*Np
	Nphalf = Np÷2+1

	# variance of dx, dy efacs
	var_dx = mean(sum(efacs[:,2:3].^2,dims=1))

	# background covariance
	#B    = diagm(ones(Ndim).*σ_pix^2) .+ σ_back^2
	fudge = 1
#	B    = diagm(ones(Ndim).*(fudge*σ_pix)^2) .+ σ_back^2
#	Binv = inv(cholesky(B))
    B    = Diagonal(ones(Ndim).*(fudge*σ_pix)^2)
	Binv = inv(B)
	Δx   = collect(-2:0.0025:2)
	Ntrial = length(Δx)
	κ    = [1.0]
	labels = ["κ = 0.1", "κ = 1.0","κ = 10"]
	Nκ   = length(κ)
	flux = zeros(Ntrial, Nκ)
	dx   = zeros(Ntrial, Nκ)
	for j=1:Nκ
		for i=1:Ntrial
			fcomp = onestar(efacs.*κ[j], B, Binv, [[Nphalf+Δx[i],Nphalf]], [[Nphalf,Nphalf]], Np=Np, σ_pix=σ_pix, fwhm=fwhm, seed=seed)
			fpar  = comp_params(fcomp)
			flux[i,j] = fpar[1].flux
			dx[i,j]   = fpar[1].x - Nphalf
		end
	end
	mcolor = [:red2,:green3,:blue2]
	var_dx = mean(sum(efacs[:,2:3].^2,dims=1))
	fac = (fudge*σ_pix)^2/(var_dx + (fudge*σ_pix)^2)
	p = plot([-1.5,1.5],[-1.5,1.5].*(-fac),linecolor=:blue,linewidth=1,label=false,legend=:topleft,
			xlabel="True Δx [pix]", ylabel="Recovered x error [pix]")

	for j=Nκ:-1:1
		#scatter!(p, Δx, dx[:,j], m=(mcolor[j],2,0.65),label=labels[j])
		xerr = dx[:,j]-Δx
		scatter!(p, Δx, xerr, m=(mcolor[j],2,0.65),label=nothing)
		wh = findall(abs.(Δx) .< 0.5)
		sigerr = std(xerr[wh])
		println(sigerr)
	end
	# p = plot([-2,2],[1.0,1.0],linecolor=:blue,linewidth=4,label=false,
	# 	xlabel="Δx [pixels]",ylabel="flux (recovered/true)")
	# for j=Nκ:-1:1
	# 	scatter!(p, Δx, flux[:,j], m=(mcolor[j],5,0.65),ylims=(.6,1.15),label=labels[j])
	# end

	return p
end




# read opECalib file and return gain for both amps of an SDSS chip
# in the case of a 1-amp chip, the two numbers should be equal.
function sdss_read_gain(filter="r", camcol=1)
	# open opECalib file
	f = FITS("opECalib-51773.fits")

	# read HDU 2 of the FITS file
	df = DataFrame(f[2])

	# mapping from band to camrow
	camrow = Dict("r"=>1, "i"=>2, "u"=>3, "z"=>4, "g"=>5)

	# select only the rows with camcol 1 and camrow 2
	chip = df[(df.CAMCOL .== camcol) .& (df.CAMROW .== camrow[filter]),:]

	# return the gain for both amps
	return [(chip.GAIN0)[1], (chip.GAIN1)[1]]
end

sdss_read_gain("r",1)




"""
    gen_cadence(;Nobs=30, fillfrac=0.5, Nday = 20)
	generate a cadence of observations

	Parameters
	----------
	Nobs : Int
		Number of observations
	Nday : Int
		Number of days to observe
	fillfrac : Float
		Fraction of time each day that is observed

	Returns
	-------
	t : Array
		Array of times of observations
"""
function gen_cadence(;Nobs=30, Nday = 20, fillfrac=0.5)
	# generate a cadence of observations
	t_raw = rand(Nobs).*Nday

	# restrict non-integer part of t_raw to fillfrac
	t_int = floor.(Int, t_raw)  # integer part of t_raw
	t_frac = t_raw .- t_int     # fractional part of t_raw
	t = t_int .+ t_frac .* fillfrac

	return t
end

# generate noise with standard deviation σ for each time t
# do this for Nsam samples
function gen_noise(t; σ=0.1, Nsam=1)
	Nobs = length(t)
	noise = randn(Nobs,Nsam).*σ
	return noise
end


#generate mock data with period P and phase phi (0..1)
function gen_mock(t, P, phi)
	return sin.(2π.*(t./P .+ phi))
end


function gen_mock!(Vf, t, P)
	for i=1:length(t)
		Vf[i,:] .= sincos(2π*(t[i]/P))
	end
	return 
end



Ptrue=2.31
σ=0.5
Nobs=150
Nday=80
doplot=true
# generate a cadence of observations  (~1 us)
t = gen_cadence(Nobs=Nobs, Nday=Nday)

# noise covariance, assumed diagonal  (~2 us)
Cn = Diagonal(ones(length(t)).*σ^2)

# generate mock data   (~5 us)
data = gen_mock(t, Ptrue, 0.0) 
if doplot p1 = scatter(t, data, title="True", xlabel="time", ylabel="flux", legend=false) end

# add noisecov
data += randn(length(t)).*σ
if doplot p2=scatter(t,data, title="Mock" , xlabel="time" ,ylabel="flux", legend=false) end
#scatter(t,data.+noise[:,1])

function PrinceFinkbeiner_dchisq(data, t, periods = 1.0:0.01:3.0)

	# initialize Δχ2 array ( << 1 us)
	nperiod = length(periods)
	Δχ2   = zeros(nperiod)
	Vf    = zeros(length(t),2)

	# loop over period hypotheses
	for i = 1:nperiod
		# "Eigenvectors" are sin() and cos()
		gen_mock!(Vf, t, periods[i])  # 3.5 us
		# Cf = (Vf*Vf')                  # 190 us
		# Δχ2[i] = data'*((Cf+Cn)\data)  # 1000 us
		
		# Use Sherman-Morrison, it agrees with the above at 1e-13 level
		CninvVf = Cn\Vf   # 1 us
		dataCninvVf = data'*CninvVf   # 0.3 us
		Δχ2[i] = dataCninvVf * ((I+Vf'*CninvVf)\dataCninvVf')  # 1.6 us  
	end
	return Δχ2
end

periods = 1.0:0.0025:3.0
dchisq = PrinceFinkbeiner_dchisq(data, t, periods)

p3 = scatter(periods, dchisq, title="Period finding", xlabel="period", ylabel="Δχ2")
p4 = scatter(mod.(t./Ptrue,1),data,title="phase wrapped",xlabel="phase",ylabel="flux")

myplot = plot([p1,p2,p3,p4]..., layout=(2,2), size=(800,500))

#scatter(periods,amplitudes,xlims=(2.2,2.4),ylims=(0,10))


#define blackbody function
function blackbody(wl, T)
	# Planck constant
	h = 6.62607015e-34
	# speed of light
	c = 299792458
	# Boltzmann constant
	k = 1.380649e-23
	# convert wavelength to frequency, ν
	ν = c ./ (wl*1e-6)
	# blackbody spectrum
	Inu = (2 * h) .* ν.^3 ./ c.^2 ./ expm1.(h .* ν ./ (k * T))
	return Inu
end



# plot blackbody spectrum in I_ν units for temperature T
function plot_blackbody(T)
	# wavelengths in microns
	λ = 0.25:0.01:10
	# blackbody spectrum, scaled so peak is order 1
	Inu = blackbody(λ, T) .* 1e8
	# plot
	p=plot(λ, Inu, xlabel="wavelength [μm]", ylabel="I_ν [arbitrary units]", title="Blackbody spectrum", axis=:log,ylims=(0.1,10),label="y",line=(2))
	plot!(p, λ, Inu.+1, label="1+y", line=(2,:dash))
	plot!(p, λ, log.(Inu.+1), label="log(1+y)", line=(2))
	return p
end
p = plot_blackbody(5000)
savefig(p,"blackbody.pdf")
using Plots

#using Makie
#using CairoMakie
# allocate local array of 16 zeros


using LinearAlgebra
using Statistics
using FITSIO
using StructArrays
using LaTeXStrings
using Printf
using KrylovKit
using Random


"""
    img = Moffat_model(cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
Return circularly symmetric Moffat model evaluated on a grid

# Arguments:
- `cen`:      center (x,y)  x,y ∈ 1..`Np`

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `fwhm`:     FWHM of PSF [pixels]
- `β`:        Moffat `β` parameter

# Output:
- `img`:      Moffat model

# Comments:
2022-Feb-06 - Written by Douglas Finkbeiner, CfA 
"""
function Moffat_model(cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
	img = zeros(Np,Np)
	Moffat_model!(img, cen; Np, fwhm=fwhm, β=β)
    return img
end


"""
    Moffat_model!(img, cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
Return circularly symmetric Moffat model evaluated on a grid

# Arguments:
- `img`:      image to fill
- `cen`:      center (x,y)  x,y ∈ 1..`Np`

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `fwhm`:     FWHM of PSF [pixels]
- `β`:        Moffat `β` parameter

# Output:
- `img`:      input image modified

# Comments:
Same as Moffat_model() but avoids mallocs and is 10% faster \
2024-Feb-08 - Written by Douglas Finkbeiner, CfA \
"""
function Moffat_model!(img, cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
	α   = fwhm/(2.0*sqrt(2.0^(1.0/β)-1.0))
	α2  = α*α
	peakval = (β-1.0)/(π*α2)
	for i=1:Np
		Δi2 = (i-(cen[2]))^2 / α2 + 1.0
		for j=1:Np
			Δj2 = (j-(cen[1]))^2 / α2
			img[i,j] = peakval .* (Δi2+Δj2).^(-β)
		end
	end
    return img
end



"""
    Moffat_model!(img, cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
Return circularly symmetric Moffat model evaluated on a grid

# Arguments:
- `img`:      image to fill
- `cen`:      center (x,y)  x,y ∈ 1..`Np`

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `fwhm`:     FWHM of PSF [pixels]
- `β`:        Moffat `β` parameter

# Output:
- `img`:      input image modified

# Comments:
Same as Moffat_model() but avoids mallocs and is 10% faster \
2024-Feb-08 - Written by Douglas Finkbeiner, CfA \
"""
function Moffat_model_add!(img, cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
	α   = fwhm/(2.0*sqrt(2.0^(1.0/β)-1.0))
	α2  = α*α
	peakval = (β-1.0)/(π*α2)
	for i=1:Np
		Δi2 = (i-(cen[2]))^2 / α2 + 1.0
		for j=1:Np
			Δj2 = (j-(cen[1]))^2 / α2
			img[i,j] += peakval * (Δi2+Δj2).^(-β)
		end
	end
    return img
end





"""
    Moffat_model_samples(Np::Int, Npsf::Int; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.05, μ_β=4.8, σ_β=0.1, Δx=0.0, Δy=0.0, smear=true)

Return circularly symmetric Moffat model evaluated on a grid

# Arguments:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `Npsf`:     number of PSF samples to compute

# Keywords:
- `σ_cen`:    RMS position variation (about the center)
- `μ_fwhm`:   mean FWHM [pixels]
- `σ_fwhm`:   RMS FWHM [pixels]
- `μ_β`:      mean Moffat β parameter
- `σ_β`:      RMS Moffat β parameter
- `Δx`:       shift in x
- `Δy`:       shift in y
- `smear`:    CHECK THIS !!!!   compute Moffat on 2x finer pixel scale and rebin

# Output:
- `psfs`:      Moffat model samples

# Comments:
- Sums to 1 if centered far from edge. \
- Atmospheric turbulence theory predicts β=4.8 
2022-Feb-06 - Written by Douglas Finkbeiner, CfA \
2024-Feb-08 - Modified to use Moffat_model!() for 20% speed increase \
                About 14 ms per 1000 PSFs on M1 core \
"""
function Moffat_model_samples(Np::Int, Npsf::Int; σ_cen=0.25, μ_fwhm=2.5, σ_fwhm=0.05, μ_β=4.8, σ_β=0.1, Δx=0.0, Δy=0.0, uniform=false, smear=false)
	μ_cen = (Np+1)÷2 .+ [Δx,Δy]
	cen   = uniform ? μ_cen .+ (rand(2,Npsf) .- 0.5) : μ_cen .+ randn(2,Npsf) .* σ_cen
	fwhm  = μ_fwhm .+ randn(Npsf) .* σ_fwhm
	β     = μ_β    .+ randn(Npsf) .* σ_β
	psfs  = Array{Float64, 3}(undef,Np,Np,Npsf)
	for i=1:Npsf
		if smear
			throw("not implemented")
			psfs[:,:,i] = 4.0 .* imresize(Moffat_model(cen[:,i].*2 .-0.5, Np=Np*2, fwhm=fwhm[i].*2, β=β[i]),(Np,Np))
		else
			thispsf = @view psfs[:,:,i]
			Moffat_model!(thispsf, cen[:,i], Np=Np, fwhm=fwhm[i], β=β[i])
		end
	end
	return psfs
end


"""
	img = Gaussian_model(cen; Np=33, fwhm=3.0)
Return circularly symmetric Gaussian model evaluated on a grid

# Arguments:
- `cen`:      center (x,y)  x,y ∈ 1..`Np`

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `fwhm`:     FWHM of PSF [pixels]

# Output:
- `img`:      Moffat model

# Comments:
2022-Feb-06 - Written by Douglas Finkbeiner, CfA
2023-Jun-02 - modified to work with any Real type (works with ForwardDiff) \\
"""
function Gaussian_model(cen::Vector{T}; Np::Int=33, fwhm=T(3.0)) where {T <: Real}
    img = zeros(T,Np,Np)
	σ = fwhm/2.355
	norm = 1.0/(2π*σ*σ)
	rtox = 1.0/(2*σ*σ)	
    for i=1:Np
        for j=1:Np
            r2 = (i-(cen[2]))^2+(j-(cen[1]))^2
			x = -r2*rtox
            img[i,j] = x < T(-120) ? T(0.0) : norm*exp(x)
        end
    end
    return img
end


"""
    Gaussian_model!(img, cen::Vector; Np::Int=33, fwhm=3.0)
Return circularly symmetric Gaussian model evaluated on a grid

# Arguments:
- `img`:      image to fill
- `cen`:      center (x,y)  x,y ∈ 1..`Np`

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `fwhm`:     FWHM of PSF [pixels]
2024-Feb-08 - Written by Douglas Finkbeiner, CfA \\
              33% faster than Gaussian_model() \\
			  but does not work with ForwardDiff \\
"""
function Gaussian_model!(img, cen::Vector; Np::Int=33, fwhm=3.0)
	σ = fwhm/2.355
	norm = 1.0/(2π*σ*σ)
	rtox = 1.0/(2*σ*σ)	
	for i=1:Np
		for j=1:Np
			r2 = (i-(cen[2]))^2+(j-(cen[1]))^2
			x = -r2*rtox
			img[i,j] = x < -120.0 ? 0.0 : norm*exp(x)
		end
	end
	return img
end


"""
    psfs = Gaussian_model_samples(Np=33, Npsf=1000; μ_fwhm=2.5, σ_fwhm=0.05, σ_cen=0.25, Δx=0.0, Δy=0.0, smear=true)
Return circularly symmetric Gaussian model evaluated on a grid

# Arguments:
- `Np`:       size of image grid (`Np`,`Np`) [pixels]
- `Npsf`:     number of PSF samples to compute

# Keywords:
- `σ_cen`:    RMS position variation (about the center)
- `μ_fwhm`:   mean FWHM [pixels]
- `σ_fwhm`:   RMS FWHM [pixels]
- `Δx`:       shift in x
- `Δy`:       shift in y
- `smear`:    compute Gaussian on 2x finer pixel scale and rebin

# Output:
- `psfs`:      Gaussian model samples

# Comments:
- Sums to 1 if centered far from edge. \
2022-Feb-06 - Written by Douglas Finkbeiner, CfA \
2024-Feb-08 - Modified to use Gaussian_model!() for 33% speedup \
				About 5 ms for 1000 PSFs on M1 core
"""
function Gaussian_model_samples(Np=33, Npsf=1000; μ_fwhm=2.5, σ_fwhm=0.05, σ_cen=0.25, Δx=0.0, Δy=0.0, uniform=false, smear=true)
	μ_cen = (Np+1)÷2 .+ [Δx,Δy]
	cen = uniform ? μ_cen .+ (rand(2,Npsf) .- 0.5) : μ_cen .+ randn(2,Npsf) .* σ_cen

	fwhm  = μ_fwhm .+ randn(Npsf) .* σ_fwhm
	psfs  = zeros(Np,Np,Npsf)
	for i=1:Npsf

		if smear
			throw("not implemented!")
			psfs[:,:,i] = 4.0 .* imresize(Gaussian_model(cen[:,i].*2, Np=Np*2, fwhm=fwhm[i].*2),(Np,Np))
		else
			thispsf = @view psfs[:,:,i]
			Gaussian_model!(thispsf, cen[:,i], Np=Np, fwhm=fwhm[i])
		end

	end

	return psfs
end


# slow version for debugging
function Sersicslow(A, x0, n, Cinv; Np=33, xlim=(-4,4), ylim=(-4,4))
	# see https://academic.oup.com/mnras/article/441/3/2528/1108396
	Nx  = Np-1
	Ny  = Np-1
	Δx  = (xlim[2]-xlim[1])/Nx
	Δy  = (ylim[2]-ylim[1])/Ny
	img = zeros(Nx+1,Ny+1)
	xvals = collect(xlim[1]:Δx:xlim[2])
	yvals = collect(ylim[1]:Δy:ylim[2])
	η = 0.5/n
	k = 1.9992*n - 0.3271
	R = zeros(2)
	#Σinv = inv(Σ)
	for j = 1:Nx+1
		R[1] = xvals[j] - x0[1]
		for i = 1:Ny+1
			R[2] = yvals[i] - x0[2]
			#R = [xvals[j],yvals[i]] - x0   # this way is slower
			dd = (R'*Cinv*R)
			img[i,j] = exp(-k* dd^η)
		end
	end
	img .*= (A/sum(img))
	return img
end


"""
	img = Sersic_model(cen, n, Cinv; Np=33, xlim=(-4,4), ylim=(-4,4))
Return Sérsic model evaluated on a grid

# Arguments:
- `A`:        amplitude (image sums to this)
- `cen`:      center (y,x)
- `n`:        Sérsic index
- `Cinv`:     inverse covariance matrix (2,2)

# Keywords:
- `Np`:       size of image grid (`Np`,`Np`)
- `xlim:`     boundaries of image in arcsec
- `ylim:`     boundaries of image in arcsec

# Output:
- `img`:      Sérsic model

# Comments:
- Uses approximation for k \\
- Sums to 1 if centered far from edge. \\
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
function Sersic_model(cen, n, Cinv; Np=33, xlim=(-4,4), ylim=(-4,4))
	# see https://academic.oup.com/mnras/article/441/3/2528/1108396
	Nx  = Np-1
	Ny  = Np-1
	Δx  = (xlim[2]-xlim[1])/Nx
	Δy  = (ylim[2]-ylim[1])/Ny
	img = zeros(Nx+1,Ny+1)
	xvals = collect(xlim[1]:Δx:xlim[2])
	yvals = collect(ylim[1]:Δy:ylim[2])
	η = 0.5/n
	k = 1.9992*n - 0.3271
	R = zeros(2,Np)
	#Σinv = inv(Σ)
	for j = 1:Nx+1
		R[1,:] .= xvals[j] - cen[1]
		R[2,:] = yvals .- cen[2]
		dd = sum(Cinv*R .* R, dims=1)'
		img[:,j] = exp.(-k .* dd.^η)
	end
	img .*= (1.0/sum(img))
	return img
end


"""
	img = mock_galaxy(cen; Np=33, xlim=nothing, ylim=nothing)
Generate a mock galaxy with Sersic_model().

# Arguments:
- `cen`:      center (y,x)

# Keywords:
- `Np`:       size of image (`Np`,`Np`)
- `xlim:`     boundaries of image in arcsec
- `ylim:`     boundaries of image in arcsec

# Output:
- `img`:      mock galaxy image

# Comments:
- Generates Sérsic model at cen with random flux 2-10, n=1-2,
  random angle. \\
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
function mock_galaxy(cen; Np=33, xlim=nothing, ylim=nothing)
	flux = rand()*8 + 2
	n    = rand() + 1   # 1 to 2
	θ    = rand()*π
	rot  = reshape([cos(θ), sin(θ), -sin(θ), cos(θ)],2,2)
	cov  = 16 .*(rot' * diagm([1.0,rand()*4+0.5]) * rot)
	img  = flux .* Sersic_model(cen, n, inv(cov), Np=Np, xlim=xlim, ylim=ylim)
	return img
end


"""
	cov = big_psf_covariance(Np=, Nsam=, fwhm=, type=, smear=, meansub=)
Generate a big covariance matrix for Gaussian or Moffat PSF.

# Keywords:
- `Np`:       size of image (`Np`,`Np`)
- `Nsam:`     number of samples
- `fwhm:`     fwhm of PSF
- `type:`     Gaussian or Moffat
- `smear:`    sub-pixel smearing of PSF (NOT TESTED)
- `meansub:`  compute zero-centered covariance

# Output:
- `cov`:      covariance matrix evaluted out to Np2=Np*2-1

# Comments:
- One can take a subarray of the output big covariance matrix
  (or its evecs) to effectively shift the psf to any location. \\
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
# about 3 sec per 10000 for M or G (10 sec on 1 M1 core)
function big_psf_covariance_deprecated(;Np=33, Nsam=10000, fwhm=nothing, type="Moffat", smear=false, meansub=false)

	# padded grid size
	Np2 = Np*2-1

	# generate training data
	if type=="Gaussian"
		μ_fwhm = isnothing(fwhm) ? 2.2 : fwhm
		psfs = Gaussian_model_samples(Np2, Nsam, μ_fwhm=μ_fwhm, σ_fwhm=0.5, σ_cen=0.25, smear=smear)
		vecs = reshape(psfs, Np2*Np2, Nsam)
	end

	if type=="Moffat"
		μ_fwhm = isnothing(fwhm) ? 2.5 : fwhm
		#psfs = Moffat_model_samples(Np2, Nsam, μ_fwhm=μ_fwhm, μ_β=4.8, σ_fwhm=0.05, σ_cen=0.15, smear=smear)
		#psfs = Moffat_model_samples(Np2, Nsam, μ_fwhm=μ_fwhm, μ_β=3.5, σ_fwhm=0.05, σ_cen=0.4, smear=smear)
		psfs = Moffat_model_samples(Np2, Nsam, μ_fwhm=μ_fwhm, μ_β=3.5, σ_fwhm=0.01, σ_cen=0.4, smear=smear)
		vecs = reshape(psfs, Np2*Np2, Nsam)
	end

	# Deprecate Sersic for now -- should use a specialized galaxy function
	# if type=="Sersic"
	# 	μ_fwhm = isnothing(fwhm) ? 2.5 : fwhm
	# 	gals = Sersic_model_samples(Np2, Nsam, μ_fwhm=μ_fwhm, σ_fwhm=0.05, σ_cen=0.15)
	# 	vecs = reshape(gals, Np2*Np2, Npsf)
	# end

	if meansub
		vecs = vecs .- mean(vecs, dims=1)
	end
	bigcov = (vecs*vecs') ./ Nsam
	return bigcov
end


"""
	cov = galaxy_covariance(Np=33, Ngal=50000, σ_pix=0.001; xlim=(-4,4), ylim=(-4,4))
Generate covariance using samples from mock_galaxy().

# Arguments:
- `Np`:       size of image (`Np`,`Np`)
- `Ngal`:     number of galaxies used to generate covariance
- `σ_pix`:    per pixel noise

# Keywords:
- `xlim:`     boundaries of image in arcsec
- `ylim:`     boundaries of image in arcsec

# Output:``
- `img`:      mock galaxy image

# Comments:
- Generates covariance using samples from `mock_galaxy()`. \\
  Galaxy center is jittered by `σ_cen.` \\
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
function galaxy_covariance(Np=33, Ngal=50000, σ_pix=0.001; xlim=(-4,4), ylim=(-4,4))
	σ_cen = 0.125

	# build the covariance matrix in batchs of 1000
	Nbatch = round(Int,ceil(Ngal/1000))
	cov = zeros(Np*Np, Np*Np)
	for ibatch = 1:Nbatch
		Ng = min(Ngal-(ibatch-1)*1000, 1000)
		println("Batch ",ibatch,"  ",Ng)
		cub = zeros(Np, Np, Ng)
		for i=1:Ng
			cen = randn(2) .* σ_cen
			cub[:,:,i] = mock_galaxy(cen, Np=Np, xlim=xlim, ylim=ylim)
		end
		ns = randn((Np*Np, Ng)).*σ_pix
		dat = ns+reshape(cub, (Np*Np, Ng))
		cov0 = dat*dat' ./Ngal
		cov .+= cov0
	end
	return cov  # return Symmetric(cov) ??
end


# 124 ms on Mac, 233 ms for 1000
function gauss_deblend(x_d, Σ_a, Σ_b)
	Σ_ab_inv = inv(cholesky(Σ_a + Σ_b))
	Σ_ad = Σ_a*Σ_ab_inv*Σ_b  # 80 ms

	Σinvx = (Σ_ab_inv*x_d)
	μ_ad = Σ_a*Σinvx
	μ_bd = Σ_b*Σinvx
	return μ_ad, μ_bd, Σ_ad
end

# 140 ms 290 ms for 1000
function gauss_deblend(x_d, Σ_a, Σ_b, μ_a, μ_b)
	Σ_ab_inv = inv(cholesky(Σ_a + Σ_b))
	Σ_ad = Σ_a*Σ_ab_inv*Σ_b

	μ_a0 = (Σ_b - Σ_a) * (Σ_ab_inv * μ_a)
	μ_b0 = (Σ_a - Σ_b) * (Σ_ab_inv * μ_b)

	μ_ad = μ_a0 .+ Σ_a*(Σ_ab_inv*x_d)
	μ_bd = μ_b0 .+ Σ_b*(Σ_ab_inv*x_d)
	return μ_ad, μ_bd, Σ_ad
end


# function newcovshift(hcube, Δrow, Δcol, Np)
# 	Np2 = size(hcube, 1)
# 	Np = (Np2+1) ÷ 2
# 	Nphalf = (Np-1) ÷ 2
# 	i1 = Np-Δcol-Nphalf:Np-Δcol+Nphalf
# 	i2 = Np-Δrow-Nphalf:Np-Δrow+Nphalf
# 	cov = hcube[i2,i1,i2,i1]
# 	covreshape = reshape(cov,Np*Np,Np*Np)
# 	return covreshape
# end



# covshift isn't really used -- we usually have a low-rank V
#  with cov = VV' and shift that.
function covshift(cov, Δrow, Δcol)
	cov2  = zeros(size(cov))
	covshift!(cov2, cov, Δrow, Δcol)
	return cov2
end


# shift covariance matrix to a different center
function covshift!(cov2, cov, Δrow, Δcol)
	diagval = 1e-6
	row1  = clamp(1-Δrow,1,Np)
	rowNp = clamp(Np-Δrow,1,Np)
	col1  = clamp(1-Δcol,1,Np)
	colNp = clamp(Np-Δcol,1,Np)

	for jcol=col1:colNp
		for jrow=row1:rowNp
			j1=jrow+Np*(jcol-1)
			j2=jrow+Δrow + Np*(jcol+Δcol-1)
			c1=(j1-1)*Np*Np-Np
			c2=Δrow + Np*(Δcol-1) + (j2-1)*Np*Np
			for icol=col1:colNp
				d1 = Np*icol + c1
				d2 = Np*icol + c2
				# for irow=clamp(1-Δrow,1,Np):clamp(Np-Δrow,1,Np)
				# 	i1=irow+d1
				# 	i2=irow+d2
				# 	cov2[i2] = cov[i1]
				# end
				cov2[row1+d2:rowNp+d2] = cov[row1+d1:rowNp+d1]
			end
		end
	end
	for k=1:Np*Np
		if (cov2[k,k] == 0) cov2[k,k] = diagval end
	end
	return cov2
end


"""
	efacs = cov_efacs(cov; Nvec=25, evals=false)
Compute the first Nvec eigenfactors of the covariance matrix.

# Arguments:
- `cov`:      covariance matrix (Ndim, Ndim)

# Keywords:
- `Nvec`:     number of eigenvectors to keep
- `evals`:    return eigenvalues also

# Output:
- `efacs`:    eigenfactors (Ndim, Nvec)

# Comments:
- Uses KrylovKit \\
- Returns eigenfactors = sqrt(eval)*eigenvector
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
function cov_efacs(cov; Nvec=25, evals=false)
	# Array size
	Ndim = size(cov, 1)
	# Allocate output array
	efacs = zeros(Ndim, Nvec)
	# Compute eigenvectors using KrylovKit
	vals,vecs,info = eigsolve(cov, Nvec, :LM)
	# scale by sqrt(eigenval)
	for i=1:Nvec efacs[:,i] = sqrt(max.(vals[i],0.0)).*vecs[i] end
	if evals
		return efacs, vals
	end
	return efacs
end


"""
	grid = image_grid(images, Npane=[6,4]; norm=false)
Map image cube to Npane[1] x Npane[2] grid of images.

# Arguments:
- `images`:   image cube (Np, Np, Ndim)

# Output:
- `grid`:     image grid

# Comments:
2022-Jan-24 - Written by Douglas Finkbeiner, CfA \\
"""
function image_grid(images, Npane=[6,4]; norm=false, flip=false, pad=0)
	
	Nx = Npane[1]   # horizontal direction
	Ny = Npane[2]
	Ndim = size(images, 3)
	Np   = size(images, 1)
	grid = zeros(Ny*(Np+pad)+pad, Nx*(Np+pad)+pad) .-1

	# loop over images and place in grid array
	for i=1:min(Ndim, Nx*Ny)
		irow = (i-1)÷Nx + 1
		icol = i - (irow-1)*Nx
		i0 = (icol-1)*(Np+pad)+(1+pad)
		if flip
			j0 = (irow-1)*(Np+pad)+(1+pad)
		else
			j0 = (Ny-irow)*(Np+pad)+(1+pad)
		end

		thisim = images[:,:,i]
		if norm thisim./=sqrt(sum(thisim.^2)) end
		grid[j0:j0+Np-1,i0:i0+Np-1] = thisim

	end
	return grid
end



function covfac_shift(efacs, irow, icol)
	Nvec = size(efacs,2)
	Np2  = round(Int, sqrt(size(efacs,1)))
	Np   = (Np2+1) ÷ 2
	i1  = Np+1-icol:Np-icol+Np
	i2  = Np+1-irow:Np-irow+Np
	bar = reshape(efacs, Np2, Np2, Nvec)
	covfac = reshape(bar[i2,i1,:],Np*Np,Nvec)
	return covfac
end



function covfactor_list(cens, efacs; Ndim=nothing, pixperasec=1)

	Nvec = size(efacs,2)
	Ncomp = length(cens)
	# per object covariance matrices
	covfaclist = Array{Float64, 3}(undef, Ndim, Nvec, Ncomp)

	for k=1:Ncomp
		cen = cens[k]
		covfaclist[:,:,k] = covfac_shift(efacs,round(Int,cen[2]*pixperasec),
		                                       round(Int,cen[1]*pixperasec))
	end

	return covfaclist
end



function covfactor_list_cshift(cens, efacs, shape=nothing)

	Nvec  = size(efacs,2)
	Ncomp = length(cens)
	if isnothing(shape)
		Nx = Int64(sqrt(size(efacs,1)))
		Ny = Ny
	else
		Nx = shape[1]
		Ny = shape[2]
	end
	Ndim = Nx*Ny
	# per object covariance matrices
	covfaclist = Array{Float64, 3}(undef, Ndim, Nvec, Ncomp)

	res_efacs = reshape(efacs,Ny,Nx,Nvec)
	for k=1:Ncomp
		cen = cens[k]
		print(size(cen))
		println(cen)
		bar = circshift(res_efacs,(round(Int,cen[2]),round(Int,cen[1]),0))
		covfaclist[:,:,k] = reshape(bar,Nx*Ny,Nvec)
	end

	return covfaclist
end


"""
	add_component_Cinv!(Cinv, W, efacs, xynew)
add one model component to Cinv and W.

# Arguments modified:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays for current `xylist` (Nobj)

# Arguments:
- `efacs`:    eigenfactors for new compoment (Ndim, Nvec)
- `xynew`:    x,y of new object, array (2)

# Comments:
- `xynew` contains 1-indexed (x,y) positions, no scaling \\
- This works for stars or galaxies, depends on the `efacs` you pass \\
2022-Feb-14 - Written by Douglas Finkbeiner, CfA \\
"""
function add_component_Cinv!(Cinv, W, efacs, xynew)

    Ndim = size(Cinv,1)

	# number of objects to add
	Nobj = length(xynew)

	# get list of covariances for objects and noise
	covfaclist = covfactor_list(xynew, efacs, Ndim=Ndim)
	V          = reshape(covfaclist,Ndim,size(covfaclist,2)*Nobj)

	# update W
	append!(W, [V])

	# Use the Woodbury Matrix Identity to update Cinv
	CinvV = Cinv*V               # 2 ms for Np=33, Nvec=20
	M = (I + (V'*CinvV))         # 0.05 ms
	Cinv .-= CinvV*(M\(CinvV'))  # 0.3 ms + 8 ms

	return 1
end


"""
	add_component_Cinv_subarray!(Cinv, W, efacs, xynew, ind)
add one model component to Cinv and W, touching only elements given by ind

# Arguments modified:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays for current `xylist` (Nobj)

# Arguments:
- `efacs`:    eigenfactors for new compoment (Ndim, Nvec)
- `xynew`:    x,y of new object, array (2)
- `ind `:     index values

# Comments:
- `xynew` contains 1-indexed (x,y) positions, no scaling \\
- This works for stars or galaxies, depends on the `efacs` you pass \\
- 20x faster for test case, but you must select `ind` carefully \\
2022-Sep-09 - Written by Douglas Finkbeiner, CfA (on DE2038) \\
"""
function add_component_Cinv_subarray!(Cinv, W, efacs, xynew, ind)
    # reduces time from 6 ms to 0.35 ms on M1, battery, 1 core
    Ndim = size(Cinv,1)

	# number of objects to add
	Nobj = length(xynew)

	# get list of covariances for objects and noise
	covfaclist = covfactor_list(xynew, efacs, Ndim=Ndim)
	V          = reshape(covfaclist,Ndim,size(covfaclist,2)*Nobj)

	# update W
	append!(W, [V])

	# Use the Woodbury Matrix Identity to update Cinv
	CinvV = Cinv[ind,ind]*V[ind,:]     # 2.9 ms for Np=33, Nvec=20, 0.11 ms for ind 192
	M = (I + (V[ind,:]'*CinvV))         # 0.06 ms
	Cinv[ind,ind] .-= CinvV*(M\(CinvV'))  # 3.6 ms  (0.1 for \) 0.18 ms

	return 1
end


"""
	delete_component_Cinv!(Cinv, W, ind)
Delete component corresponding to W[ind] from Cinv and W.

# Arguments modified:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays for current `xylist` (Nobj)

# Arguments:
- `ind`:      index of component to delete

# Comments:
- This uses the Woodbury formula to remove a component from Cinv \\
2022-Feb-20 - Written by Douglas Finkbeiner, CfA \\
"""
function delete_component_Cinv!(Cinv, W, ind)

	Ncomp = length(W)
	if ind > Ncomp throw("ind must be <= length(W)") end

	# eigenfactors to remove from Cinv
	V = W[ind]

	# update W
	deleteat!(W, ind)

	# Use the Woodbury Matrix Identity to update Cinv
	CinvV = Cinv*V               # 2 ms for Np=33, Nvec=20
	M = (I - (V'*CinvV))         # 0.05 ms
	Cinv .+= CinvV*(M\(CinvV'))  # 0.3 ms + 8 ms

	return 1
end


"""
	ΔTS = ΔTS_without_component(Cinv, W, x_d, ind)
Return change in TS if component `ind` were removed.

# Arguments modified:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays (Nobj)

# Arguments:
- `x_d`:      data vector (Ndim)
- `ind`:      index of component to delete

# Comments:
- This uses the Woodbury formula to calculate how the \\
  χ² test statistic would change if a component were removed, \\
  but without forming a new Cinv matrix.
2022-Feb-20 - Written by Douglas Finkbeiner, CfA \\
"""
function ΔTS_without_component(Cinv, W, x_d, ind)

	Ncomp = length(W)
	if ind > Ncomp throw("ind must be <= length(W)") end

	# eigenfactors to remove from Cinv
	V = W[ind]

	# Use the Woodbury Matrix Identity to compute ΔTS
	M = (I - (V'*(Cinv*V)))
	VtCinvXd = V'*(Cinv*x_d)
	ΔTS = VtCinvXd'*(M\VtCinvXd)

	return ΔTS
end


#function ΔTS_convolution
	# Use the Woodbury Matrix Identity to compute ΔTS

	# This can be done fast for diagonal Cinv
	# or for a low-rank representation of Cinv
	# Note for a stationary Cinv, M is a constant as V shifts...
	# we could work on the residual image with stationary Cinv plus pixel noise
#	M = (I - (V'*(Cinv*V)))

	# This step can be done for every offset of V with a convolution.
	# yielding (Nvec x Npix) array
#	VtCinvXd = V'*(Cinv*x_d)
#	ΔTS = VtCinvXd'*(M\VtCinvXd)

#	return
#end


"""
	μ_comp, TS = eval_apportionment_model(Cinv, W, xylist, B, data; gain=nothing)
Evaluate apportionment model.

# Arguments:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays for `xylist` (Nobj)
- `xylist`:   list of object centers (Nobj)
- `B`:        inverse covariance of background (Ndim, Ndim)
- `data`:     data, array (Np,Np) or vector (Ndim)

# Output:
- `μ_comp`:   mean estimate of model components

# Comments:
- `xylist` contains 1-indexed (x,y) positions, no scaling \\
2022-Feb-19 - Written by Douglas Finkbeiner, CfA \\
"""
function eval_apportionment_model(Cinv, W, xylist, B, data; gain=nothing)

	Nobj  = length(xylist)
	Ncomp = Nobj+1   # add one for noise/residual map
	Np    = size(data,1)

	# data vector
	x_d = vec(data)
	CinvXd = Cinv*x_d
	TS = x_d' * CinvXd
	# allocate memory for component means
	μ_comp = Array{Float64, 3}(undef,Np,Np,Ncomp)
	μ_comp[:,:,end] .= 0.0
	if !isnothing(gain) var_comp = zeros(Np*Np,Ncomp) end

	for k = 1:Nobj
		V = W[k]
		μ_0 = V*(V' * CinvXd)
		if isnothing(gain)
			μ_comp[:,:,k] = μ_0
		else
			var_comp[:,k] = μ_0 ./ gain
			μ_comp[:,:,k] = μ_0 .+ var_comp[:,k] .* CinvXd     # 0.01 ms each...
		end
	end

	# estimate noise component (component Nobj+1)
	var_rest = isnothing(gain) ? 0.0 : vec(sum(var_comp, dims=2))
	μ_comp[:,:,Ncomp] = vec(B*CinvXd - var_rest .* CinvXd)

	return μ_comp, TS
end



"""
	μ_comp = update_apportionment_model!(Cinv, W, xylist, B, data, efacs, xynew; gain=nothing)
Add one component (star, galaxy, etc.) to the model.

# Arguments modified:
- `Cinv`:     inverse covariance matrix (Ndim, Ndim)
- `W`:        list of efac arrays for current `xylist` (Nobj)
- `xylist`:   list of object centers (Nobj)

# Arguments:
- `B`:        inverse covariance of background (Ndim, Ndim)
- `data`:     data, array (Np,Np) or vector (Ndim)
- `efacs`:    eigenfactors for new compoment (Ndim, Nvec)
- `xynew`:    x,y of new object, array (2)

# Output:
- `μ_comp`:   mean estimate of model components

# Comments:
- `xylist`, `xynew` contain 1-indexed (x,y) positions, no scaling \\
- This works for stars or galaxies, depends on the `efacs` you pass \\
2022-Feb-14 - Written by Douglas Finkbeiner, CfA \\
"""
function update_apportionment_model!(Cinv, W, xylist, B, data, efacs, xynew; gain=nothing)

	if xynew[1] isa Number
		xy = [xynew]
	else
		xy = xynew
	end

	# update Cinv and W to include new object(s) at xy
	add_component_Cinv!(Cinv, W, efacs, xy)

	# append xynew to list of centers
	append!(xylist, xy)

	# evaluate the updated model
	μ_comp, TS = eval_apportionment_model(Cinv, W, xylist, B, data; gain=nothing)

	return μ_comp, TS
end


function comp_params(μ_comp)
	Np = size(μ_comp,1)
	Nphalf = (Np-1)÷2
	ramp = collect(-Nphalf:Nphalf)

	Nobj = size(μ_comp,3)

	# sum over rows at each column
	s1 = max.(0.0,sum(μ_comp,dims=1)[1,:,:])
	xcen = reshape(sum(s1.*ramp,dims=1)./sum(s1,dims=1),Nobj).+(Nphalf+1)
	s2 = max.(0.0,sum(μ_comp,dims=2)[:,1,:])
	ycen = reshape(sum(s2.*ramp,dims=1)./sum(s2,dims=1),Nobj).+(Nphalf+1)

	flux = max.(0.0,sum(μ_comp,dims=(1,2))[1,1,:])
	par  = StructArray(x=xcen, y=ycen, flux=flux)
	return par
end


function comp_params_except_last(μ_comp)
	Np = size(μ_comp,1)
	Nphalf = (Np-1)÷2
	ramp = collect(-Nphalf:Nphalf)

	Nobj = size(μ_comp,3)-1

	# sum over rows at each column
	s1 = max.(0.0,sum(μ_comp,dims=1)[1,:,1:end-1])
	xcen = reshape(sum(s1.*ramp,dims=1)./sum(s1,dims=1),Nobj).+(Nphalf+1)
	s2 = max.(0.0,sum(μ_comp,dims=2)[:,1,1:end-1])
	ycen = reshape(sum(s2.*ramp,dims=1)./sum(s2,dims=1),Nobj).+(Nphalf+1)

	flux = max.(0.0,sum(μ_comp,dims=(1,2))[1,1,1:end-1])
	par  = StructArray(x=xcen, y=ycen, flux=flux)
	return par
end

"""
	p=eigenimage_figure(bigcov; panels=[6,4], binfac=4, stretch=0.3)
Generate image grid of eigenfactor images

# Arguments:
- `bigcov`:   covariance matrix (padded in PSF case)

# Keywords:
- `panels`:   panel layout [Nx,Ny]
- `binfac`:   bin image up by binfac x binfac
- `stretch`:  color stretch factor (low is more contrast)


# Output:
- `p`:        plot object

# Comments:
Called by psf_eigenimage_figure() and starfield_eigenimage_figure() \\
2022-Apr-27 - Written by Douglas Finkbeiner, CfA \\
"""
function eigenimage_figure(bigcov; panels=[6,4], binfac=4, stretch=0.3)

	Np2 = Int64(sqrt(size(bigcov,1)))
	Nx, Ny = panels

	# compute eigenvectors weighted by eigenvalues
	Nvec=Nx*Ny
	efacs, evals = cov_efacs(bigcov, Nvec=Nvec, evals=true)
	println("szz",size(efacs))
	# reshape to Np2,Np2 images and pack into a grid
	imgs  = reshape(efacs,Np2,Np2,Nvec)
	igrid = image_grid(imgs, [Nx,Ny],norm=true)

	# bin up by 4x4 so pdf rendering is pixelated
	igrid = imresize(igrid,ratio=binfac,method=Constant())
	Np2 *= binfac

	# make image
	p = heatmap(igrid, clims=(-1,1).*stretch, c=cgrad([:red, :white, :blue]), axis=([],false))
	plot!(p,size=(750,460))

	# add annotations
	es = [@sprintf("√λ = %8.03g",sqrt(evals[i])) for i=1:Nx*Ny]
	labels = [(mod(i,Nx)*Np2+Np2*0.5,
	           (Ny-1-i÷Nx)*Np2+Np2*0.88,
			   (es[i+1], 8, :center))
			   for i=0:Nx*Ny-1]
	annotate!(p,labels)
	for i=0:Nx
		plot!(p,[Np2,Np2].*i.+0.5,[0,Ny*Np2].+0.5,line=(:black),legend=false)
	end
	for i=0:Ny
		plot!(p,[0,Nx*Np2].+0.5,[Np2,Np2].*i.+0.5,line=(:black),legend=false)
	end

	return p
end



function psf_eigenimage_figure(; Np=15, Nsam=20000, stretch=0.3, type="Moffat")

	Random.seed!(4)
	bigcov = get_covariance(Np=Np, Nsam=Nsam, type=type, smear=false)

	p=eigenimage_figure(bigcov, stretch=stretch)

	return p
end



function starfield_eigenimage_figure(; Np=15, Nsam=1000, stretch=0.2, varstar=nothing)

	Random.seed!(4)
	bigcov = starfield_covariance(Np, Nsam; varstar=varstar)

	p=eigenimage_figure(bigcov, stretch=stretch)

	return p
end



function covariance_scaling_figure(; Np=15, efacs=nothing, Nsam=20000, σ_pix=0.01, fwhm=2.5, Δx=0.0, seed=nothing, type="Moffat")

	igrid = nothing
	β     = 4.8

	paperfig = false
	Np2 = Np*2-1
	if isnothing(efacs)
		bigcov = g(Np=Np, Nsam=Nsam, type=type, smear=false)
		Nvec=20
		efacs0, evals = cov_efacs(bigcov, Nvec=Nvec, evals=true)
		x_d = twostar_Δxy_data_grid(Np, xvals, yvals; flux=flux, fwhm=fwhm, β=4.8, σ_pix=σ_pix)
	else
		efacs0=efacs
	end

	explist = [-3,-2,-1,0,1,2,3]
	expstr  = ["10^{$ec}" for ec in explist]
	efacs_coeff = 10.0 .^explist
	for ec in efacs_coeff
		scaled_efacs = efacs0.*ec
		im3 = twostar(scaled_efacs,[[8.0+Δx,12.0]],[[8.0,12.0],[8.0,4.0]],Np=15,σ_pix=σ_pix,fwhm=fwhm,seed=seed)[1]
		igrid = isnothing(igrid) ? im3[:,1:Np] : [igrid im3[:,1:Np]]
	end
	rat = 4
	Ns = Np*rat
	igrid = imresize(igrid,ratio=rat,method=Constant())
	p = heatmap(igrid,clims=(-0.07,0.07), c=cgrad([:red, :white, :blue]), axis=([],false), colorbar=false)

	Nx, Ny = length(efacs_coeff), 3
	for i=0:Nx
		plot!(p,[Ns,Ns].*i.+0.5,[0,Ny*Ns].+0.5,line=(:black),legend=false)
	end
	for i=0:Ny
		plot!(p,[0,Nx*Ns].+0.5,[Ns,Ns].*i.+0.5,line=(:black),legend=false)
	end

	labelfont = paperfig ? 11 : 18
	labels = [(Ns*(i-0.5),
	           Ns*3.15,
			   (L"\kappa = %$(expstr[i])", 18, :center))
			   for i=1:Nx]
	annotate!(p,labels)
	framelabel = [(Ns*0.5,-6,
			   (L"\mathrm{%$type ~~~ FWHM=} %$fwhm~\mathrm{pix}~~~\beta=%$β ~~~ \sigma_p=%$σ_pix ~~~ \Delta x = %$Δx ", 18, :left))]
	annotate!(p,framelabel)


	# plot circles
	θ = collect(0:4:364)*(π/180)
	rad = 3.0*rat
	sθ, cθ = rad.*sin.(θ), rad.*cos.(θ)
	plot!(p, sθ.+(Ns/2+0.5), cθ.+(1.7666*Ns+0.5),legend=false, c=:black)
	plot!(p, sθ.+(Ns/2+0.5), cθ.+(1.2333*Ns+0.5),legend=false, c=:black)

	mm = Plots.mm
	if paperfig
		plot!(p,size=(750,320), right_margin = 3Plots.mm)
	else
		plot!(p, aspect_ratio=1, right_margin = 3mm)
	end

	return p

end


# return Animation object
function animate_psf_shift(;Np=15, Nsam=10000, σ_pix=0.01, type="Moffat")
	a = Animation()
	bigcov = get_covariance(Np=Np, Nsam=Nsam, type=type, smear=false)
	Nvec=20
	efacs, evals = cov_efacs(bigcov, Nvec=Nvec, evals=true)
	bigcov = 0
	for seed in [2,nothing]
		for Δx in collect(-1:0.05:1)
			p=covariance_scaling_figure(Np=15, efacs=efacs, σ_pix=σ_pix, fwhm=2.5, Δx=Δx, seed=seed)
			plot!(size=(1280,720))
			frame(a, p)
		end
	end
	return a
end


function animate_psf_fwhm(;Np=15, Nsam=10000, σ_pix=0.01, type="Moffat")
	a = Animation()
	bigcov = get_covariance(Np=Np, Nsam=Nsam, type=type, smear=false)
	Nvec=20
	efacs, evals = cov_efacs(bigcov, Nvec=Nvec, evals=true)
	bigcov = 0

	for seed in [2,nothing]
		for fwhm in collect(1.5:0.1:3.0)
			p=covariance_scaling_figure(Np=15, efacs=efacs, σ_pix=σ_pix, fwhm=fwhm, seed=seed)
			plot!(size=(1280,720))
			frame(a, p)
		end
	end
	return a
end

# using WCS
# using FITSIO
#
# fname = "../frame-z-000125-1-0586.fits.bz2"
# byts = read(`bzcat $fname`)
# f = FITS(byts)
# h = read_header(f[1],String)
# wcs = (WCS.from_header(h))[1]
# ra = 76.55
# dec = -1.14
# world_to_pix(wcs,[ra,dec])
#
#
# using CodecBzip2
# bz2_byts = read(fname)
# byts = transcode(Bzip2Decompressor, bz2_byts)
#



# Use the Schur complement trick to add rows and columns to an already known inverse matrix. 
# Assume symmetric matrix.
#	   | A  B |
#  M = |	  |
#      | B' D |
#
function schur_add_block_inv(D, Dinv, A, B)
	nA = size(A,1)
	nD = size(D,1)
	M1 = vcat(hcat(I,zeros(nA,nD)),hcat(-Dinv*B',I))
	M2 = vcat(hcat(inv(A-B*Dinv*B'),zeros(nA,nD)),hcat(zeros(nD,nA),Dinv))

	Minv = (M1*M2)*M1'
	return Minv
end

# test schur_add_block_inv()
# generate 5x5 symmetric positive definite matrix

function schur_test()
	nA = 2
	nB = 1000
	M = randn(nA+nB,2*(nA+nB))
	M = M*M' ./ sqrt(nA+nB)

	A = M[1:nA,1:nA]
	B = M[1:nA,nA+1:end]
	D = M[nA+1:end,nA+1:end]
	Dinv = inv(D)

	#M = vcat(hcat(A,B),hcat(B',D))
	Minv = inv(M)
	Minv2 = schur_add_block_inv(D, Dinv, A, B)
	println("error ",norm(Minv-Minv2))
	return
end


# Augmented matrix times vector
function schur_add_block_inv_vec(D, Dinv, A, B, x)
	nA = size(A,1)
	x1 = x[1:nA]
	x2 = x[nA+1:end]

	BDinv = B*Dinv
	M1tx1 = x1-BDinv*x2
	#M1tx2 = x2

	M2x1 = inv(A-BDinv*B')*M1tx1
	M2x2 = Dinv*x2

	M3x1 = M2x1
	M3x2 = -BDinv'*M2x1 + M2x2
	Minvx = vcat(M3x1,M3x2)

	return Minvx
end

function test_schur_add_block_inv_vec()
	x = rand(nA+nB)
	Minvx = Minv*x   # 100 us
	Minvx2 = schur_add_block_inv_vec(D, Dinv, A, B, x)  # 280 us, 20x faster than choleskky(M)\x
	println("error ",norm(Minvx-Minvx2))
	return
end

# Compute the distribution of greatest of N draws from chi-square for dof degrees of freedom. 
function chisq_highest_of_N(N, ndof, Nsamp=100000)
	# N = number of draws
	# ndof = degrees of freedom
	# Nsamp = number of samples to draw
	# return histogram of highest of N draws

	# draw Nsamp samples of N draws from chi-square
	# this is a matrix of Nsamp x N
	x = rand(Chisq(ndof),Nsamp,N)

	# find the maximum of each row
	xmax = maximum(x,dims=2)

	# histogram the maximums
	histogram(xmax,bins=0:0.2:30)

	return h
end



# new problem.  Let's try Woodbury but using the Schur complement trick.

# Use the Schur complement trick to add rows and columns to an already known inverse matrix.
# Assume symmetric matrix.
#	   | A  B |
#  M = |	  |
#      | B' D |
#

# UU'  150 us

function more_tests()
	Nx = 1000
	Nv = 500
	Nu = 5
	V = randn(Nx,Nv)
	U = randn(Nx,Nu)
	W = hcat(U,V)
	x = randn(Nx)
	a = Diagonal(rand(Nx))

	D = V'*(a\V) + I    # 3800 us
	Dinv = inv(D)       # 3000 us
	VDinv = V*Dinv      #  200 us
	AinvVDinv = a\VDinv #  300 us
	B=U'*(a\V)          # 1100 us
	A=U'*(a\U)+I
	P=inv(A-B*Dinv*B')  #  130 us
	Q=-P*B*Dinv         #  115
	S=(I-B'*Q)
	VtAinvx = V'*(a\x)  # 25 us
	UtAinvx = U'*(a\x)  #  6 us
	DinvVtAinvx = Dinv*VtAinvx  #  46 us

	PUa = P*UtAinvx         #  0 us
	QUa = Q*VtAinvx         #  3 us
	SVa = DinvVtAinvx - Q'*B*DinvVtAinvx #  5 us
	QtUa = Q'*UtAinvx         #  1 us

	Innerinvx = vcat(PUa+QUa, SVa+QtUa)  #  1 us

	Minvx = a\x - a\(W*Innerinvx)		# 56 us

	M = a + W*W'
	return
end




# Function to generate an Npix by Npix image, containing Nstar stars.  The stars are a Moffat psf. 
# The stars are centered at random positions, and have random fluxes.
"""
    starfield(Npix, Nstar, fwhm, β, seed=1)
Generate an image of stars.

# Arguments:
- `Npix`:     image size
- `Nstar`:    number of stars
- `fwhm`:     full width at half maximum of Moffat psf
- `β`:        Moffat β parameter

# Keywords:
- `seed`:     random seed

# Output:
- `im`:       image

# Comments:
- This is a simple model of a starfield.  The stars are centered at random positions, and have random fluxes.
- The image is a sum of Moffat psfs, each centered at a random position, and with a random flux.

TBW
"""
function starfield(Npix, Nstar, fwhm, β, seed=1)
	Random.seed!(seed)
	# generate a grid of x,y positions
	xvals = collect(1:Npix)
	yvals = collect(1:Npix)
	xx = repeat(xvals,1,Npix)
	yy = repeat(yvals',Npix,1)
	# generate random positions
	x = rand(1:Npix,Nstar)
	y = rand(1:Npix,Nstar)
	# generate random fluxes
	flux = randn(Nstar)
	# generate the image
	im = zeros(Npix,Npix)
	for i=1:Nstar
		#Moffat_model(cen::Vector; Np::Int=33, fwhm=2.5, β=4.8)
		# add star to image using Moffat_model()
		im .+= Moffat_model([x[i],y[i]],Np=Npix,fwhm=fwhm,β=β).*flux[i]
	end
	return im
end

# test starfield
function test_starfield()
	im = starfield(100, 300, 2.5, 4.8)
	p = heatmap(im, clims=(0,1), c=:cividis, axis=([],false))
	return p
end



"""
    psf_efacs(Nvec=20; Np=33, Nsam=20000, σ_cen=0.25, μ_fwhm=nothing, σ_fwhm=0.1, objtype=nothing, μ_β=nothing, σ_β=0.1, uniform=false)
	Generate eigenfactors for a Moffat or Gaussian PSF.

	Parameters
	----------
	Nvec : Int
		Number of eigenfactors.
	Np : Int
		Size of image in pixels.
	Nsam : Int
		Number of samples.
	σ_cen : Float64
		Standard deviation of the centroid.
	μ_fwhm : Float64
		Mean FWHM of the PSF.
	σ_fwhm : Float64
		Standard deviation of the FWHM.
	objtype : String
		Type of object: "Moffat", "Gaussian", or "Both".
	μ_β : Float64
		Mean β of the Moffat PSF.
	σ_β : Float64
		Standard deviation of β.
	uniform : Bool
		Uniform sampling of position within pixel (to nearest pixel center)

	Returns
	-------
	efacs : Array{Float64,2}
		Eigenfactors for the PSF.

	Notes
	-----
	Generate eigenfactors for a Moffat or Gaussian PSF.
	For Moffat, the samples are normalized to sum to 1.0 if not near edge.
	For Gaussian, the samples are normalized to sum to 1.0.
	For Moffat, μ_β must be specified.
	This version avoids explicit construction of the covariance matrix.
Generate eigenfactors (sqrt(eval)*evec)for a Moffat or Gaussian PSF
"""
function psf_efacs(Nvec=20; Np=33, Nsam=20000, σ_cen=0.25, μ_fwhm=nothing, σ_fwhm=0.1, objtype=nothing, μ_β=nothing, σ_β=0.1, uniform=false)

	if objtype=="Moffat" && isnothing(μ_β) throw(ArgumentError("μ_β must be specified for Moffat PSF")) end
		
	# padded grid size
	Np2 = Np*2-1

	# generate covariance matrix for padded region (to make shifts easy)
	# Moffat and Gaussian samples are normalized to sum to 1.0 if not near edge
	if objtype=="Moffat"
		psfs  = Moffat_model_samples(Np2, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, smear=false, uniform=uniform)
	elseif objtype=="Gaussian"
		psfs  = Gaussian_model_samples(Np2, Nsam; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, smear=false, uniform=uniform)
	elseif objtype=="Both"
		psfs1 = Moffat_model_samples(Np2, Nsam÷2; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, μ_β=μ_β, σ_β=σ_β, smear=false, uniform=uniform)
		psfs2 = Gaussian_model_samples(Np2, Nsam÷2; σ_cen=σ_cen, μ_fwhm=μ_fwhm, σ_fwhm=σ_fwhm, smear=false, uniform=uniform)
		psfs  = cat(psfs1, psfs2, dims=3)
	end
	
	Dstar = reshape(psfs,Np2*Np2,:)
   
	# this returns covariance times vector
	eigsolve_helper(xi) = Dstar*(Dstar'*xi)/Nsam

	# Array size
	Ndim = Np2*Np2

	# Compute eigenvectors using KrylovKit
	vals,vecs,info = eigsolve(eigsolve_helper, Ndim, Nvec, :LM, issymmetric=true)

	# Allocate output array
	efacs = zeros(Ndim, Nvec)

	# scale by sqrt(eigenval)
	for i=1:Nvec efacs[:,i] = sqrt(max.(vals[i],0.0)).*vecs[i] end

	return efacs
end


"""
    generate_obj_cube(objxy; Np=33, σ_pix=0.003, fwhm=nothing, seed=nothing, objtype="Moffat")
	Generate a set of objects at positions objxy, with a given PSF FWHM and object type.
	
	Parameters
	----------
	objxy : List of 2-element arrays
		List of object positions (x,y), i.e. (col,row)
	Np : Int
		Size of image in pixels.
	σ_pix : Float64
		Noise level in pixels.
	fwhm : Float64
		FWHM of PSF in pixels.
	seed : Int
		Random seed.
	objtype : String
		type of object: "Moffat", "Gaussian", or "Sersic".

	Returns
	-------
	img : Array{Float64,3}
		Image cube, Np x Np x Nobj.
		
	Notes
	-----
	Time for a single star, Np=33:   Gaussian: 13 usec   Moffat: 30 usec
    Time for 1000 stars, Np=33:      Gaussian: 11 msec   Moffat: 27 msec
TBW
"""
function generate_obj_cube(objxy; Np::Int=33, σ_pix::Float64=0.0, fwhm=nothing, seed=nothing, objtype="Moffat",β=3.5)
	Random.seed!(seed)     # seed=nothing is a random seed
	Nobj = length(objxy)

    # allocate undefined array for image cube
    img = Array{Float64,3}(undef,Np,Np,Nobj)

    # add objects to image cube
	for k=1:Nobj
		if objtype=="Moffat"
			img[:,:,k] = Moffat_model(objxy[k]; Np=Np, fwhm=fwhm, β=β)
		elseif objtype=="Gaussian"
			img[:,:,k] = Gaussian_model(objxy[k]; Np=Np, fwhm=fwhm)
		elseif objtype=="Sersic"
            println("Sersic not implemented yet")            
			#img[:,:,k] = Sersic_model(objxy[k]; Np=Np, fwhm=fwhm, n=1.0)
            return 0
		end
	end

    if σ_pix > 0.0
        img += (randn(Np,Np,Nobj) .* σ_pix)
    end

	# if Nobj == 1 return a single image
	if Nobj == 1
		return img[:,:,1]
	end

	return img
end



function comp_x_and_flux(μ_comp)
	Np = size(μ_comp,1)
	Nphalf = (Np-1)÷2
	ramp = collect(-Nphalf:Nphalf)

	Nobj = size(μ_comp,3)
	# sum over rows at each column
	s1 = max.(0.0,sum(μ_comp,dims=1)[1,:,:])
	xcen = reshape(sum(s1.*ramp,dims=1)./sum(s1,dims=1),Nobj).+(Nphalf+1)

	flux = max.(0.0,sum(μ_comp,dims=(1,2))[1,1,:])
	par  = StructArray(x=xcen, flux=flux)
	return par
end




"""
    fluxbias_SNR_plot(res)
	Plot fractional flux bias vs SNR
	
	Parameters
	----------
	res : DataFrame
		DataFrame with columns SNR, fmean, fstd, μ_fwhm, σ_fwhm, σ_cen, Np, Ntrial, Neff, dofwt

	Returns
	-------
	plot :
		Plot object
		
"""
function fluxbias_SNR_plot(res; ylims=(-0.05,0.1))
    # Theoretical model
	xrange = extrema(res[!,"SNR"])
	xx = xrange[1]:xrange[2]
	p=plot(xx,-1.0./xx.^2 .- 0.0./xx.^4, label="model")
	plot!(p,xx,xx.*0,line=(:gray),label=nothing)

    # list of unique FWHMs
	FWHMs = unique(res.μ_fwhm)

	# loop over FWHMs
	for fwhm in FWHMs
		w=findall(res.μ_fwhm .== fwhm)
		scatter!(p,res.SNR[w], res.fmean[w] .-1, label = "Ma FWHM $fwhm", 
			xlabel="SNR", ylabel="Fractional flux bias")
	end

    # loop over FWHMs again for aperture photometry
    for fwhm in FWHMs
        w=findall(res.μ_fwhm .== fwhm)
        scatter!(p,res.SNR[w], res.apmean[w] .-1, label = "Ap FWHM $fwhm")
        plot!(p,res.SNR[w], res.apmean[w] .-1, label = nothing, line=(:gray,:dash))
    end

	# set plot limits
	plot!(xlims=(xrange[1]-1,xrange[2]+1), ylims=ylims)

	return p
end



function fluxstd_SNR_plot(res)
    # Theoretical model
	xrange = extrema(res[!,"SNR"])
	xx = xrange[1]:xrange[2]
	p=plot(xx,1.0./xx, label="model")
	plot!(p,xx,xx.*0,line=(:gray),label=nothing)

	# list of unique FWHMs
	FWHMs = unique(res.μ_fwhm)

	# loop over FWHMs
	ferr = res.fstd
	for fwhm in FWHMs
		w=findall(res.μ_fwhm .== fwhm)
		scatter!(p,res.SNR[w], ferr[w], label = "Ma FWHM $fwhm", 
			xlabel="SNR", ylabel="Fractional flux std")
	end

    # loop over FWHMs again for aperture photometry
    for fwhm in FWHMs
        w=findall(res.μ_fwhm .== fwhm)
        scatter!(p,res.SNR[w], res.apstd[w], label = "Ap FWHM $fwhm")
        plot!(p,res.SNR[w], res.apstd[w], label = nothing, line=(:gray,:dash))
    end

	# set plot limits
	plot!(xlims=(xrange[1]-1,xrange[2]+1), ylims=(0.0,0.40))

	return p
end




# times for Np=27   20 usec for 1 star
#                   370 usec for 100 (5x faster than generating Moffats)
# ~ 1/3 or 1/2 of time is allocating
function MADGICS_single_star(Cinv, V, img)
	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy)
	x_d    = reshape(img, sx*sy, Nimg)

	CinvXd = Cinv*x_d  # 0.7 usec    47
	CinvV  = Cinv*V  # 3.8 usec

	M = (I + (V'*CinvV))  # 4.7 usec
	CinvXd .-= CinvV*(M\(CinvV'*x_d))  # 6.4 usec    178

	μ_comp = V*(V' * CinvXd) # 4.5 usec    138
	return μ_comp
end


function MADGICS_single_star2(Cinv, V, img)   # slightly faster
	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy)
	x_d    = reshape(img, sx*sy, Nimg)

	CinvV  = Cinv*V  # 3.8 usec
	M = (I + (V'*CinvV))  # 4.7 usec

	blarg = V'*CinvV*(M\(CinvV'*x_d))  #   63
	μ_comp = V*(V'*Cinv*x_d - blarg) # 4.5 usec    148
	return μ_comp
end


# times Np=27 on M1, Nvec=20, Cinv diag   137 usec
# with 2nd iteration takes 3 ms
function MADGICS_double_star!(μ_comp1, μ_comp2, Cinv, V1, V2, x1, y1, x2, y2, img; σ_pix=σ_pix, Niter=1, doPCA=false, fbpoly=nothing)  
	# if μ_comp is nothing or undefined, then crash
	if isnothing(μ_comp1) error("μ_comp1 is nothing") end
	if isnothing(μ_comp2) error("μ_comp2 is nothing") end


	Np = round(Int,sqrt(size(V1,1)))
	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy) # works for both 2-D and 3-D img
	#if Nimg != 1
	#	error("MADGICS_double_star! only works for Nimg=1")
	#end
	Nphalf = (Np+1)÷2
	x_d    = reshape(img, sx*sy, Nimg)

	# === Iteration 1 ===
	# concatenate V1 and V2 to make W
	W = hcat(V1,V2)  # 14 usec

	CinvV = Cinv*V1
	M = I + (V1'*CinvV)
	VtCinvXd = CinvV'*x_d
	CtotinvXd1 = Cinv*x_d - CinvV*(M\(VtCinvXd))
	V2tBinv = V2'*(Cinv- CinvV*(M\(CinvV')))
	Δχ2 = sum(x_d.*  (V2tBinv'*(Symmetric(I+V2tBinv*V2)\(V2tBinv*x_d))) ,dims=1)



	CinvW = Cinv*W  # 20 usec
	# M = (I + (W'[:,ind]*CinvW[ind,:]))  # much faster with ind
	WtCinvW = Symmetric(W'*CinvW)
	M = I + WtCinvW  # 60 usec
	WtCinvXd = CinvW'*x_d        #   8 usec
	CtotinvXd = Cinv*x_d - CinvW*(cholesky(M)\(WtCinvXd))  #  16 usec
	#Δχ2 = sum(x_d.*CtotinvXd,dims=1)  #  8 usec
	#PCA version
	if doPCA
		alph = cholesky(WtCinvW)\(CinvW'*x_d)
		mul!(μ_comp1, V1, alph[1:size(V1,2),:])  # 8 usec
		mul!(μ_comp2, V2, alph[1+size(V1,2):end,:])  # 8 usec
	else
		mul!(μ_comp1, V1, V1'*CtotinvXd)  # 8 usec
		mul!(μ_comp2, V2, V2'*CtotinvXd)  # 8 usec
	end

	if Niter == 1 return Δχ2 end
	# === Iteration 2 ===
	# determine the shift in the star positions
	
	fpar = comp_x_and_flux(reshape(μ_comp1, Np, Np, Nimg))  # 2 usec
	dx1 = clamp.((fpar.x .-Nphalf) .- x1, -0.55, 0.55)  # 2 usec
	dy1 = 0.0
	flux1 = fpar.flux
	if ~isnothing(fbpoly) flux1 = flux1 ./ (fbpoly.(dx1)) end
	
	reweight_eigenvalues!(μ_comp2, V2, σ_pix)
	fpar = comp_x_and_flux(reshape(μ_comp2, Np, Np, Nimg))
	dx2 = clamp.((fpar.x .-Nphalf) .- x2, -0.55, 0.55)
	dy2 = 0.0
	flux2 = fpar.flux
	if ~isnothing(fbpoly) flux2 = flux2 ./ (fbpoly.(dx2)) end

	for i = 1:Nimg
		# shift V1 and V2
		V1shift = (1*flux1[i]/sum(V1[:,1])).*V_subpixel_shift(V1, dx1[i], dy1)  # 1.5 ms !!!!
		V2shift = (1*flux2[i]/sum(V2[:,1])).*V_subpixel_shift(V2, dx2[i], dy2)

		# concatenate V1 and V2 to make W
		W = hcat(V1shift,V2shift)  # 14 usec
		CinvW  = Cinv*W  # 20 usec

		M = Symmetric(I + (W'*CinvW))  # 60 usec
		WtCinvXd = CinvW'*x_d[:,i]        #   8 usec
		CtotinvXd = Cinv*x_d[:,i] - CinvW*(cholesky(M)\(WtCinvXd)) 
		μ_comp1[:,i] = V1shift*(V1shift'*CtotinvXd)  # 8 usec
		μ_comp2[:,i] = V2shift*(V2shift'*CtotinvXd)  # 8 usec
		#thiscomp = μ_comp2[:,[i]]
		#println("dx1", dx1[i], "dx2", dx2[i], "flux1", flux1[i], "flux2", flux2[i])
		#println("nan check",sum(isfinite.(thiscomp)))
		#reweight_eigenvalues!(thiscomp, V2shift, σ_pix)
		#μ_comp2[:,[i]] = thiscomp
	end
	

	return Δχ2
end



function MADGICS_single_star_inprogress!(μ_comp, Cinv, V, img)   # slightly faster
	# if μ_comp is nothing or undefined, then crash
	if isnothing(μ_comp)
		# exit with error message
		error("μ_comp is nothing")
	end

	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy) # works for both 2-D and 3-D img
	x_d    = reshape(img, sx*sy, Nimg)
	Nphalf = (sx-1)÷2

	# iteration 1
	MADGICS_single_star!(μ_comp, Cinv, V, img) 
	fpar = comp_x_and_flux(reshape(fcomp, sx, sy, Nimg))
	flux[j,i,:] = fpar.flux
	dx[j,i,:]   = fpar.x .- Nphalf             # loop over indexes???????????????????

	# iteration 2
	Vshift = V_subpixel_shift(V, dx[j,i,:])
	MADGICS_single_star!(μ_comp, Cinv, Vshift.*flux, img)
	return
end


function MADGICS_single_star!(μ_comp, Cinv, V, img; Niter=2)   # slightly faster
	# if μ_comp is nothing or undefined, then crash
	if isnothing(μ_comp)
		# exit with error message
		error("μ_comp is nothing")
	end

	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy) # works for both 2-D and 3-D img
	x_d    = reshape(img, sx*sy, Nimg)

	CinvV  = Cinv*V  # 3.8 usec
	M = (I + (V'*CinvV))  # 4.7 usec

	VtCinvXd = (CinvV'*x_d)        #   53 usec for 100
	blarg = V'*CinvV*(M\VtCinvXd)  #   9 usec
	mul!(μ_comp, V,(VtCinvXd - blarg)) #  37 usec
	return
end


fastmul! = AppleAccelerateLinAlgWrapper.gemm!
function MADGICS_single_star_acc!(μ_comp, Cinv, V, img)   # slightly faster
	sx, sy = size(img)
	Nimg   = length(img) ÷ (sx*sy)
	x_d    = reshape(img, sx*sy, Nimg)
	Nvec   = size(V,2)

	CinvV  = Cinv*V  # 3.8 usec
	M = (I + (V'*CinvV))  # 4.7 usec

	VtCinvXd = zeros(Nvec, Nimg)
	fastmul!(VtCinvXd, CinvV',x_d)        #   18 usec
	VtCinvXd .-= V'*CinvV*(M\VtCinvXd)  #   13 usec
	fastmul!(μ_comp,V,VtCinvXd) #  26 usec
	return
end



########  ##     ##  #######  ########  #######  ##     ## ######## ######## ########  ##    ## 
##     ## ##     ## ##     ##    ##    ##     ## ###   ### ##          ##    ##     ##  ##  ##  
##     ## ##     ## ##     ##    ##    ##     ## #### #### ##          ##    ##     ##   ####   
########  ######### ##     ##    ##    ##     ## ## ### ## ######      ##    ########     ##    
##        ##     ## ##     ##    ##    ##     ## ##     ## ##          ##    ##   ##      ##    
##        ##     ## ##     ##    ##    ##     ## ##     ## ##          ##    ##    ##     ##    
##        ##     ##  #######     ##     #######  ##     ## ########    ##    ##     ##    ##    


"""
    simple_aperture_photometry(img, objrad, skyrad=(6,8))
	Compute flux and flux error, skymean, and skyrms for input image cube

	Parameters
	----------
	img : 3D array
		Image cube
	objrad : Float
		Aperture radius
	skyrad : Tuple
		Inner and outer radii of sky annulus

	Returns
	-------
	flux : Float
		Flux
	fluxerr : Float
		Flux error
	skymean : Float
		Mean sky value
	skyrms : Float
		Sky rms

	Notes
	-----
	160 ns each
    This function assumes uniform Gaussian noise.  For bright stars, must include Poisson noise also.
	
TBW
"""
function simple_aperture_photometry(img, objrad, skyrad=(6,8))
	Np = size(img,1)
	Nphalf = Np÷2+1

	# distance from image center
	dist = reshape([sqrt((i-Nphalf)^2 + (j-Nphalf)^2) for i=1:Np, j=1:Np], Np*Np)  # 3 usec

	# list of pixels in annulus
	wh = findall((dist .> skyrad[1]) .& (dist .< skyrad[2]))  # 3 usec
	NannPix = length(wh)

	# compute sky
	img2 = reshape(img,Np*Np,size(img,3))  
	@views imgwh = img2[wh,:]  # 1.5 usec
	skymean = mean(imgwh,dims=1)[1,:]  # 42 usec
    # skyrms is the RMS of the sky pixels, NOT the rms of the mean sky value
    skyrms  = std(imgwh,dims=1)[1,:]  # 83 usec

	# list of pixels within aperture
	wh = findall(dist .< objrad)  # 1.5
	NobjPix = length(wh)

	# compute flux, fluxerr
	@views imgwh = img2[wh,:]
	flux = sum(imgwh,dims=1)[1,:] .- skymean .* NobjPix  # 32 usec
	fluxerr = skyrms .* sqrt(NobjPix + NobjPix^2/NannPix)  #  per pixel noise + sky error

	return flux, fluxerr, skymean, skyrms
end


using LsqFit
"""
    PSF_photometry(img, objxy, fwhm)
    Do PSF photometry on a single star

    Parameters
    ----------
    img : 2D array
        Image
    objxy : Tuple
        Initial guess of object position
    fwhm : Float
        Initial guess of FWHM

    Returns
    -------
    p : Array
        Best fit parameters

    Notes
    -----
    0.75 msec each
"""
function PSF_photometry(img::AbstractMatrix{Float64}, objxy, fwhm; β=3.5, objtype="Gaussian")
    Np = size(img,1)

    # initial guess for Parameters
    p0 = [objxy[1], objxy[2], fwhm, 1.0]
    lb = [objxy[1]-1.0, objxy[2]-1.0, fwhm-0.5, 0.0]
    ub = [objxy[1]+1.0, objxy[2]+1.0, fwhm+0.5, Inf]

    if objtype=="Moffat"
        println("mofo")
        Mmodel(_,p) = reshape(Moffat_model([p[1],p[2]], fwhm=p[3], Np=Np, β=β),Np*Np)*p[4]
        fit = LsqFit.curve_fit(Mmodel, [0], reshape(img,Np*Np), p0, lower=lb, upper=ub, autodiff=:forward)
        return fit
    elseif objtype=="Gaussian"
        println("gaussian")
        Gmodel(_,p) = reshape(Gaussian_model([p[1],p[2]], fwhm=p[3], Np=Np),Np*Np)*p[4]
        fit = LsqFit.curve_fit(Gmodel, [0], reshape(img,Np*Np), p0, lower=lb, upper=ub, autodiff=:forward)
        return fit
    end

    return 0
end    




# another method for PSF photometry for image cube
function PSF_photometry(img::AbstractArray{AbstractFloat,3}, objxy::AbstractArray, fwhm::Float64; Np=33, β=3.5, objtype="Gaussian")
    # check that length of objxy equals number of images
    println("in big routine")
    if length(objxy) != size(img,3)
        error("length of objxy must equal number of images")
    end
    # loop over images
    Nimg = size(img,3)
    fit = zeros(4,Nimg)
    for i=1:Nimg
        thisfit = PSF_photometry(img[:,:,i], objxy[i], fwhm; Np=Np, β=β, objtype=objtype).param
        if !thisfit.converged
            println("PSF_photometry: fit did not converge for image $i")
        end
        fit[:,i] = thisfit
    end

    return fit
end



    #######            ######  ########    ###    ########   ######  
   ##     ##          ##    ##    ##      ## ##   ##     ## ##    ## 
          ##          ##          ##     ##   ##  ##     ## ##       
    #######            ######     ##    ##     ## ########   ######  
   ##                       ##    ##    ######### ##   ##         ## 
   ##                 ##    ##    ##    ##     ## ##    ##  ##    ## 
   #########           ######     ##    ##     ## ##     ##  ######  




"""
   twostar_Δxy_data_grid(Np=33, xvals=-2.5:0.05:2.5, yvals=-2.5:0.05:2.5;
   flux=[1.0, 1.0], fwhm=nothing, β=nothing, σ_pix=0.03)

   Create set of mock images of two stars, each with a flux, fwhm, and β. 
   The star separation is (Δx, Δy), with a flux-weighted center
   at the center of the grid.

   # Arguments
   - `Np`: number of pixels in each dimension
   - `xvals::Vector{Float64}=-2.5:0.05:2.5`: Δx values for grid
   - `yvals::Vector{Float64}=-2.5:0.05:2.5`: Δy values for grid
   - `flux::Vector{Float64}=[1.0, 1.0]`: fluxes of the two stars
   - `fwhm`: FWHM of the Moffat profile
   - `β`: β of the Moffat profile
   - `σ_pix::Float64=0.03`: pixel noise

   # Returns
   - `x_d::Array{Float64,2}`: 2xNimg array of offsets relative to center of grid

   # Comments
   grid of mock data with (Δx,Δy) offsets for twostar test
   star 1 starts lower left, moves right through each row, then up. 
   Takes 0.25 sec for 5041 images and Np=27,  0.36 sec for Np=33

   # Example
   ```jldoctest
   julia> x_d = twostar_Δxy_data_grid(27, flux=[10.0,20.0], fwhm=2.5,β=4.8)
   ```
Mock images with 2 stars
"""
function twostar_Δxy_data_grid(Np=33, xvals=-2.5:0.05:2.5, yvals=-2.5:0.05:2.5;
   flux=[1.0, 1.0], fwhm=nothing, β=nothing, σ_pix=0.03)

   # prepare coefficients for offsets (to maintain flux-weighted center)
   fsum        = flux[1]+flux[2]
   frat1,frat2 = flux./fsum
   
   # number of pixels in each test image
   Ndim  = Np*Np

   # size of array of offsets
   Nx    = length(xvals)
   Ny    = length(yvals)
   Nimg  = Nx*Ny

   # δ is the 2xNimg array of offsets relative to cen
   x_d = randn(Ndim, Nimg).*σ_pix
   δi = [i for j=yvals for i=xvals]
   δj = [j for j=yvals for i=xvals]
   δ  = [δi δj]'
   cen = [Np+1,Np+1] .÷ 2

   # loop over images, placing two stars in each
   for k=1:Nimg
	   @views thisimg = reshape(x_d[:,k],Np,Np)
	   thisimg .+= Moffat_model(cen.-δ[:,k].*frat2; Np=Np, fwhm=fwhm, β=β) .*flux[1]
	   thisimg .+= Moffat_model(cen.+δ[:,k].*frat1; Np=Np, fwhm=fwhm, β=β) .*flux[2]
   end

   return x_d
end


"""
    ΔTS_for_new_component(Cinv_diag, Cinv_const, V, x_d; VtCinvXd=nothing)
	Compute the ΔTS for a two-star case for Nimg images. 

	Parameters
	----------
	Cinv_diag : Array{Float64,1}
		Diagonal of the inverse covariance matrix.
	Cinv_const : Float64
		Constant term in the inverse covariance matrix.
	V : Array{Float64,2}
		Covariance factor list
	x_d : Array{Float64,2}
		Data matrix, each column corresponds to one image.
	VtCinvXd : Array{Float64,2}
		V'*Cinv*x_d.  If not provided, it is computed.

	Returns
	-------
	ΔTS : Array{Float64,1}
		ΔTS for the two-star case.

	Notes
	-----
	Compute change in TS when adding a new component described by V. 
	Uses the Woodbury Matrix Identity to compute the ΔTS.
	For speed (10x faster), pass pre-computed VtCinvXd if available.
Compute ΔTS for an added component
"""
function ΔTS_for_new_component(Cinv_diag, Cinv_const, V, x_d; VtCinvXd=nothing)

	# Use the Woodbury Matrix Identity to compute ΔTS
	CinvV = (Cinv_diag.*V) .+ (Cinv_const.*sum(V,dims=1)) #  11 μs
	M = (I + (V'*(CinvV)))  # 12 μs
	if isnothing(VtCinvXd)  VtCinvXd = CinvV'*x_d end   # 3 ms
	ΔTS = -sum(VtCinvXd.*(inv(cholesky(Symmetric(M)))*VtCinvXd),dims=1) # 0.28 ms

	return ΔTS[1,:]
end


"""
    twostar_configurations_ΔTS(Np, x_d, efacs; σ_pix=0.03, σ_back=0.001)
	Compute the ΔTS for many configurations of two stars.

	Parameters
	----------
	Np : Int
		Size of image in pixels.
	x_d : Array{Float64,2}
		Data matrix, each column corresponds to one image.
	efacs : Array{Float64,2}
		List of covariance factors for each pixel.
	σ_pix : Float64
		Noise level in pixels.
	σ_back : Float64
		Background noise level.

	Returns
	-------
	TS : Array{Float64,2}
		ΔTS for the two-star case for each configuration.

	Notes
	-----
	Compute ΔTS on a stack of images for various two-star cases 
	  (including two at center, i.e. one)
	Uses the Woodbury Matrix Identity to compute the ΔTS.
	13 ms for 5041 images, ring2=false
	23 ms for ring2=true

Compute ΔTS for many configurations of two stars
"""
function twostar_configurations_ΔTS(Np, x_d, efacs; σ_pix=0.03, σ_back=0.001, ring2=false)

	Ndim = size(x_d, 1)
	Nvec = size(efacs, 2)

	# list of offsets, separation 1
	Δ1 = [(0,0),(0,0),(0,0),(0,0), (0,0), (0,0),  (0,0), (0,0), (0,0)]
	Δ2 = [(0,0),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

	# separation 2
	append!(Δ1,[( 1, 1),( 1, 0),( 1,-1),( 0,-1)])
	append!(Δ2,[(-1,-1),(-1, 0),(-1, 1),( 0, 1)])
	if ring2
		append!(Δ1,[( 2, 2),( 2, 0),( 2,-2),( 0,-2),( 2, 1),( 2,-1),( 1,-2),(-1,-2)])
		append!(Δ2,[(-2,-2),(-2, 0),(-2, 2),( 0, 2),(-2,-1),(-2, 1),(-1, 2),( 1, 2)])
	end

	ulist  = unique(vcat(Δ1, Δ2))
	Ntrial = length(Δ1)
	cen = [Np+1,Np+1] .÷ 2

	# The inverse covariance is diagm(Cinv_diag) .+ Cinv_const
	Cinv_diag = zeros(Ndim) .+ 1.0/σ_pix^2   # 2 μs
	Cinv_const = -σ_back^2 / (σ_pix^2 * (Ndim*σ_back^2 + σ_pix^2))

	xynewgrid  = [u .+ cen for u in ulist]
	covfaclist = covfactor_list(xynewgrid, efacs, Ndim=Ndim)
	V          = reshape(covfaclist,Ndim,size(covfaclist,2)*length(xynewgrid)) 

	CinvV = (Cinv_diag.*V) .+ (Cinv_const.*sum(V,dims=1)) #  66 μs
	VtCinvXd = CinvV'*x_d   # 21 ms

	# Do all Nimg images, allocate TS array for output
	Nimg = size(x_d, 2)
	TS = Array{Float64,2}(undef,Ntrial,Nimg)  # 2 μs

	for k=1:Ntrial
		# get list of covariances for the two candidate positions
		ind1 = findall(xynewgrid .== [cen.+Δ1[k]])[1]
		ind2 = findall(xynewgrid .== [cen.+Δ2[k]])[1]
		Vrange1 = Nvec*(ind1-1)+1:Nvec*ind1
		Vrange2 = Nvec*(ind2-1)+1:Nvec*ind2
		V12 = hcat(V[:,Vrange1], V[:, Vrange2]) # 25 μs
		VtCinvXd12 = VtCinvXd[[Vrange1; Vrange2],:] # 80 μs
		TS[k,:] = ΔTS_for_new_component(Cinv_diag, Cinv_const, V12, x_d, VtCinvXd=VtCinvXd12)  # 400 μs
	end

	return TS
end


"""
    twostar_fig_ΔTS(xvals, yvals; Np=33, fwhm=4.0, σ_pix=0.01, σ_back=0.001, SNR=[10.0,10.0], Ntrial=100)
	Compute the ΔTS for a grid of binary star configurations.

	Parameters
	----------
	xvals : Array{Float64,1}
		Δx values for grid
	yvals : Array{Float64,1}
		Δy values for grid
	Np : Int
		Size of image in pixels.
	fwhm : Float
		FWHM of the Moffat profile.
	σ_pix : Float
		Noise level in pixels.
	σ_back : Float
		Background noise level.
	SNR : Array{Float64,1}
		SNR of the two stars.
	Ntrial : Int
		Number of trials (to reduce noise in figure)

	Returns
	-------
	img : Array{Float64,2}
		ΔTS for the two-star case for each configuration.	
	imin : Array{Int,2}
		Index of minimum value for each configuration.

	Notes
	-----
	Compute ΔTS for a grid of binary star configurations.
	For each configuration, the ΔTS is the difference between
	the best 2-star case and the 1-star case.

Compute ΔTS for a grid of binary star configurations
"""
function twostar_fig_ΔTS(xvals, yvals; Np=33, fwhm=4.0, σ_pix=0.01, σ_back=0.001, SNR=[10.0,10.0], Ntrial=100)

	# generate psf covariance and its eigenfactors
	Nvec = 8

	# generate efacs for the PSF 1.5s for 20000
	efacs = psf_efacs(Nvec; Np=Np, Nsam=20000, σ_cen=0.25, μ_fwhm=fwhm, σ_fwhm=0.1, objtype="Moffat", μ_β=4.8, σ_β=0.1, uniform=false)
	Neff = 1.0/(sum(efacs[:,1].^2))

	# flux as a function of SNR and Neff and σ_pix
	flux = SNR .* (σ_pix*sqrt(Neff))
	
	TS=nothing
	ΔTS = zeros(length(yvals)*length(xvals),Ntrial)
	# compute ΔTS for all the cases (relative to no-star case)
	# timing -- first takes 1.4 ms, subsequent take 3.5-4 μs each.
	for i=1:Ntrial
		x_d = twostar_Δxy_data_grid(Np, xvals, yvals; flux=flux, fwhm=fwhm, β=4.8, σ_pix=σ_pix) # 110 ms
		TS  = twostar_configurations_ΔTS(Np, x_d, efacs, σ_pix=σ_pix, σ_back=σ_back, ring2=false)   # 13 ms for 2 thread

	# this ΔTS is difference between "1-star" case and best 2-star case.
		ΔTS[:,i] = TS[1,:] - minimum(TS[2:end,:],dims=1)[1,:]
	end
	img = reshape(mean(ΔTS, dims=2), length(yvals), length(xvals))

	# index of minimum value (for the last trial only)
	imin = reshape([argmin(TS[:,i]) for i=1:size(ΔTS, 1)], length(yvals), length(xvals))

	return img, imin
end





   ########  ##        #######  ########  ######  
   ##     ## ##       ##     ##    ##    ##    ## 
   ##     ## ##       ##     ##    ##    ##       
   ########  ##       ##     ##    ##     ######  
   ##        ##       ##     ##    ##          ## 
   ##        ##       ##     ##    ##    ##    ## 
   ##        ########  #######     ##     ######  
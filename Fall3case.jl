module Fall3case

using Statistics, LinearAlgebra

"GARCH(1,1) at (α0, α1, β1)"
function garch(ϵ,θ)
    (α0,α1,β1) = θ
    σ² = fill(NaN,length(ϵ))
    if α0>0 && α1≥0 && β1≥0 && α1+β1<1
        σ²[1] = α0/(1-β1-α1)
        for i = 2:length(σ²)    
            σ²[i] = α0+α1*ϵ[i-1]^2+β1*σ²[i-1]
        end
    end
    return σ²
end

"GARCH(1,1) partials wrt (α0, α1, β1)"
function garch_grad(ϵ,θ)
    (α0,α1,β1) = θ
    σ² = garch(ϵ,θ)
    grad = fill([NaN;NaN;NaN],length(σ²))
        if α0>0 && α1≥0 && β1≥0 && α1+β1<1
        grad[1] = [
            1/(1-β1-α1);
            α0/(1-β1-α1)^2;
            α0/(1-β1-α1)^2 ]
        for i = 2:length(grad)
            grad[i] = [
                1+β1*grad[i-1][1];
                ϵ[i-1]^2+β1*grad[i-1][2];
                σ²[i-1]+β1*grad[i-1][3] ]
        end
    end
    return grad
end

"negative quasi log-likelihood for GARCH"
function qmle_obj(ϵ,θ)
    σ² = garch(ϵ,θ)
    return (log.(2π*σ²)+ϵ.^2 ./σ²)/2
end

"negative quasi log-likelihood for GARCH (α0,α1,β1) partials"
function qmle_grad(ϵ,θ)
    σ² = garch(ϵ,θ)
    return (1 .-ϵ.^2 ./σ²)./(2*σ²).*garch_grad(ϵ,θ)
end

"Newton's method minimizer"
function newtMin(h_obj::Function,h_grad::Function,h_hess::Function,u0::Vector
                    ;maxiter=100,tol=1.e-14,δ=1.e-4)
    u1 = u0
    h1 = h_obj(u1)
    if isnan(h1)
        throw(DomainError(u0,"invalid initial value"))
    end
    N = maxiter
    while N>0
        u0 = u1
        h0 = h1
        k = 0
        while N>0 && (k==0 || isnan(h1)
            || h1-h0>δ*dot(u1-u0,h_grad(u0)))
            u1 = u0-2.0^k*h_hess(u0)\h_grad(u0)
            h1 = h_obj(u1)
            k -= 1
            N -= 1
        end
        if abs(h1-h0)<tol
            return u1
        end
    end
    return u0
end

"BHHH solver for maximum likelihood estimates"
function bhhh(x::Vector,obj::Function,grad::Function,θ₀::Vector)
    h_obj = θ->mean(obj(x,θ))
    h_grad = θ->mean(grad(x,θ))
    h_hess = θ->cov(grad(x,θ))
    return newtMin(h_obj,h_grad,h_hess,θ₀)
end

"GARCH(1,1) fit"
function garch_fit(ϵ)
    return bhhh(ϵ,qmle_obj,qmle_grad,[var(ϵ)*(1-.7-.2),.2,.7])
end

export garch,garch_fit

end # module
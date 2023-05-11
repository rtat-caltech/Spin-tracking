using LinearAlgebra
import SparseArrays
import StaticArraysCore
import ArrayInterface
import Tricks: static_hasmethod
import Lazy: @forward
import Setfield: @set!

# overload
import Base: zero, one, oneunit
import Base: +, -, *, /, \, ∘, ==, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Matrix, Diagonal
import SparseArrays: sparse, issparse
import SciMLOperators: islinear, getops, isconstant, update_coefficients!

mutable struct RotOperator{T,AType<:AbstractMatrix{T},F} <: SciMLOperators.AbstractSciMLOperator{T}
    A::AType
    update_func::F
    RotOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where{AType} =
        new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

# constructors
Base.similar(L::RotOperator, ::Type{T}, dims::Dims) where{T} = RotOperator(similar(L.A, T, dims))

# traits
@forward RotOperator.A (
                           LinearAlgebra.issymmetric,
                           LinearAlgebra.ishermitian,
                           LinearAlgebra.isposdef,

                           issquare,
                           has_ldiv,
                           has_ldiv!,
                          )
islinear(::RotOperator) = true

Base.size(L::RotOperator) = size(L.A)
for op in (
           :adjoint,
           :transpose,
          )
    @eval function Base.$op(L::RotOperator)
        if isconstant(L)
            RotOperator($op(L.A))
        else
            update_func = (A,u,p,t) -> $op(L.update_func($op(L.A),u,p,t))
            RotOperator($op(L.A); update_func = update_func)
        end
    end
end
Base.conj(L::RotOperator) = RotOperator(
                                              conj(L.A);
                                              update_func= (A,u,b,t) -> conj(L.update_func(conj(L.A),u,p,t))
                                             )

has_adjoint(A::RotOperator) = has_adjoint(A.A)
update_coefficients!(L::RotOperator,u,p,t) = (L.A = L.update_func(L.A,u,p,t); nothing)

getops(L::RotOperator) = (L.A)
isconstant(L::RotOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(L::RotOperator) = iszero(L.A)

SparseArrays.sparse(L::RotOperator) = sparse(L.A)
SparseArrays.issparse(L::RotOperator) = issparse(L.A)

# TODO - add tests for RotOperator indexing
# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::RotOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::RotOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::RotOperator, v, I::Vararg{Int, N}) where{N} = (L.A[I...] = v)

Base.eachcol(L::RotOperator) = eachcol(L.A)
Base.eachrow(L::RotOperator) = eachrow(L.A)
Base.length(L::RotOperator) = length(L.A)
Base.iterate(L::RotOperator,args...) = iterate(L.A,args...)
Base.axes(L::RotOperator) = axes(L.A)
Base.eachindex(L::RotOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:RotOperator{T,AType}}) where{T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::RotOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::RotOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArraysCore.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::RotOperator) = L
Base.ndims(::Type{<:RotOperator{T,AType}}) where{T,AType} = ndims(AType)
ArrayInterface.issingular(L::RotOperator) = ArrayInterface.issingular(L.A)
Base.copy(L::RotOperator) = RotOperator(copy(L.A);update_func=L.update_func)

# operator application
Base.:*(L::RotOperator, u::AbstractVecOrMat) = L.A * u
Base.:\(L::RotOperator, u::AbstractVecOrMat) = L.A \ u
LinearAlgebra.mul!(v::AbstractVecOrMat, L::RotOperator, u::AbstractVecOrMat) = mul!(v, L.A, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, L::RotOperator, u::AbstractVecOrMat, a, b) = mul!(v, L.A, u, a, b)
LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::RotOperator, u::AbstractVecOrMat) = ldiv!(v, L.A, u)
LinearAlgebra.ldiv!(L::RotOperator, u::AbstractVecOrMat) = ldiv!(L.A, u)
using FlexibleFMM: child_offsets, child_slot, colleague_offsets, self_colleague_slot,
    m2l_table_offsets, m2l_table_index, is_far_offset, far_offsets, far_index_table,
    level_width, as_svectors

@testset "child offsets and slots (D=$D)" for D in (2, 3)
    offs = child_offsets(Val(D))
    @test length(offs) == 2^D
    @test allunique(offs)
    @test all(all(0 .<= o .<= 1) for o in offs)
    # coordinate 1 is the least significant bit
    @test offs[1] == zero(SVector{D,Int})
    @test offs[2][1] == 1 && all(offs[2][2:end] .== 0)
    c = SVector{D,Float64}(ntuple(_ -> 0.5, Val(D)))
    for (k, o) in enumerate(offs)
        # a point displaced from the center in the direction of child k lands in slot k
        x = c + 0.25 * (2 .* SVector{D,Float64}(o) .- 1)
        @test child_slot(x, c) == k
    end
    # equality goes to the lower child
    @test child_slot(c, c) == 1
    cn = SVector{D,Float64}(ntuple(d -> d == 1 ? nextfloat(c[1]) : c[d], Val(D)))
    @test child_slot(cn, c) == 2
    @test @allocated(child_slot(c, c)) == 0
end

@testset "colleague offsets (D=$D)" for D in (2, 3)
    offs = colleague_offsets(Val(D))
    @test length(offs) == 3^D
    @test allunique(offs)
    @test all(maximum(abs, o) <= 1 for o in offs)
    @test offs[self_colleague_slot(Val(D))] == zero(SVector{D,Int})
    @test offs[2][1] == 0 && offs[1][1] == -1     # coordinate 1 fastest
end

@testset "M2L offset table (D=$D)" for D in (2, 3)
    table = m2l_table_offsets(Val(D))
    @test length(table) == 7^D
    @test allunique(table)
    @test all(m2l_table_index(table[i]) == i for i in eachindex(table))
    far = far_offsets(Val(D))
    @test length(far) == 7^D - 3^D
    @test all(is_far_offset, far)
    @test !is_far_offset(zero(SVector{D,Int}))
    fit = far_index_table(Val(D))
    @test count(!=(0), fit) == length(far)
    @test all(fit[m2l_table_index(far[n])] == n for n in eachindex(far))
    @test all(fit[m2l_table_index(o)] == 0 for o in table if !is_far_offset(o))
end

@testset "level widths" begin
    @test level_width(1.0, 1) == 1.0
    @test level_width(1.0, 4) == 0.125
    @test level_width(0.3, 5) == 0.3 / 16
    @test level_width(3.0, 2) === 1.5
end

@testset "as_svectors" begin
    v = [P2(1, 2), P2(3, 4)]
    @test as_svectors(Val(2), v) === v
    M = [1.0 3.0; 2.0 4.0]
    @test as_svectors(Val(2), M) == v
    @test_throws DimensionMismatch as_svectors(Val(3), M)
end

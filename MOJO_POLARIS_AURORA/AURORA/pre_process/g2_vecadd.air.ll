; ModuleID = '02_vecadd_gpu.mojo'
source_filename = "02_vecadd_gpu.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_02_vecadd_gpu_vecadd_kernel_T6A6AoA6A6A_6556589aa203961e(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %3, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %5 = bitcast ptr addrspace(2) %3 to ptr addrspace(2)
  %.loaded = load i32, ptr addrspace(2) %5, align 4
  %6 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  %7 = load { ptr addrspace(1), { { {} }, { {} } } }, ptr addrspace(1) %6, align 8
  %8 = extractvalue { ptr addrspace(1), { { {} }, { {} } } } %7, 0
  %9 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %10 = load { ptr addrspace(1), { { {} }, { {} } } }, ptr addrspace(1) %9, align 8
  %11 = extractvalue { ptr addrspace(1), { { {} }, { {} } } } %10, 0
  %12 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %13 = load { ptr addrspace(1), { { {} }, { {} } } }, ptr addrspace(1) %12, align 8
  %14 = extractvalue { ptr addrspace(1), { { {} }, { {} } } } %13, 0
  %15 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %16 = zext i32 %15 to i64
  %17 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %18 = zext i32 %17 to i64
  %19 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %20 = sext i32 %19 to i64
  %21 = mul i64 %18, %20
  %22 = add i64 %21, %16
  %23 = sext i32 %.loaded to i64
  %24 = icmp slt i64 %22, %23
  br i1 %24, label %25, label %43

25:                                               ; preds = %4
  %26 = trunc i64 %22 to i32
  %27 = sext i32 %26 to i64
  %28 = bitcast ptr addrspace(1) %14 to ptr addrspace(1)
  %29 = getelementptr inbounds float, ptr addrspace(1) %28, i64 %27
  %30 = bitcast ptr addrspace(1) %29 to ptr addrspace(1)
  %31 = bitcast ptr addrspace(1) %30 to ptr addrspace(1)
  %32 = load float, ptr addrspace(1) %31, align 4
  %33 = bitcast ptr addrspace(1) %11 to ptr addrspace(1)
  %34 = getelementptr inbounds float, ptr addrspace(1) %33, i64 %27
  %35 = bitcast ptr addrspace(1) %34 to ptr addrspace(1)
  %36 = bitcast ptr addrspace(1) %35 to ptr addrspace(1)
  %37 = load float, ptr addrspace(1) %36, align 4
  %38 = fadd contract float %32, %37
  %39 = bitcast ptr addrspace(1) %8 to ptr addrspace(1)
  %40 = getelementptr inbounds float, ptr addrspace(1) %39, i64 %27
  %41 = bitcast ptr addrspace(1) %40 to ptr addrspace(1)
  %42 = bitcast ptr addrspace(1) %41 to ptr addrspace(1)
  store float %38, ptr addrspace(1) %42, align 4
  br label %44

43:                                               ; preds = %4
  br label %44

44:                                               ; preds = %25, %43
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "memory"="argmem: readwrite" "metal.kernel"="true" "metal.thread_args_added"="true" "metal.thread_index_in_simdgroup_idx"="8" "metal.thread_position_in_threadgroup_idx"="5" "metal.threadgroup_position_in_grid_idx"="4" "metal.threads_per_grid_idx"="7" "metal.threads_per_threadgroup_idx"="6" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!12, !13, !14}
!air.language_version = !{!15}
!air.source_file_name = !{!16}
!llvm.ident = !{!17}
!air.version = !{!18}
!llvm.module.flags = !{!19, !20, !21, !22, !23, !24, !25, !26, !27}

!0 = !{ptr @_02_vecadd_gpu_vecadd_kernel_T6A6AoA6A6A_6556589aa203961e, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!7 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threadgroup_position_in_grid", !"air.arg_unused"}
!8 = !{i32 5, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"thread_position_in_threadgroup", !"air.arg_unused"}
!9 = !{i32 6, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_threadgroup", !"air.arg_unused"}
!10 = !{i32 7, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_grid", !"air.arg_unused"}
!11 = !{i32 8, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_index_in_simdgroup"}
!12 = !{!"air.compile.denorms_disable"}
!13 = !{!"air.compile.fast_math_disable"}
!14 = !{!"air.compile.framebuffer_fetch_enable"}
!15 = !{!"Metal", i32 3, i32 2, i32 0}
!16 = !{!"02_vecadd_gpu.metal"}
!17 = !{!"Apple metal version 32023.620 (metalfe-32023.620)"}
!18 = !{i32 2, i32 7, i32 0}
!19 = !{i32 1, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!20 = !{i32 2, !"wchar_size", i32 4}
!21 = !{i32 4, !"frame-pointer", i32 2}
!22 = !{i32 7, !"air.max_device_buffers", i32 31}
!23 = !{i32 7, !"air.max_constant_buffers", i32 31}
!24 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!25 = !{i32 7, !"air.max_textures", i32 128}
!26 = !{i32 7, !"air.max_read_write_textures", i32 8}
!27 = !{i32 7, !"air.max_samplers", i32 16}

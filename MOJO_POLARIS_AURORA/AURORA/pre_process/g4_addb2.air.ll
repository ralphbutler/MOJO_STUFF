; ModuleID = '<split-module>'
source_filename = "04b_train_mlp_gpu.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_04b_train_mlp_gpu_add_bias1_ker6A6A_1facf35c98de617c(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %4 = bitcast ptr addrspace(2) %2 to ptr addrspace(2)
  %.loaded = load i32, ptr addrspace(2) %4, align 4
  %5 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %6 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %5, align 8
  %7 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %8 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %7, align 8
  %9 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %8, 0
  %10 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %6, 0
  %11 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %12 = zext i32 %11 to i64
  %13 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %14 = zext i32 %13 to i64
  %15 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %16 = sext i32 %15 to i64
  %17 = mul i64 %14, %16
  %18 = add i64 %17, %12
  %19 = trunc i64 %18 to i32
  %20 = sext i32 %19 to i64
  %21 = bitcast ptr addrspace(1) %9 to ptr addrspace(1)
  %22 = getelementptr inbounds float, ptr addrspace(1) %21, i64 %20
  %23 = bitcast ptr addrspace(1) %22 to ptr addrspace(1)
  %24 = sext i32 %.loaded to i64
  %25 = icmp slt i64 %18, %24
  br i1 %25, label %26, label %33

26:                                               ; preds = %3
  %27 = bitcast ptr addrspace(1) %23 to ptr addrspace(1)
  %28 = load float, ptr addrspace(1) %27, align 4
  %29 = bitcast ptr addrspace(1) %10 to ptr addrspace(1)
  %30 = load float, ptr addrspace(1) %29, align 4
  %31 = fadd contract float %28, %30
  %32 = bitcast ptr addrspace(1) %23 to ptr addrspace(1)
  store float %31, ptr addrspace(1) %32, align 4
  br label %34

33:                                               ; preds = %3
  br label %34

34:                                               ; preds = %33, %26
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "memory"="argmem: readwrite" "metal.kernel"="true" "metal.thread_args_added"="true" "metal.thread_index_in_simdgroup_idx"="7" "metal.thread_position_in_threadgroup_idx"="4" "metal.threadgroup_position_in_grid_idx"="3" "metal.threads_per_grid_idx"="6" "metal.threads_per_threadgroup_idx"="5" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!11, !12, !13}
!air.language_version = !{!14}
!air.source_file_name = !{!15}
!llvm.ident = !{!16}
!air.version = !{!17}
!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24, !25, !26}

!0 = !{ptr @_04b_train_mlp_gpu_add_bias1_ker6A6A_1facf35c98de617c, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!6 = !{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threadgroup_position_in_grid", !"air.arg_unused"}
!7 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"thread_position_in_threadgroup", !"air.arg_unused"}
!8 = !{i32 5, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_threadgroup", !"air.arg_unused"}
!9 = !{i32 6, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_grid", !"air.arg_unused"}
!10 = !{i32 7, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_index_in_simdgroup"}
!11 = !{!"air.compile.denorms_disable"}
!12 = !{!"air.compile.fast_math_disable"}
!13 = !{!"air.compile.framebuffer_fetch_enable"}
!14 = !{!"Metal", i32 3, i32 2, i32 0}
!15 = !{!"04b_train_mlp_gpu.metal"}
!16 = !{!"Apple metal version 32023.620 (metalfe-32023.620)"}
!17 = !{i32 2, i32 7, i32 0}
!18 = !{i32 1, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!19 = !{i32 2, !"wchar_size", i32 4}
!20 = !{i32 4, !"frame-pointer", i32 2}
!21 = !{i32 7, !"air.max_device_buffers", i32 31}
!22 = !{i32 7, !"air.max_constant_buffers", i32 31}
!23 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!24 = !{i32 7, !"air.max_textures", i32 128}
!25 = !{i32 7, !"air.max_read_write_textures", i32 8}
!26 = !{i32 7, !"air.max_samplers", i32 16}

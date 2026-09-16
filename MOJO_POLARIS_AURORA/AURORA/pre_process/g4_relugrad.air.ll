; ModuleID = '<split-module>'
source_filename = "04b_train_mlp_gpu.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i32 @air.threads_per_threadgroup.y()

declare i32 @air.threadgroup_position_in_grid.y()

declare i32 @air.thread_position_in_threadgroup.y()

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_04b_train_mlp_gpu_relu_grad_ker6A6A_389c38ee13190a49(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %3, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %4, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %6 = bitcast ptr addrspace(2) %3 to ptr addrspace(2)
  %.loaded = load i32, ptr addrspace(2) %6, align 4
  %7 = bitcast ptr addrspace(2) %4 to ptr addrspace(2)
  %.loaded7 = load i32, ptr addrspace(2) %7, align 4
  %8 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  %9 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %8, align 8
  %10 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %11 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %10, align 8
  %12 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %13 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %12, align 8
  %14 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %13, 0
  %15 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %11, 0
  %16 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %9, 0
  %17 = sext i32 %.loaded7 to i64
  %18 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 1
  %19 = zext i32 %18 to i64
  %20 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 1
  %21 = zext i32 %20 to i64
  %22 = extractelement <3 x i32> %threads_per_threadgroup, i64 1
  %23 = sext i32 %22 to i64
  %24 = mul i64 %21, %23
  %25 = add i64 %24, %19
  %26 = trunc i64 %25 to i32
  %27 = mul i32 %26, 16
  %28 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %29 = zext i32 %28 to i64
  %30 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %31 = zext i32 %30 to i64
  %32 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %33 = sext i32 %32 to i64
  %34 = mul i64 %31, %33
  %35 = add i64 %34, %29
  %36 = trunc i64 %35 to i32
  %37 = add i32 %27, %36
  %38 = sext i32 %37 to i64
  %39 = bitcast ptr addrspace(1) %15 to ptr addrspace(1)
  %40 = getelementptr inbounds float, ptr addrspace(1) %39, i64 %38
  %41 = bitcast ptr addrspace(1) %40 to ptr addrspace(1)
  %42 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  %43 = getelementptr inbounds float, ptr addrspace(1) %42, i64 %38
  %44 = bitcast ptr addrspace(1) %43 to ptr addrspace(1)
  %45 = bitcast ptr addrspace(1) %14 to ptr addrspace(1)
  %46 = getelementptr inbounds float, ptr addrspace(1) %45, i64 %38
  %47 = bitcast ptr addrspace(1) %46 to ptr addrspace(1)
  %48 = sext i32 %.loaded to i64
  %49 = icmp slt i64 %25, %48
  %50 = icmp slt i64 %35, %17
  %51 = select i1 %49, i1 %50, i1 false
  br i1 %51, label %52, label %63

52:                                               ; preds = %5
  %53 = bitcast ptr addrspace(1) %47 to ptr addrspace(1)
  %54 = load float, ptr addrspace(1) %53, align 4
  %55 = fcmp ogt float %54, 0.000000e+00
  br i1 %55, label %56, label %59

56:                                               ; preds = %52
  %57 = bitcast ptr addrspace(1) %41 to ptr addrspace(1)
  %58 = load float, ptr addrspace(1) %57, align 4
  br label %60

59:                                               ; preds = %52
  br label %60

60:                                               ; preds = %59, %56
  %61 = phi float [ 0.000000e+00, %59 ], [ %58, %56 ]
  %62 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  store float %61, ptr addrspace(1) %62, align 4
  br label %64

63:                                               ; preds = %5
  br label %64

64:                                               ; preds = %63, %60
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "memory"="argmem: readwrite" "metal.kernel"="true" "metal.thread_args_added"="true" "metal.thread_index_in_simdgroup_idx"="9" "metal.thread_position_in_threadgroup_idx"="6" "metal.threadgroup_position_in_grid_idx"="5" "metal.threads_per_grid_idx"="8" "metal.threads_per_threadgroup_idx"="7" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!13, !14, !15}
!air.language_version = !{!16}
!air.source_file_name = !{!17}
!llvm.ident = !{!18}
!air.version = !{!19}
!llvm.module.flags = !{!20, !21, !22, !23, !24, !25, !26, !27, !28}

!0 = !{ptr @_04b_train_mlp_gpu_relu_grad_ker6A6A_389c38ee13190a49, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!8 = !{i32 5, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threadgroup_position_in_grid", !"air.arg_unused"}
!9 = !{i32 6, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"thread_position_in_threadgroup", !"air.arg_unused"}
!10 = !{i32 7, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_threadgroup", !"air.arg_unused"}
!11 = !{i32 8, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_grid", !"air.arg_unused"}
!12 = !{i32 9, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_index_in_simdgroup"}
!13 = !{!"air.compile.denorms_disable"}
!14 = !{!"air.compile.fast_math_disable"}
!15 = !{!"air.compile.framebuffer_fetch_enable"}
!16 = !{!"Metal", i32 3, i32 2, i32 0}
!17 = !{!"04b_train_mlp_gpu.metal"}
!18 = !{!"Apple metal version 32023.620 (metalfe-32023.620)"}
!19 = !{i32 2, i32 7, i32 0}
!20 = !{i32 1, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!21 = !{i32 2, !"wchar_size", i32 4}
!22 = !{i32 4, !"frame-pointer", i32 2}
!23 = !{i32 7, !"air.max_device_buffers", i32 31}
!24 = !{i32 7, !"air.max_constant_buffers", i32 31}
!25 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!26 = !{i32 7, !"air.max_textures", i32 128}
!27 = !{i32 7, !"air.max_read_write_textures", i32 8}
!28 = !{i32 7, !"air.max_samplers", i32 16}

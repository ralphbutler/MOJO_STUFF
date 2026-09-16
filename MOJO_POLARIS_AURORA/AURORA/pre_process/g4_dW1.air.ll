; ModuleID = '<split-module>'
source_filename = "04b_train_mlp_gpu.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i64 @air.max.s.i64(i64, i64)

declare i32 @air.threads_per_threadgroup.y()

declare i32 @air.threadgroup_position_in_grid.y()

declare i32 @air.thread_position_in_threadgroup.y()

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_04b_train_mlp_gpu_matmul_at_b_k6A6A_a7e596135c3daf7d(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %3, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %4, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %5, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %7 = bitcast ptr addrspace(2) %3 to ptr addrspace(2)
  %.loaded = load i32, ptr addrspace(2) %7, align 4
  %8 = bitcast ptr addrspace(2) %4 to ptr addrspace(2)
  %.loaded7 = load i32, ptr addrspace(2) %8, align 4
  %9 = bitcast ptr addrspace(2) %5 to ptr addrspace(2)
  %.loaded10 = load i32, ptr addrspace(2) %9, align 4
  %10 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  %11 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %10, align 8
  %12 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %13 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %12, align 8
  %14 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %15 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %14, align 8
  %16 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %15, 0
  %17 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %13, 0
  %18 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %11, 0
  %19 = sext i32 %.loaded to i64
  %20 = sext i32 %.loaded10 to i64
  %21 = call i64 @air.max.s.i64(i64 %19, i64 0)
  %22 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 1
  %23 = zext i32 %22 to i64
  %24 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 1
  %25 = zext i32 %24 to i64
  %26 = extractelement <3 x i32> %threads_per_threadgroup, i64 1
  %27 = sext i32 %26 to i64
  %28 = mul i64 %25, %27
  %29 = add i64 %28, %23
  %30 = trunc i64 %29 to i32
  %31 = mul i32 %30, 16
  %32 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %33 = zext i32 %32 to i64
  %34 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %35 = zext i32 %34 to i64
  %36 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %37 = sext i32 %36 to i64
  %38 = mul i64 %35, %37
  %39 = add i64 %38, %33
  %40 = trunc i64 %39 to i32
  %41 = add i32 %31, %40
  %42 = sext i32 %41 to i64
  %43 = bitcast ptr addrspace(1) %18 to ptr addrspace(1)
  %44 = getelementptr inbounds float, ptr addrspace(1) %43, i64 %42
  %45 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  %46 = sext i32 %.loaded7 to i64
  %47 = icmp slt i64 %29, %46
  %48 = icmp slt i64 %39, %20
  %49 = select i1 %47, i1 %48, i1 false
  br i1 %49, label %50, label %93

50:                                               ; preds = %6
  br label %51

51:                                               ; preds = %68, %50
  %52 = phi float [ 0.000000e+00, %50 ], [ %89, %68 ]
  %53 = phi i64 [ %21, %50 ], [ %69, %68 ]
  %54 = sub i64 %21, %53
  %55 = sub i64 %53, 1
  %56 = icmp eq i64 %53, 0
  %57 = select i1 %56, i64 %53, i64 %55
  br label %58

58:                                               ; preds = %51
  br i1 %56, label %59, label %60

59:                                               ; preds = %58
  br label %62

60:                                               ; preds = %58
  br label %61

61:                                               ; preds = %60
  br label %65

62:                                               ; preds = %59
  %63 = phi i64 [ %57, %59 ]
  %64 = phi i64 [ %54, %59 ]
  br label %90

65:                                               ; preds = %61
  %66 = phi i64 [ %57, %61 ]
  %67 = phi i64 [ %54, %61 ]
  br label %68

68:                                               ; preds = %65
  %69 = phi i64 [ %66, %65 ]
  %70 = phi i64 [ %67, %65 ]
  %71 = trunc i64 %70 to i32
  %72 = mul i32 %71, 2
  %73 = add i32 %72, %30
  %74 = sext i32 %73 to i64
  %75 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  %76 = getelementptr inbounds float, ptr addrspace(1) %75, i64 %74
  %77 = bitcast ptr addrspace(1) %76 to ptr addrspace(1)
  %78 = bitcast ptr addrspace(1) %77 to ptr addrspace(1)
  %79 = load float, ptr addrspace(1) %78, align 4
  %80 = mul i32 %71, 16
  %81 = add i32 %80, %40
  %82 = sext i32 %81 to i64
  %83 = bitcast ptr addrspace(1) %17 to ptr addrspace(1)
  %84 = getelementptr inbounds float, ptr addrspace(1) %83, i64 %82
  %85 = bitcast ptr addrspace(1) %84 to ptr addrspace(1)
  %86 = bitcast ptr addrspace(1) %85 to ptr addrspace(1)
  %87 = load float, ptr addrspace(1) %86, align 4
  %88 = fmul contract float %79, %87
  %89 = fadd contract float %52, %88
  br label %51

90:                                               ; preds = %62
  %91 = phi float [ %52, %62 ]
  %92 = bitcast ptr addrspace(1) %45 to ptr addrspace(1)
  store float %91, ptr addrspace(1) %92, align 4
  br label %94

93:                                               ; preds = %6
  br label %94

94:                                               ; preds = %93, %90
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "memory"="argmem: readwrite" "metal.kernel"="true" "metal.thread_args_added"="true" "metal.thread_index_in_simdgroup_idx"="10" "metal.thread_position_in_threadgroup_idx"="7" "metal.threadgroup_position_in_grid_idx"="6" "metal.threads_per_grid_idx"="9" "metal.threads_per_threadgroup_idx"="8" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!14, !15, !16}
!air.language_version = !{!17}
!air.source_file_name = !{!18}
!llvm.ident = !{!19}
!air.version = !{!20}
!llvm.module.flags = !{!21, !22, !23, !24, !25, !26, !27, !28, !29}

!0 = !{ptr @_04b_train_mlp_gpu_matmul_at_b_k6A6A_a7e596135c3daf7d, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!8 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
!9 = !{i32 6, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threadgroup_position_in_grid", !"air.arg_unused"}
!10 = !{i32 7, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"thread_position_in_threadgroup", !"air.arg_unused"}
!11 = !{i32 8, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_threadgroup", !"air.arg_unused"}
!12 = !{i32 9, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_grid", !"air.arg_unused"}
!13 = !{i32 10, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_index_in_simdgroup"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_disable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Metal", i32 3, i32 2, i32 0}
!18 = !{!"04b_train_mlp_gpu.metal"}
!19 = !{!"Apple metal version 32023.620 (metalfe-32023.620)"}
!20 = !{i32 2, i32 7, i32 0}
!21 = !{i32 1, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!22 = !{i32 2, !"wchar_size", i32 4}
!23 = !{i32 4, !"frame-pointer", i32 2}
!24 = !{i32 7, !"air.max_device_buffers", i32 31}
!25 = !{i32 7, !"air.max_constant_buffers", i32 31}
!26 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!27 = !{i32 7, !"air.max_textures", i32 128}
!28 = !{i32 7, !"air.max_read_write_textures", i32 8}
!29 = !{i32 7, !"air.max_samplers", i32 16}

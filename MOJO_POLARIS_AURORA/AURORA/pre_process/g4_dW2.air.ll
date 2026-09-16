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
define void @_04b_train_mlp_gpu_matmul_at_b_k6A6A_e260f8de89d59a8a(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %3, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %4, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %5, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
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
  %31 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %32 = zext i32 %31 to i64
  %33 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %34 = zext i32 %33 to i64
  %35 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %36 = sext i32 %35 to i64
  %37 = mul i64 %34, %36
  %38 = add i64 %37, %32
  %39 = trunc i64 %38 to i32
  %40 = add i32 %30, %39
  %41 = sext i32 %40 to i64
  %42 = bitcast ptr addrspace(1) %18 to ptr addrspace(1)
  %43 = getelementptr inbounds float, ptr addrspace(1) %42, i64 %41
  %44 = bitcast ptr addrspace(1) %43 to ptr addrspace(1)
  %45 = sext i32 %.loaded7 to i64
  %46 = icmp slt i64 %29, %45
  %47 = icmp slt i64 %38, %20
  %48 = select i1 %46, i1 %47, i1 false
  br i1 %48, label %49, label %91

49:                                               ; preds = %6
  br label %50

50:                                               ; preds = %67, %49
  %51 = phi float [ 0.000000e+00, %49 ], [ %87, %67 ]
  %52 = phi i64 [ %21, %49 ], [ %68, %67 ]
  %53 = sub i64 %21, %52
  %54 = sub i64 %52, 1
  %55 = icmp eq i64 %52, 0
  %56 = select i1 %55, i64 %52, i64 %54
  br label %57

57:                                               ; preds = %50
  br i1 %55, label %58, label %59

58:                                               ; preds = %57
  br label %61

59:                                               ; preds = %57
  br label %60

60:                                               ; preds = %59
  br label %64

61:                                               ; preds = %58
  %62 = phi i64 [ %56, %58 ]
  %63 = phi i64 [ %53, %58 ]
  br label %88

64:                                               ; preds = %60
  %65 = phi i64 [ %56, %60 ]
  %66 = phi i64 [ %53, %60 ]
  br label %67

67:                                               ; preds = %64
  %68 = phi i64 [ %65, %64 ]
  %69 = phi i64 [ %66, %64 ]
  %70 = trunc i64 %69 to i32
  %71 = mul i32 %70, 16
  %72 = add i32 %71, %30
  %73 = sext i32 %72 to i64
  %74 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  %75 = getelementptr inbounds float, ptr addrspace(1) %74, i64 %73
  %76 = bitcast ptr addrspace(1) %75 to ptr addrspace(1)
  %77 = bitcast ptr addrspace(1) %76 to ptr addrspace(1)
  %78 = load float, ptr addrspace(1) %77, align 4
  %79 = add i32 %70, %39
  %80 = sext i32 %79 to i64
  %81 = bitcast ptr addrspace(1) %17 to ptr addrspace(1)
  %82 = getelementptr inbounds float, ptr addrspace(1) %81, i64 %80
  %83 = bitcast ptr addrspace(1) %82 to ptr addrspace(1)
  %84 = bitcast ptr addrspace(1) %83 to ptr addrspace(1)
  %85 = load float, ptr addrspace(1) %84, align 4
  %86 = fmul contract float %78, %85
  %87 = fadd contract float %51, %86
  br label %50

88:                                               ; preds = %61
  %89 = phi float [ %51, %61 ]
  %90 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  store float %89, ptr addrspace(1) %90, align 4
  br label %92

91:                                               ; preds = %6
  br label %92

92:                                               ; preds = %91, %88
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

!0 = !{ptr @_04b_train_mlp_gpu_matmul_at_b_k6A6A_e260f8de89d59a8a, !1, !2}
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

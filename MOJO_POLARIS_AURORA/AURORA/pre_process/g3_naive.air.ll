; ModuleID = '03a_matmul_naive.mojo'
source_filename = "03a_matmul_naive.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

declare i32 @air.threads_per_threadgroup.y()

declare i32 @air.threadgroup_position_in_grid.y()

declare i32 @air.thread_position_in_threadgroup.y()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_03a_matmul_naive_matmul_kernel6A6AoA_3f1188b4f1582b98(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %4 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  %5 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %4, align 8
  %6 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %5, 0
  %7 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %8 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %7, align 8
  %9 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %8, 0
  %10 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %11 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %10, align 8
  %12 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %11, 0
  %13 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 1
  %14 = zext i32 %13 to i64
  %15 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 1
  %16 = zext i32 %15 to i64
  %17 = extractelement <3 x i32> %threads_per_threadgroup, i64 1
  %18 = sext i32 %17 to i64
  %19 = mul i64 %16, %18
  %20 = add i64 %19, %14
  %21 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %22 = zext i32 %21 to i64
  %23 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %24 = zext i32 %23 to i64
  %25 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %26 = sext i32 %25 to i64
  %27 = mul i64 %24, %26
  %28 = add i64 %27, %22
  %29 = icmp slt i64 %20, 2048
  %30 = icmp slt i64 %28, 2048
  %31 = select i1 %29, i1 %30, i1 false
  br i1 %31, label %32, label %85

32:                                               ; preds = %3
  br label %33

33:                                               ; preds = %50, %32
  %34 = phi float [ 0.000000e+00, %32 ], [ %73, %50 ]
  %35 = phi i64 [ 2048, %32 ], [ %51, %50 ]
  %36 = sub i64 2048, %35
  %37 = sub i64 %35, 1
  %38 = icmp eq i64 %35, 0
  br label %39

39:                                               ; preds = %33
  %40 = select i1 %38, i64 %35, i64 %37
  br i1 %38, label %41, label %42

41:                                               ; preds = %39
  br label %44

42:                                               ; preds = %39
  br label %43

43:                                               ; preds = %42
  br label %47

44:                                               ; preds = %41
  %45 = phi i64 [ %40, %41 ]
  %46 = phi i64 [ %36, %41 ]
  br label %74

47:                                               ; preds = %43
  %48 = phi i64 [ %40, %43 ]
  %49 = phi i64 [ %36, %43 ]
  br label %50

50:                                               ; preds = %47
  %51 = phi i64 [ %48, %47 ]
  %52 = phi i64 [ %49, %47 ]
  %53 = trunc i64 %20 to i32
  %54 = mul i32 %53, 2048
  %55 = trunc i64 %52 to i32
  %56 = add i32 %54, %55
  %57 = sext i32 %56 to i64
  %58 = bitcast ptr addrspace(1) %12 to ptr addrspace(1)
  %59 = getelementptr inbounds float, ptr addrspace(1) %58, i64 %57
  %60 = bitcast ptr addrspace(1) %59 to ptr addrspace(1)
  %61 = bitcast ptr addrspace(1) %60 to ptr addrspace(1)
  %62 = load float, ptr addrspace(1) %61, align 4
  %63 = mul i32 %55, 2048
  %64 = trunc i64 %28 to i32
  %65 = add i32 %63, %64
  %66 = sext i32 %65 to i64
  %67 = bitcast ptr addrspace(1) %9 to ptr addrspace(1)
  %68 = getelementptr inbounds float, ptr addrspace(1) %67, i64 %66
  %69 = bitcast ptr addrspace(1) %68 to ptr addrspace(1)
  %70 = bitcast ptr addrspace(1) %69 to ptr addrspace(1)
  %71 = load float, ptr addrspace(1) %70, align 4
  %72 = fmul contract float %62, %71
  %73 = fadd contract float %34, %72
  br label %33

74:                                               ; preds = %44
  %75 = phi float [ %34, %44 ]
  %76 = trunc i64 %20 to i32
  %77 = trunc i64 %28 to i32
  %78 = mul i32 %76, 2048
  %79 = add i32 %78, %77
  %80 = sext i32 %79 to i64
  %81 = bitcast ptr addrspace(1) %6 to ptr addrspace(1)
  %82 = getelementptr inbounds float, ptr addrspace(1) %81, i64 %80
  %83 = bitcast ptr addrspace(1) %82 to ptr addrspace(1)
  %84 = bitcast ptr addrspace(1) %83 to ptr addrspace(1)
  store float %75, ptr addrspace(1) %84, align 4
  br label %86

85:                                               ; preds = %3
  br label %86

86:                                               ; preds = %74, %85
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

!0 = !{ptr @_03a_matmul_naive_matmul_kernel6A6AoA_3f1188b4f1582b98, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!6 = !{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threadgroup_position_in_grid", !"air.arg_unused"}
!7 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"thread_position_in_threadgroup", !"air.arg_unused"}
!8 = !{i32 5, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_threadgroup", !"air.arg_unused"}
!9 = !{i32 6, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads_per_grid", !"air.arg_unused"}
!10 = !{i32 7, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_index_in_simdgroup"}
!11 = !{!"air.compile.denorms_disable"}
!12 = !{!"air.compile.fast_math_disable"}
!13 = !{!"air.compile.framebuffer_fetch_enable"}
!14 = !{!"Metal", i32 3, i32 2, i32 0}
!15 = !{!"03a_matmul_naive.metal"}
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

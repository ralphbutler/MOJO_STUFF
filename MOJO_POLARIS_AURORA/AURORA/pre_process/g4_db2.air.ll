; ModuleID = '<split-module>'
source_filename = "04b_train_mlp_gpu.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

declare i64 @air.max.s.i64(i64, i64)

declare i32 @air.threads_per_threadgroup.x()

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @_04b_train_mlp_gpu_colsum_kernel6A6A_c5034b8eeae550b8(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, ptr addrspace(2) noundef "air-buffer-no-alias" "nocapture" "noundef" %3, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #0 {
  %5 = bitcast ptr addrspace(2) %2 to ptr addrspace(2)
  %.loaded = load i32, ptr addrspace(2) %5, align 4
  %6 = bitcast ptr addrspace(2) %3 to ptr addrspace(2)
  %.loaded5 = load i32, ptr addrspace(2) %6, align 4
  %7 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %8 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %7, align 8
  %9 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %10 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %9, align 8
  %11 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %10, 0
  %12 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %8, 0
  %13 = sext i32 %.loaded to i64
  %14 = call i64 @air.max.s.i64(i64 %13, i64 0)
  %15 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %16 = zext i32 %15 to i64
  %17 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %18 = zext i32 %17 to i64
  %19 = extractelement <3 x i32> %threads_per_threadgroup, i64 0
  %20 = sext i32 %19 to i64
  %21 = mul i64 %18, %20
  %22 = add i64 %21, %16
  %23 = trunc i64 %22 to i32
  %24 = sext i32 %23 to i64
  %25 = bitcast ptr addrspace(1) %12 to ptr addrspace(1)
  %26 = getelementptr inbounds float, ptr addrspace(1) %25, i64 %24
  %27 = bitcast ptr addrspace(1) %26 to ptr addrspace(1)
  %28 = sext i32 %.loaded5 to i64
  %29 = icmp slt i64 %22, %28
  br i1 %29, label %30, label %63

30:                                               ; preds = %4
  br label %31

31:                                               ; preds = %48, %30
  %32 = phi float [ 0.000000e+00, %30 ], [ %59, %48 ]
  %33 = phi i64 [ %14, %30 ], [ %49, %48 ]
  %34 = sub i64 %14, %33
  %35 = sub i64 %33, 1
  %36 = icmp eq i64 %33, 0
  %37 = select i1 %36, i64 %33, i64 %35
  br label %38

38:                                               ; preds = %31
  br i1 %36, label %39, label %40

39:                                               ; preds = %38
  br label %42

40:                                               ; preds = %38
  br label %41

41:                                               ; preds = %40
  br label %45

42:                                               ; preds = %39
  %43 = phi i64 [ %37, %39 ]
  %44 = phi i64 [ %34, %39 ]
  br label %60

45:                                               ; preds = %41
  %46 = phi i64 [ %37, %41 ]
  %47 = phi i64 [ %34, %41 ]
  br label %48

48:                                               ; preds = %45
  %49 = phi i64 [ %46, %45 ]
  %50 = phi i64 [ %47, %45 ]
  %51 = trunc i64 %50 to i32
  %52 = add i32 %51, %23
  %53 = sext i32 %52 to i64
  %54 = bitcast ptr addrspace(1) %11 to ptr addrspace(1)
  %55 = getelementptr inbounds float, ptr addrspace(1) %54, i64 %53
  %56 = bitcast ptr addrspace(1) %55 to ptr addrspace(1)
  %57 = bitcast ptr addrspace(1) %56 to ptr addrspace(1)
  %58 = load float, ptr addrspace(1) %57, align 4
  %59 = fadd contract float %32, %58
  br label %31

60:                                               ; preds = %42
  %61 = phi float [ %32, %42 ]
  %62 = bitcast ptr addrspace(1) %27 to ptr addrspace(1)
  store float %61, ptr addrspace(1) %62, align 4
  br label %64

63:                                               ; preds = %4
  br label %64

64:                                               ; preds = %63, %60
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

!0 = !{ptr @_04b_train_mlp_gpu_colsum_kernel6A6A_c5034b8eeae550b8, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"int"}
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
!16 = !{!"04b_train_mlp_gpu.metal"}
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

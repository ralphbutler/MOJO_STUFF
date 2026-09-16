; ModuleID = '03b_matmul_tiled.mojo'
source_filename = "03b_matmul_tiled.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx15.0.0"

@_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem = internal addrspace(3) global [256 x float] undef, align 4
@_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem = internal addrspace(3) global [256 x float] undef, align 4

; Function Attrs: convergent
declare void @air.wg.barrier(i32, i32) #0

declare i32 @air.threadgroup_position_in_grid.x()

declare i32 @air.threadgroup_position_in_grid.y()

declare i32 @air.thread_position_in_threadgroup.y()

declare i32 @air.thread_position_in_threadgroup.x()

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn
define void @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442(ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %0, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "readonly" %1, ptr addrspace(1) noundef nonnull "air-buffer-no-alias" "nocapture" "noundef" "writeonly" %2, <3 x i32> %threadgroup_position_in_grid, <3 x i32> %thread_position_in_threadgroup, <3 x i32> %threads_per_threadgroup, <3 x i32> %threads_per_grid, i32 %thread_index_in_simdgroup) local_unnamed_addr #1 {
  %4 = alloca { i64 }, i64 1, align 8
  %5 = bitcast ptr %4 to ptr
  %6 = alloca { i64 }, i64 1, align 8
  %7 = bitcast ptr %6 to ptr
  %8 = alloca { i64 }, i64 1, align 8
  %9 = bitcast ptr %8 to ptr
  %10 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  %11 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %10, align 8
  %12 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %11, 0
  %13 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %14 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %13, align 8
  %15 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %14, 0
  %16 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %17 = load { ptr addrspace(1), { { {}, {} }, { {}, {} } } }, ptr addrspace(1) %16, align 8
  %18 = extractvalue { ptr addrspace(1), { { {}, {} }, { {}, {} } } } %17, 0
  %19 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 0
  %20 = zext i32 %19 to i64
  %21 = extractelement <3 x i32> %thread_position_in_threadgroup, i64 1
  %22 = zext i32 %21 to i64
  %23 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 1
  %24 = zext i32 %23 to i64
  %25 = mul i64 %24, 16
  %26 = add i64 %25, %22
  %27 = extractelement <3 x i32> %threadgroup_position_in_grid, i64 0
  %28 = zext i32 %27 to i64
  %29 = mul i64 %28, 16
  %30 = add i64 %29, %20
  br label %31

31:                                               ; preds = %145, %3
  %32 = phi float [ 0.000000e+00, %3 ], [ %434, %145 ]
  %33 = phi i64 [ 0, %3 ], [ %67, %145 ]
  %34 = bitcast ptr %9 to ptr
  store i64 %33, ptr %34, align 8
  %35 = bitcast ptr %9 to ptr
  %36 = load { i64 }, ptr %35, align 8
  %37 = insertvalue { { i64 }, i8 } undef, { i64 } %36, 0
  %38 = insertvalue { { i64 }, i8 } %37, i8 1, 1
  %39 = add i64 %33, 16
  %40 = icmp sge i64 %33, 2048
  br label %41

41:                                               ; preds = %31
  %42 = select i1 %40, { { i64 }, i8 } zeroinitializer, { { i64 }, i8 } %38
  %43 = select i1 %40, i64 %33, i64 %39
  %44 = extractvalue { { i64 }, i8 } %42, 0
  %45 = bitcast ptr %7 to ptr
  store { i64 } %44, ptr %45, align 8
  %46 = bitcast ptr %7 to ptr
  %47 = load i64, ptr %46, align 8
  %48 = bitcast ptr %5 to ptr
  store { i64 } %44, ptr %48, align 8
  %49 = bitcast ptr %5 to ptr
  %50 = load {}, ptr %49, align 1
  %51 = extractvalue { { i64 }, i8 } %42, 1
  %52 = icmp eq i8 %51, 0
  br i1 %52, label %53, label %54

53:                                               ; preds = %41
  br label %56

54:                                               ; preds = %41
  br label %55

55:                                               ; preds = %54
  br label %60

56:                                               ; preds = %53
  %57 = phi i64 [ %47, %53 ]
  %58 = phi {} [ %50, %53 ]
  %59 = phi i64 [ %43, %53 ]
  br label %435

60:                                               ; preds = %55
  %61 = phi i64 [ %47, %55 ]
  %62 = phi {} [ %50, %55 ]
  %63 = phi i64 [ %43, %55 ]
  br label %64

64:                                               ; preds = %60
  %65 = phi i64 [ %61, %60 ]
  %66 = phi {} [ %62, %60 ]
  %67 = phi i64 [ %63, %60 ]
  %68 = icmp slt i64 %26, 2048
  br i1 %68, label %69, label %72

69:                                               ; preds = %64
  %70 = add i64 %65, %20
  %71 = icmp slt i64 %70, 2048
  br label %73

72:                                               ; preds = %64
  br label %73

73:                                               ; preds = %69, %72
  %74 = phi i1 [ false, %72 ], [ %71, %69 ]
  br i1 %74, label %75, label %97

75:                                               ; preds = %73
  %76 = add i64 %65, %20
  %77 = trunc i64 %26 to i32
  %78 = mul i32 %77, 2048
  %79 = trunc i64 %76 to i32
  %80 = add i32 %78, %79
  %81 = sext i32 %80 to i64
  %82 = bitcast ptr addrspace(1) %18 to ptr addrspace(1)
  %83 = getelementptr inbounds float, ptr addrspace(1) %82, i64 %81
  %84 = bitcast ptr addrspace(1) %83 to ptr addrspace(1)
  %85 = bitcast ptr addrspace(1) %84 to ptr addrspace(1)
  %86 = load float, ptr addrspace(1) %85, align 4
  %87 = trunc i64 %22 to i32
  %88 = trunc i64 %20 to i32
  %89 = mul i32 %87, 16
  %90 = add i32 %89, %88
  %91 = sext i32 %90 to i64
  %92 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %93 = bitcast ptr addrspace(3) %92 to ptr addrspace(3)
  %94 = getelementptr inbounds float, ptr addrspace(3) %93, i64 %91
  %95 = bitcast ptr addrspace(3) %94 to ptr addrspace(3)
  %96 = bitcast ptr addrspace(3) %95 to ptr addrspace(3)
  store float %86, ptr addrspace(3) %96, align 4
  br label %108

97:                                               ; preds = %73
  %98 = trunc i64 %22 to i32
  %99 = trunc i64 %20 to i32
  %100 = mul i32 %98, 16
  %101 = add i32 %100, %99
  %102 = sext i32 %101 to i64
  %103 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %104 = bitcast ptr addrspace(3) %103 to ptr addrspace(3)
  %105 = getelementptr inbounds float, ptr addrspace(3) %104, i64 %102
  %106 = bitcast ptr addrspace(3) %105 to ptr addrspace(3)
  %107 = bitcast ptr addrspace(3) %106 to ptr addrspace(3)
  store float 0.000000e+00, ptr addrspace(3) %107, align 4
  br label %108

108:                                              ; preds = %75, %97
  %109 = add i64 %65, %22
  %110 = icmp slt i64 %109, 2048
  %111 = icmp slt i64 %30, 2048
  %112 = select i1 %110, i1 %111, i1 false
  br i1 %112, label %113, label %134

113:                                              ; preds = %108
  %114 = trunc i64 %109 to i32
  %115 = mul i32 %114, 2048
  %116 = trunc i64 %30 to i32
  %117 = add i32 %115, %116
  %118 = sext i32 %117 to i64
  %119 = bitcast ptr addrspace(1) %15 to ptr addrspace(1)
  %120 = getelementptr inbounds float, ptr addrspace(1) %119, i64 %118
  %121 = bitcast ptr addrspace(1) %120 to ptr addrspace(1)
  %122 = bitcast ptr addrspace(1) %121 to ptr addrspace(1)
  %123 = load float, ptr addrspace(1) %122, align 4
  %124 = trunc i64 %22 to i32
  %125 = trunc i64 %20 to i32
  %126 = mul i32 %124, 16
  %127 = add i32 %126, %125
  %128 = sext i32 %127 to i64
  %129 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %130 = bitcast ptr addrspace(3) %129 to ptr addrspace(3)
  %131 = getelementptr inbounds float, ptr addrspace(3) %130, i64 %128
  %132 = bitcast ptr addrspace(3) %131 to ptr addrspace(3)
  %133 = bitcast ptr addrspace(3) %132 to ptr addrspace(3)
  store float %123, ptr addrspace(3) %133, align 4
  br label %145

134:                                              ; preds = %108
  %135 = trunc i64 %22 to i32
  %136 = trunc i64 %20 to i32
  %137 = mul i32 %135, 16
  %138 = add i32 %137, %136
  %139 = sext i32 %138 to i64
  %140 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %141 = bitcast ptr addrspace(3) %140 to ptr addrspace(3)
  %142 = getelementptr inbounds float, ptr addrspace(3) %141, i64 %139
  %143 = bitcast ptr addrspace(3) %142 to ptr addrspace(3)
  %144 = bitcast ptr addrspace(3) %143 to ptr addrspace(3)
  store float 0.000000e+00, ptr addrspace(3) %144, align 4
  br label %145

145:                                              ; preds = %113, %134
  call void @air.wg.barrier(i32 2, i32 1)
  %146 = trunc i64 %22 to i32
  %147 = mul i32 %146, 16
  %148 = sext i32 %147 to i64
  %149 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %150 = bitcast ptr addrspace(3) %149 to ptr addrspace(3)
  %151 = getelementptr inbounds float, ptr addrspace(3) %150, i64 %148
  %152 = bitcast ptr addrspace(3) %151 to ptr addrspace(3)
  %153 = bitcast ptr addrspace(3) %152 to ptr addrspace(3)
  %154 = load float, ptr addrspace(3) %153, align 4
  %155 = trunc i64 %20 to i32
  %156 = sext i32 %155 to i64
  %157 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %158 = bitcast ptr addrspace(3) %157 to ptr addrspace(3)
  %159 = getelementptr inbounds float, ptr addrspace(3) %158, i64 %156
  %160 = bitcast ptr addrspace(3) %159 to ptr addrspace(3)
  %161 = bitcast ptr addrspace(3) %160 to ptr addrspace(3)
  %162 = load float, ptr addrspace(3) %161, align 4
  %163 = fmul contract float %154, %162
  %164 = fadd contract float %32, %163
  %165 = add i32 %147, 1
  %166 = sext i32 %165 to i64
  %167 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %168 = bitcast ptr addrspace(3) %167 to ptr addrspace(3)
  %169 = getelementptr inbounds float, ptr addrspace(3) %168, i64 %166
  %170 = bitcast ptr addrspace(3) %169 to ptr addrspace(3)
  %171 = bitcast ptr addrspace(3) %170 to ptr addrspace(3)
  %172 = load float, ptr addrspace(3) %171, align 4
  %173 = add i32 %155, 16
  %174 = sext i32 %173 to i64
  %175 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %176 = bitcast ptr addrspace(3) %175 to ptr addrspace(3)
  %177 = getelementptr inbounds float, ptr addrspace(3) %176, i64 %174
  %178 = bitcast ptr addrspace(3) %177 to ptr addrspace(3)
  %179 = bitcast ptr addrspace(3) %178 to ptr addrspace(3)
  %180 = load float, ptr addrspace(3) %179, align 4
  %181 = fmul contract float %172, %180
  %182 = fadd contract float %164, %181
  %183 = add i32 %147, 2
  %184 = sext i32 %183 to i64
  %185 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %186 = bitcast ptr addrspace(3) %185 to ptr addrspace(3)
  %187 = getelementptr inbounds float, ptr addrspace(3) %186, i64 %184
  %188 = bitcast ptr addrspace(3) %187 to ptr addrspace(3)
  %189 = bitcast ptr addrspace(3) %188 to ptr addrspace(3)
  %190 = load float, ptr addrspace(3) %189, align 4
  %191 = add i32 %155, 32
  %192 = sext i32 %191 to i64
  %193 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %194 = bitcast ptr addrspace(3) %193 to ptr addrspace(3)
  %195 = getelementptr inbounds float, ptr addrspace(3) %194, i64 %192
  %196 = bitcast ptr addrspace(3) %195 to ptr addrspace(3)
  %197 = bitcast ptr addrspace(3) %196 to ptr addrspace(3)
  %198 = load float, ptr addrspace(3) %197, align 4
  %199 = fmul contract float %190, %198
  %200 = fadd contract float %182, %199
  %201 = add i32 %147, 3
  %202 = sext i32 %201 to i64
  %203 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %204 = bitcast ptr addrspace(3) %203 to ptr addrspace(3)
  %205 = getelementptr inbounds float, ptr addrspace(3) %204, i64 %202
  %206 = bitcast ptr addrspace(3) %205 to ptr addrspace(3)
  %207 = bitcast ptr addrspace(3) %206 to ptr addrspace(3)
  %208 = load float, ptr addrspace(3) %207, align 4
  %209 = add i32 %155, 48
  %210 = sext i32 %209 to i64
  %211 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %212 = bitcast ptr addrspace(3) %211 to ptr addrspace(3)
  %213 = getelementptr inbounds float, ptr addrspace(3) %212, i64 %210
  %214 = bitcast ptr addrspace(3) %213 to ptr addrspace(3)
  %215 = bitcast ptr addrspace(3) %214 to ptr addrspace(3)
  %216 = load float, ptr addrspace(3) %215, align 4
  %217 = fmul contract float %208, %216
  %218 = fadd contract float %200, %217
  %219 = add i32 %147, 4
  %220 = sext i32 %219 to i64
  %221 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %222 = bitcast ptr addrspace(3) %221 to ptr addrspace(3)
  %223 = getelementptr inbounds float, ptr addrspace(3) %222, i64 %220
  %224 = bitcast ptr addrspace(3) %223 to ptr addrspace(3)
  %225 = bitcast ptr addrspace(3) %224 to ptr addrspace(3)
  %226 = load float, ptr addrspace(3) %225, align 4
  %227 = add i32 %155, 64
  %228 = sext i32 %227 to i64
  %229 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %230 = bitcast ptr addrspace(3) %229 to ptr addrspace(3)
  %231 = getelementptr inbounds float, ptr addrspace(3) %230, i64 %228
  %232 = bitcast ptr addrspace(3) %231 to ptr addrspace(3)
  %233 = bitcast ptr addrspace(3) %232 to ptr addrspace(3)
  %234 = load float, ptr addrspace(3) %233, align 4
  %235 = fmul contract float %226, %234
  %236 = fadd contract float %218, %235
  %237 = add i32 %147, 5
  %238 = sext i32 %237 to i64
  %239 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %240 = bitcast ptr addrspace(3) %239 to ptr addrspace(3)
  %241 = getelementptr inbounds float, ptr addrspace(3) %240, i64 %238
  %242 = bitcast ptr addrspace(3) %241 to ptr addrspace(3)
  %243 = bitcast ptr addrspace(3) %242 to ptr addrspace(3)
  %244 = load float, ptr addrspace(3) %243, align 4
  %245 = add i32 %155, 80
  %246 = sext i32 %245 to i64
  %247 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %248 = bitcast ptr addrspace(3) %247 to ptr addrspace(3)
  %249 = getelementptr inbounds float, ptr addrspace(3) %248, i64 %246
  %250 = bitcast ptr addrspace(3) %249 to ptr addrspace(3)
  %251 = bitcast ptr addrspace(3) %250 to ptr addrspace(3)
  %252 = load float, ptr addrspace(3) %251, align 4
  %253 = fmul contract float %244, %252
  %254 = fadd contract float %236, %253
  %255 = add i32 %147, 6
  %256 = sext i32 %255 to i64
  %257 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %258 = bitcast ptr addrspace(3) %257 to ptr addrspace(3)
  %259 = getelementptr inbounds float, ptr addrspace(3) %258, i64 %256
  %260 = bitcast ptr addrspace(3) %259 to ptr addrspace(3)
  %261 = bitcast ptr addrspace(3) %260 to ptr addrspace(3)
  %262 = load float, ptr addrspace(3) %261, align 4
  %263 = add i32 %155, 96
  %264 = sext i32 %263 to i64
  %265 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %266 = bitcast ptr addrspace(3) %265 to ptr addrspace(3)
  %267 = getelementptr inbounds float, ptr addrspace(3) %266, i64 %264
  %268 = bitcast ptr addrspace(3) %267 to ptr addrspace(3)
  %269 = bitcast ptr addrspace(3) %268 to ptr addrspace(3)
  %270 = load float, ptr addrspace(3) %269, align 4
  %271 = fmul contract float %262, %270
  %272 = fadd contract float %254, %271
  %273 = add i32 %147, 7
  %274 = sext i32 %273 to i64
  %275 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %276 = bitcast ptr addrspace(3) %275 to ptr addrspace(3)
  %277 = getelementptr inbounds float, ptr addrspace(3) %276, i64 %274
  %278 = bitcast ptr addrspace(3) %277 to ptr addrspace(3)
  %279 = bitcast ptr addrspace(3) %278 to ptr addrspace(3)
  %280 = load float, ptr addrspace(3) %279, align 4
  %281 = add i32 %155, 112
  %282 = sext i32 %281 to i64
  %283 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %284 = bitcast ptr addrspace(3) %283 to ptr addrspace(3)
  %285 = getelementptr inbounds float, ptr addrspace(3) %284, i64 %282
  %286 = bitcast ptr addrspace(3) %285 to ptr addrspace(3)
  %287 = bitcast ptr addrspace(3) %286 to ptr addrspace(3)
  %288 = load float, ptr addrspace(3) %287, align 4
  %289 = fmul contract float %280, %288
  %290 = fadd contract float %272, %289
  %291 = add i32 %147, 8
  %292 = sext i32 %291 to i64
  %293 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %294 = bitcast ptr addrspace(3) %293 to ptr addrspace(3)
  %295 = getelementptr inbounds float, ptr addrspace(3) %294, i64 %292
  %296 = bitcast ptr addrspace(3) %295 to ptr addrspace(3)
  %297 = bitcast ptr addrspace(3) %296 to ptr addrspace(3)
  %298 = load float, ptr addrspace(3) %297, align 4
  %299 = add i32 %155, 128
  %300 = sext i32 %299 to i64
  %301 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %302 = bitcast ptr addrspace(3) %301 to ptr addrspace(3)
  %303 = getelementptr inbounds float, ptr addrspace(3) %302, i64 %300
  %304 = bitcast ptr addrspace(3) %303 to ptr addrspace(3)
  %305 = bitcast ptr addrspace(3) %304 to ptr addrspace(3)
  %306 = load float, ptr addrspace(3) %305, align 4
  %307 = fmul contract float %298, %306
  %308 = fadd contract float %290, %307
  %309 = add i32 %147, 9
  %310 = sext i32 %309 to i64
  %311 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %312 = bitcast ptr addrspace(3) %311 to ptr addrspace(3)
  %313 = getelementptr inbounds float, ptr addrspace(3) %312, i64 %310
  %314 = bitcast ptr addrspace(3) %313 to ptr addrspace(3)
  %315 = bitcast ptr addrspace(3) %314 to ptr addrspace(3)
  %316 = load float, ptr addrspace(3) %315, align 4
  %317 = add i32 %155, 144
  %318 = sext i32 %317 to i64
  %319 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %320 = bitcast ptr addrspace(3) %319 to ptr addrspace(3)
  %321 = getelementptr inbounds float, ptr addrspace(3) %320, i64 %318
  %322 = bitcast ptr addrspace(3) %321 to ptr addrspace(3)
  %323 = bitcast ptr addrspace(3) %322 to ptr addrspace(3)
  %324 = load float, ptr addrspace(3) %323, align 4
  %325 = fmul contract float %316, %324
  %326 = fadd contract float %308, %325
  %327 = add i32 %147, 10
  %328 = sext i32 %327 to i64
  %329 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %330 = bitcast ptr addrspace(3) %329 to ptr addrspace(3)
  %331 = getelementptr inbounds float, ptr addrspace(3) %330, i64 %328
  %332 = bitcast ptr addrspace(3) %331 to ptr addrspace(3)
  %333 = bitcast ptr addrspace(3) %332 to ptr addrspace(3)
  %334 = load float, ptr addrspace(3) %333, align 4
  %335 = add i32 %155, 160
  %336 = sext i32 %335 to i64
  %337 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %338 = bitcast ptr addrspace(3) %337 to ptr addrspace(3)
  %339 = getelementptr inbounds float, ptr addrspace(3) %338, i64 %336
  %340 = bitcast ptr addrspace(3) %339 to ptr addrspace(3)
  %341 = bitcast ptr addrspace(3) %340 to ptr addrspace(3)
  %342 = load float, ptr addrspace(3) %341, align 4
  %343 = fmul contract float %334, %342
  %344 = fadd contract float %326, %343
  %345 = add i32 %147, 11
  %346 = sext i32 %345 to i64
  %347 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %348 = bitcast ptr addrspace(3) %347 to ptr addrspace(3)
  %349 = getelementptr inbounds float, ptr addrspace(3) %348, i64 %346
  %350 = bitcast ptr addrspace(3) %349 to ptr addrspace(3)
  %351 = bitcast ptr addrspace(3) %350 to ptr addrspace(3)
  %352 = load float, ptr addrspace(3) %351, align 4
  %353 = add i32 %155, 176
  %354 = sext i32 %353 to i64
  %355 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %356 = bitcast ptr addrspace(3) %355 to ptr addrspace(3)
  %357 = getelementptr inbounds float, ptr addrspace(3) %356, i64 %354
  %358 = bitcast ptr addrspace(3) %357 to ptr addrspace(3)
  %359 = bitcast ptr addrspace(3) %358 to ptr addrspace(3)
  %360 = load float, ptr addrspace(3) %359, align 4
  %361 = fmul contract float %352, %360
  %362 = fadd contract float %344, %361
  %363 = add i32 %147, 12
  %364 = sext i32 %363 to i64
  %365 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %366 = bitcast ptr addrspace(3) %365 to ptr addrspace(3)
  %367 = getelementptr inbounds float, ptr addrspace(3) %366, i64 %364
  %368 = bitcast ptr addrspace(3) %367 to ptr addrspace(3)
  %369 = bitcast ptr addrspace(3) %368 to ptr addrspace(3)
  %370 = load float, ptr addrspace(3) %369, align 4
  %371 = add i32 %155, 192
  %372 = sext i32 %371 to i64
  %373 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %374 = bitcast ptr addrspace(3) %373 to ptr addrspace(3)
  %375 = getelementptr inbounds float, ptr addrspace(3) %374, i64 %372
  %376 = bitcast ptr addrspace(3) %375 to ptr addrspace(3)
  %377 = bitcast ptr addrspace(3) %376 to ptr addrspace(3)
  %378 = load float, ptr addrspace(3) %377, align 4
  %379 = fmul contract float %370, %378
  %380 = fadd contract float %362, %379
  %381 = add i32 %147, 13
  %382 = sext i32 %381 to i64
  %383 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %384 = bitcast ptr addrspace(3) %383 to ptr addrspace(3)
  %385 = getelementptr inbounds float, ptr addrspace(3) %384, i64 %382
  %386 = bitcast ptr addrspace(3) %385 to ptr addrspace(3)
  %387 = bitcast ptr addrspace(3) %386 to ptr addrspace(3)
  %388 = load float, ptr addrspace(3) %387, align 4
  %389 = add i32 %155, 208
  %390 = sext i32 %389 to i64
  %391 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %392 = bitcast ptr addrspace(3) %391 to ptr addrspace(3)
  %393 = getelementptr inbounds float, ptr addrspace(3) %392, i64 %390
  %394 = bitcast ptr addrspace(3) %393 to ptr addrspace(3)
  %395 = bitcast ptr addrspace(3) %394 to ptr addrspace(3)
  %396 = load float, ptr addrspace(3) %395, align 4
  %397 = fmul contract float %388, %396
  %398 = fadd contract float %380, %397
  %399 = add i32 %147, 14
  %400 = sext i32 %399 to i64
  %401 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %402 = bitcast ptr addrspace(3) %401 to ptr addrspace(3)
  %403 = getelementptr inbounds float, ptr addrspace(3) %402, i64 %400
  %404 = bitcast ptr addrspace(3) %403 to ptr addrspace(3)
  %405 = bitcast ptr addrspace(3) %404 to ptr addrspace(3)
  %406 = load float, ptr addrspace(3) %405, align 4
  %407 = add i32 %155, 224
  %408 = sext i32 %407 to i64
  %409 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %410 = bitcast ptr addrspace(3) %409 to ptr addrspace(3)
  %411 = getelementptr inbounds float, ptr addrspace(3) %410, i64 %408
  %412 = bitcast ptr addrspace(3) %411 to ptr addrspace(3)
  %413 = bitcast ptr addrspace(3) %412 to ptr addrspace(3)
  %414 = load float, ptr addrspace(3) %413, align 4
  %415 = fmul contract float %406, %414
  %416 = fadd contract float %398, %415
  %417 = add i32 %147, 15
  %418 = sext i32 %417 to i64
  %419 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_0._gpu_shared_mem to ptr addrspace(3)
  %420 = bitcast ptr addrspace(3) %419 to ptr addrspace(3)
  %421 = getelementptr inbounds float, ptr addrspace(3) %420, i64 %418
  %422 = bitcast ptr addrspace(3) %421 to ptr addrspace(3)
  %423 = bitcast ptr addrspace(3) %422 to ptr addrspace(3)
  %424 = load float, ptr addrspace(3) %423, align 4
  %425 = add i32 %155, 240
  %426 = sext i32 %425 to i64
  %427 = bitcast ptr addrspace(3) @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442._global_alloc_1._gpu_shared_mem to ptr addrspace(3)
  %428 = bitcast ptr addrspace(3) %427 to ptr addrspace(3)
  %429 = getelementptr inbounds float, ptr addrspace(3) %428, i64 %426
  %430 = bitcast ptr addrspace(3) %429 to ptr addrspace(3)
  %431 = bitcast ptr addrspace(3) %430 to ptr addrspace(3)
  %432 = load float, ptr addrspace(3) %431, align 4
  %433 = fmul contract float %424, %432
  %434 = fadd contract float %416, %433
  call void @air.wg.barrier(i32 2, i32 1)
  br label %31

435:                                              ; preds = %56
  %436 = phi float [ %32, %56 ]
  %437 = icmp slt i64 %26, 2048
  %438 = icmp slt i64 %30, 2048
  %439 = select i1 %437, i1 %438, i1 false
  br i1 %439, label %440, label %450

440:                                              ; preds = %435
  %441 = trunc i64 %26 to i32
  %442 = trunc i64 %30 to i32
  %443 = mul i32 %441, 2048
  %444 = add i32 %443, %442
  %445 = sext i32 %444 to i64
  %446 = bitcast ptr addrspace(1) %12 to ptr addrspace(1)
  %447 = getelementptr inbounds float, ptr addrspace(1) %446, i64 %445
  %448 = bitcast ptr addrspace(1) %447 to ptr addrspace(1)
  %449 = bitcast ptr addrspace(1) %448 to ptr addrspace(1)
  store float %436, ptr addrspace(1) %449, align 4
  br label %451

450:                                              ; preds = %435
  br label %451

451:                                              ; preds = %440, %450
  ret void
}

attributes #0 = { convergent }
attributes #1 = { convergent mustprogress nofree norecurse nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "memory"="argmem: readwrite" "metal.kernel"="true" "metal.thread_args_added"="true" "metal.thread_index_in_simdgroup_idx"="7" "metal.thread_position_in_threadgroup_idx"="4" "metal.threadgroup_position_in_grid_idx"="3" "metal.threads_per_grid_idx"="6" "metal.threads_per_threadgroup_idx"="5" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!11, !12, !13}
!air.language_version = !{!14}
!air.source_file_name = !{!15}
!llvm.ident = !{!16}
!air.version = !{!17}
!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24, !25, !26}

!0 = !{ptr @_03b_matmul_tiled_matmul_tiled6A6AoA6A_3b84f8d63567e442, !1, !2}
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
!15 = !{!"03b_matmul_tiled.metal"}
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

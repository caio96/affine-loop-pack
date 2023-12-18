// RUN: mlir-opt %s -split-input-file -affine-loop-pack | FileCheck %s

// CHECK-DAG:   #map = affine_map<(d0) -> (d0)>
// CHECK-DAG:   #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL: func.func @contraction_3d(%arg0: memref<50x60xf64>, %arg1: memref<80x100x50xf64>, %arg2: memref<100x80x60xf64>)
// CHECK:       %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  affine.for %arg3 = 0 to 50 {
// CHECK-NEXT:    affine.for %arg4 = 0 to 60 {
// CHECK-NEXT:      affine.store %cst, %arg0[%arg3, %arg4] : memref<50x60xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.for %arg3 = 0 to 50 {
// CHECK-NEXT:    %0 = affine.apply #map(%arg3)
// CHECK-NEXT:    %alloc = memref.alloc() : memref<1x80x100xf64>
// CHECK-NEXT:    affine.for %arg4 = 0 to 80 {
// CHECK-NEXT:      affine.for %arg5 = 0 to 100 {
// CHECK-NEXT:        affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
// CHECK-NEXT:          %1 = affine.load %arg1[%arg4, %arg5, %arg6] : memref<80x100x50xf64>
// CHECK-NEXT:          affine.store %1, %alloc[%arg6 - %arg3, %arg4, %arg5] : memref<1x80x100xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.for %arg4 = 0 to 60 {
// CHECK-NEXT:      affine.for %arg5 = 0 to 80 {
// CHECK-NEXT:        affine.for %arg6 = 0 to 100 {
// CHECK-NEXT:          %1 = affine.load %alloc[0, %arg5, %arg6] : memref<1x80x100xf64>
// CHECK-NEXT:          %2 = affine.load %arg2[%arg6, %arg5, %arg4] : memref<100x80x60xf64>
// CHECK-NEXT:          %3 = arith.mulf %1, %2 : f64
// CHECK-NEXT:          %4 = affine.load %arg0[%arg3, %arg4] : memref<50x60xf64>
// CHECK-NEXT:          %5 = arith.addf %4, %3 : f64
// CHECK-NEXT:          affine.store %5, %arg0[%arg3, %arg4] : memref<50x60xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<1x80x100xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @contraction_3d(%arg0: memref<50x60xf64>, %arg1: memref<80x100x50xf64>, %arg2: memref<100x80x60xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg3 = 0 to 50 {
    affine.for %arg4 = 0 to 60 {
      affine.store %cst, %arg0[%arg3, %arg4] : memref<50x60xf64>
    }
  }
  affine.for %arg3 = 0 to 50 {
    affine.for %arg4 = 0 to 60 {
      affine.for %arg5 = 0 to 80 {
        affine.for %arg6 = 0 to 100 {
          %0 = affine.load %arg1[%arg5, %arg6, %arg3] : memref<80x100x50xf64>
          %1 = affine.load %arg2[%arg6, %arg5, %arg4] : memref<100x80x60xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg0[%arg3, %arg4] : memref<50x60xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg0[%arg3, %arg4] : memref<50x60xf64>
        }
      }
    }
  }
  return
}

// -----

// CHECK-LABEL:  func.func @doitgen(%arg0: memref<150x140x160xf64>, %arg1: memref<160x160xf64>, %arg2: memref<160xf64>)
// CHECK:        %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.for %arg3 = 0 to 150 {
// CHECK-NEXT:     %alloc = memref.alloc() : memref<160x160xf64>
// CHECK-NEXT:     affine.for %arg4 = 0 to 160 {
// CHECK-NEXT:       affine.for %arg5 = 0 to 160 {
// CHECK-NEXT:         %0 = affine.load %arg1[%arg4, %arg5] : memref<160x160xf64>
// CHECK-NEXT:         affine.store %0, %alloc[%arg5, %arg4] : memref<160x160xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg4 = 0 to 140 {
// CHECK-NEXT:       affine.for %arg5 = 0 to 160 {
// CHECK-NEXT:         affine.store %cst, %arg2[%arg5] : memref<160xf64>
// CHECK-NEXT:         affine.for %arg6 = 0 to 160 {
// CHECK-NEXT:           %0 = affine.load %arg0[%arg3, %arg4, %arg6] : memref<150x140x160xf64>
// CHECK-NEXT:           %1 = affine.load %alloc[%arg5, %arg6] : memref<160x160xf64>
// CHECK-NEXT:           %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:           %3 = affine.load %arg2[%arg5] : memref<160xf64>
// CHECK-NEXT:           %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:           affine.store %4, %arg2[%arg5] : memref<160xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg5 = 0 to 160 {
// CHECK-NEXT:         %0 = affine.load %arg2[%arg5] : memref<160xf64>
// CHECK-NEXT:         affine.store %0, %arg0[%arg3, %arg4, %arg5] : memref<150x140x160xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %alloc : memref<160x160xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
func.func @doitgen(%arg0: memref<150x140x160xf64>, %arg1: memref<160x160xf64>, %arg2: memref<160xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg3 = 0 to 150 {
    affine.for %arg4 = 0 to 140 {
      affine.for %arg5 = 0 to 160 {
        affine.store %cst, %arg2[%arg5] : memref<160xf64>
        affine.for %arg6 = 0 to 160 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg6] : memref<150x140x160xf64>
          %1 = affine.load %arg1[%arg6, %arg5] : memref<160x160xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg2[%arg5] : memref<160xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg2[%arg5] : memref<160xf64>
        }
      }
      affine.for %arg5 = 0 to 160 {
        %0 = affine.load %arg2[%arg5] : memref<160xf64>
        affine.store %0, %arg0[%arg3, %arg4, %arg5] : memref<150x140x160xf64>
      }
    }
  }
  return
}

// -----

// CHECK-DAG:   #map = affine_map<(d0, d1) -> (d0 * 100)>
// CHECK-DAG:   #map1 = affine_map<(d0, d1) -> (d1 * 100)>
// CHECK-DAG:   #map2 = affine_map<(d0) -> (d0 * 100)>
// CHECK-DAG:   #map3 = affine_map<(d0) -> (d0 * 100 + 100)>
// CHECK-DAG:   #map4 = affine_map<(d0, d1, d2, d3) -> (d0 * 100 + d1)>
// CHECK-DAG:   #map5 = affine_map<(d0, d1, d2, d3) -> (d2 * 100 + d3)>
// CHECK-LABEL: func.func @tiled_gemm(%arg0: f64, %arg1: f64, %arg2: memref<1000x1100xf64>, %arg3: memref<1000x1200xf64>, %arg4: memref<1200x1100xf64>)
// CHECK:       affine.for %arg5 = 0 to 10 {
// CHECK-NEXT:    affine.for %arg6 = 0 to 11 {
// CHECK-NEXT:      affine.for %arg7 = 0 to 100 {
// CHECK-NEXT:        affine.for %arg8 = 0 to 100 {
// CHECK-NEXT:          %0 = affine.load %arg2[%arg5 * 100 + %arg7, %arg6 * 100 + %arg8] : memref<1000x1100xf64>
// CHECK-NEXT:          %1 = arith.mulf %0, %arg1 : f64
// CHECK-NEXT:          affine.store %1, %arg2[%arg5 * 100 + %arg7, %arg6 * 100 + %arg8] : memref<1000x1100xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.for %arg5 = 0 to 10 {
// CHECK-NEXT:    affine.for %arg6 = 0 to 11 {
// CHECK-NEXT:      affine.for %arg7 = 0 to 12 {
// CHECK-NEXT:        %0 = affine.apply #map(%arg7, %arg6)
// CHECK-NEXT:        %1 = affine.apply #map1(%arg7, %arg6)
// CHECK-NEXT:        %alloc = memref.alloc() : memref<100x100xf64>
// CHECK-NEXT:        affine.for %arg8 = #map2(%arg7) to #map3(%arg7) {
// CHECK-NEXT:          affine.for %arg9 = #map2(%arg6) to #map3(%arg6) {
// CHECK-NEXT:            %2 = affine.load %arg4[%arg8, %arg9] : memref<1200x1100xf64>
// CHECK-NEXT:            affine.store %2, %alloc[%arg8 - %arg7 * 100, %arg9 - %arg6 * 100] : memref<100x100xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg8 = 0 to 100 {
// CHECK-NEXT:          affine.for %arg9 = 0 to 100 {
// CHECK-NEXT:            %2 = affine.load %arg3[%arg5 * 100 + %arg8, %arg7 * 100 + %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:            %3 = arith.mulf %arg0, %2 : f64
// CHECK-NEXT:            affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:              %4 = affine.load %arg2[%arg5 * 100 + %arg8, %arg6 * 100 + %arg10] : memref<1000x1100xf64>
// CHECK-NEXT:              %5 = affine.apply #map4(%arg7, %arg9, %arg6, %arg10)
// CHECK-NEXT:              %6 = affine.apply #map5(%arg7, %arg9, %arg6, %arg10)
// CHECK-NEXT:              %7 = affine.load %alloc[%arg9, %arg10] : memref<100x100xf64>
// CHECK-NEXT:              %8 = arith.mulf %3, %7 : f64
// CHECK-NEXT:              %9 = arith.addf %4, %8 : f64
// CHECK-NEXT:              affine.store %9, %arg2[%arg5 * 100 + %arg8, %arg6 * 100 + %arg10] : memref<1000x1100xf64>
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.dealloc %alloc : memref<100x100xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @tiled_gemm(%arg0: f64, %arg1: f64, %arg2: memref<1000x1100xf64>, %arg3: memref<1000x1200xf64>, %arg4: memref<1200x1100xf64>) {
  affine.for %arg5 = 0 to 10 {
    affine.for %arg6 = 0 to 11 {
      affine.for %arg7 = 0 to 100 {
        affine.for %arg8 = 0 to 100 {
          %0 = affine.load %arg2[%arg5 * 100 + %arg7, %arg6 * 100 + %arg8] : memref<1000x1100xf64>
          %1 = arith.mulf %0, %arg1 : f64
          affine.store %1, %arg2[%arg5 * 100 + %arg7, %arg6 * 100 + %arg8] : memref<1000x1100xf64>
        }
      }
    }
  }
  affine.for %arg5 = 0 to 10 {
    affine.for %arg6 = 0 to 11 {
      affine.for %arg7 = 0 to 12 {
        affine.for %arg8 = 0 to 100 {
          affine.for %arg9 = 0 to 100 {
            %0 = affine.load %arg3[%arg5 * 100 + %arg8, %arg7 * 100 + %arg9] : memref<1000x1200xf64>
            %1 = arith.mulf %arg0, %0 : f64
            affine.for %arg10 = 0 to 100 {
              %2 = affine.load %arg2[%arg5 * 100 + %arg8, %arg6 * 100 + %arg10] : memref<1000x1100xf64>
              %3 = affine.load %arg4[%arg7 * 100 + %arg9, %arg6 * 100 + %arg10] : memref<1200x1100xf64>
              %4 = arith.mulf %1, %3 : f64
              %5 = arith.addf %2, %4 : f64
              affine.store %5, %arg2[%arg5 * 100 + %arg8, %arg6 * 100 + %arg10] : memref<1000x1100xf64>
            }
          }
        }
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #map = affine_map<(d0) -> (d0 * 100)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 100 + 100)>
// CHECK-DAG: #map2 = affine_map<(d0, d1, d2) -> (d0 + 1000)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
// CHECK-DAG: #map4 = affine_map<(d0, d1) -> (d0 * 100)>
// CHECK-DAG: #map5 = affine_map<(d0, d1) -> (d1 * 100)>
// CHECK-DAG: #map6 = affine_map<(d0, d1, d2, d3) -> (d0 * 100 + d1)>
// CHECK-DAG: #map7 = affine_map<(d0, d1, d2, d3) -> (d2 * 100 + d3)>
// CHECK-DAG: #map8 = affine_map<(d0, d1) -> (d1 * 100 - 1099)>
// CHECK-DAG: #map9 = affine_map<(d0) -> (d0 * 100 - 1099)>
// CHECK-DAG: #map10 = affine_map<(d0) -> (1200, d0 * 100 - 999)>
// CHECK-DAG: #map11 = affine_map<(d0) -> (d0 * -100 + 2299, 100)>
// CHECK-DAG: #map12 = affine_map<(d0, d1, d2, d3) -> (d2 * 100 + d3 - 1099)>
// CHECK-DAG: #set = affine_set<(d0) : (-d0 + 8 >= 0)>
// CHECK-DAG: #set1 = affine_set<(d0) : (d0 - 9 >= 0)>
// CHECK-DAG: #set2 = affine_set<(d0) : (d0 - 10 == 0)>
// CHECK-DAG: #set3 = affine_set<(d0) : (-d0 + 9 >= 0)>
// CHECK-DAG: #set4 = affine_set<(d0) : (d0 - 11 >= 0)>
// CHECK-LABEL: func.func @tiled_fused_2mm(%arg0: f64, %arg1: f64, %arg2: memref<800x900xf64>, %arg3: memref<800x1100xf64>, %arg4: memref<1100x900xf64>, %arg5: memref<900x1200xf64>, %arg6: memref<800x1200xf64>) {
// CHECK:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.for %arg7 = 0 to 8 {
// CHECK-NEXT:     affine.for %arg8 = 0 to 12 {
// CHECK-NEXT:       affine.if #set(%arg8) {
// CHECK-NEXT:         affine.for %arg9 = 0 to 100 {
// CHECK-NEXT:           affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:             %0 = affine.load %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
// CHECK-NEXT:             %1 = arith.mulf %0, %arg1 : f64
// CHECK-NEXT:             affine.store %1, %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
// CHECK-NEXT:             affine.store %cst, %arg2[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x900xf64>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.if #set1(%arg8) {
// CHECK-NEXT:         affine.for %arg9 = 0 to 100 {
// CHECK-NEXT:           affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:             %0 = affine.load %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
// CHECK-NEXT:             %1 = arith.mulf %0, %arg1 : f64
// CHECK-NEXT:             affine.store %1, %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %arg7 = 0 to 8 {
// CHECK-NEXT:     affine.for %arg8 = 0 to 9 {
// CHECK-NEXT:       affine.for %arg9 = 0 to 23 {
// CHECK-NEXT:         affine.if #set2(%arg9) {
// CHECK-NEXT:           %0 = affine.apply #map(%arg8)
// CHECK-NEXT:           %alloc = memref.alloc() : memref<100x100xf64>
// CHECK-NEXT:           affine.for %arg10 = 1000 to 1100 {
// CHECK-NEXT:             affine.for %arg11 = #map(%arg8) to #map1(%arg8) {
// CHECK-NEXT:               %1 = affine.load %arg4[%arg10, %arg11] : memref<1100x900xf64>
// CHECK-NEXT:               affine.store %1, %alloc[%arg11 - %arg8 * 100, %arg10 - 1000] : memref<100x100xf64>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:             affine.for %arg11 = 0 to 100 {
// CHECK-NEXT:               affine.for %arg12 = 0 to 100 {
// CHECK-NEXT:                 %6 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:                 %7 = affine.load %arg3[%arg7 * 100 + %arg10, %arg12 + 1000] : memref<800x1100xf64>
// CHECK-NEXT:                 %8 = arith.mulf %arg0, %7 : f64
// CHECK-NEXT:                 %9 = affine.apply #map2(%arg12, %arg8, %arg11)
// CHECK-NEXT:                 %10 = affine.apply #map3(%arg12, %arg8, %arg11)
// CHECK-NEXT:                 %11 = affine.load %alloc[%arg11, %arg12] : memref<100x100xf64>
// CHECK-NEXT:                 %12 = arith.mulf %8, %11 : f64
// CHECK-NEXT:                 %13 = arith.addf %6, %12 : f64
// CHECK-NEXT:                 affine.store %13, %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:               }
// CHECK-NEXT:               %1 = affine.load %arg6[%arg7 * 100 + %arg10, 0] : memref<800x1200xf64>
// CHECK-NEXT:               %2 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:               %3 = affine.load %arg5[%arg8 * 100 + %arg11, 0] : memref<900x1200xf64>
// CHECK-NEXT:               %4 = arith.mulf %2, %3 : f64
// CHECK-NEXT:               %5 = arith.addf %1, %4 : f64
// CHECK-NEXT:               affine.store %5, %arg6[%arg7 * 100 + %arg10, 0] : memref<800x1200xf64>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           memref.dealloc %alloc : memref<100x100xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.if #set3(%arg9) {
// CHECK-NEXT:           %0 = affine.apply #map4(%arg9, %arg8)
// CHECK-NEXT:           %1 = affine.apply #map5(%arg9, %arg8)
// CHECK-NEXT:           %alloc = memref.alloc() : memref<100x100xf64>
// CHECK-NEXT:           affine.for %arg10 = #map(%arg9) to #map1(%arg9) {
// CHECK-NEXT:             affine.for %arg11 = #map(%arg8) to #map1(%arg8) {
// CHECK-NEXT:               %2 = affine.load %arg4[%arg10, %arg11] : memref<1100x900xf64>
// CHECK-NEXT:               affine.store %2, %alloc[%arg11 - %arg8 * 100, %arg10 - %arg9 * 100] : memref<100x100xf64>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:             affine.for %arg11 = 0 to 100 {
// CHECK-NEXT:               affine.for %arg12 = 0 to 100 {
// CHECK-NEXT:                 %2 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:                 %3 = affine.load %arg3[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12] : memref<800x1100xf64>
// CHECK-NEXT:                 %4 = arith.mulf %arg0, %3 : f64
// CHECK-NEXT:                 %5 = affine.apply #map6(%arg9, %arg12, %arg8, %arg11)
// CHECK-NEXT:                 %6 = affine.apply #map7(%arg9, %arg12, %arg8, %arg11)
// CHECK-NEXT:                 %7 = affine.load %alloc[%arg11, %arg12] : memref<100x100xf64>
// CHECK-NEXT:                 %8 = arith.mulf %4, %7 : f64
// CHECK-NEXT:                 %9 = arith.addf %2, %8 : f64
// CHECK-NEXT:                 affine.store %9, %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           memref.dealloc %alloc : memref<100x100xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.if #set4(%arg9) {
// CHECK-NEXT:           %0 = affine.apply #map4(%arg8, %arg9)
// CHECK-NEXT:           %1 = affine.apply #map8(%arg8, %arg9)
// CHECK-NEXT:           %alloc = memref.alloc() : memref<100x100xf64>
// CHECK-NEXT:           affine.for %arg10 = #map(%arg8) to #map1(%arg8) {
// CHECK-NEXT:             affine.for %arg11 = #map9(%arg9) to min #map10(%arg9) {
// CHECK-NEXT:               %2 = affine.load %arg5[%arg10, %arg11] : memref<900x1200xf64>
// CHECK-NEXT:               affine.store %2, %alloc[%arg10 - %arg8 * 100, %arg11 - %arg9 * 100 + 1099] : memref<100x100xf64>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.for %arg10 = 0 to 100 {
// CHECK-NEXT:             affine.for %arg11 = 0 to 100 {
// CHECK-NEXT:               %2 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
// CHECK-NEXT:               affine.for %arg12 = 0 to min #map11(%arg9) {
// CHECK-NEXT:                 %3 = affine.load %arg6[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12 - 1099] : memref<800x1200xf64>
// CHECK-NEXT:                 %4 = affine.apply #map6(%arg8, %arg11, %arg9, %arg12)
// CHECK-NEXT:                 %5 = affine.apply #map12(%arg8, %arg11, %arg9, %arg12)
// CHECK-NEXT:                 %6 = affine.load %alloc[%arg11, %arg12] : memref<100x100xf64>
// CHECK-NEXT:                 %7 = arith.mulf %2, %6 : f64
// CHECK-NEXT:                 %8 = arith.addf %3, %7 : f64
// CHECK-NEXT:                 affine.store %8, %arg6[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12 - 1099] : memref<800x1200xf64>
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           memref.dealloc %alloc : memref<100x100xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
#map = affine_map<(d0) -> (d0 * -100 + 2299, 100)>
#set = affine_set<(d0) : (-d0 + 8 >= 0)>
#set1 = affine_set<(d0) : (d0 - 9 >= 0)>
#set2 = affine_set<(d0) : (d0 - 10 == 0)>
#set3 = affine_set<(d0) : (-d0 + 9 >= 0)>
#set4 = affine_set<(d0) : (d0 - 11 >= 0)>
func.func @tiled_fused_2mm(%arg0: f64, %arg1: f64, %arg2: memref<800x900xf64>, %arg3: memref<800x1100xf64>, %arg4: memref<1100x900xf64>, %arg5: memref<900x1200xf64>, %arg6: memref<800x1200xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg7 = 0 to 8 {
    affine.for %arg8 = 0 to 12 {
      affine.if #set(%arg8) {
        affine.for %arg9 = 0 to 100 {
          affine.for %arg10 = 0 to 100 {
            %0 = affine.load %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
            %1 = arith.mulf %0, %arg1 : f64
            affine.store %1, %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
            affine.store %cst, %arg2[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x900xf64>
          }
        }
      }
      affine.if #set1(%arg8) {
        affine.for %arg9 = 0 to 100 {
          affine.for %arg10 = 0 to 100 {
            %0 = affine.load %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
            %1 = arith.mulf %0, %arg1 : f64
            affine.store %1, %arg6[%arg7 * 100 + %arg9, %arg8 * 100 + %arg10] : memref<800x1200xf64>
          }
        }
      }
    }
  }
  affine.for %arg7 = 0 to 8 {
    affine.for %arg8 = 0 to 9 {
      affine.for %arg9 = 0 to 23 {
        affine.if #set2(%arg9) {
          affine.for %arg10 = 0 to 100 {
            affine.for %arg11 = 0 to 100 {
              affine.for %arg12 = 0 to 100 {
                %5 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
                %6 = affine.load %arg3[%arg7 * 100 + %arg10, %arg12 + 1000] : memref<800x1100xf64>
                %7 = arith.mulf %arg0, %6 : f64
                %8 = affine.load %arg4[%arg12 + 1000, %arg8 * 100 + %arg11] : memref<1100x900xf64>
                %9 = arith.mulf %7, %8 : f64
                %10 = arith.addf %5, %9 : f64
                affine.store %10, %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
              }
              %0 = affine.load %arg6[%arg7 * 100 + %arg10, 0] : memref<800x1200xf64>
              %1 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
              %2 = affine.load %arg5[%arg8 * 100 + %arg11, 0] : memref<900x1200xf64>
              %3 = arith.mulf %1, %2 : f64
              %4 = arith.addf %0, %3 : f64
              affine.store %4, %arg6[%arg7 * 100 + %arg10, 0] : memref<800x1200xf64>
            }
          }
        }
        affine.if #set3(%arg9) {
          affine.for %arg10 = 0 to 100 {
            affine.for %arg11 = 0 to 100 {
              affine.for %arg12 = 0 to 100 {
                %0 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
                %1 = affine.load %arg3[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12] : memref<800x1100xf64>
                %2 = arith.mulf %arg0, %1 : f64
                %3 = affine.load %arg4[%arg9 * 100 + %arg12, %arg8 * 100 + %arg11] : memref<1100x900xf64>
                %4 = arith.mulf %2, %3 : f64
                %5 = arith.addf %0, %4 : f64
                affine.store %5, %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
              }
            }
          }
        }
        affine.if #set4(%arg9) {
          affine.for %arg10 = 0 to 100 {
            affine.for %arg11 = 0 to 100 {
              %0 = affine.load %arg2[%arg7 * 100 + %arg10, %arg8 * 100 + %arg11] : memref<800x900xf64>
              affine.for %arg12 = 0 to min #map(%arg9) {
                %1 = affine.load %arg6[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12 - 1099] : memref<800x1200xf64>
                %2 = affine.load %arg5[%arg8 * 100 + %arg11, %arg9 * 100 + %arg12 - 1099] : memref<900x1200xf64>
                %3 = arith.mulf %0, %2 : f64
                %4 = arith.addf %1, %3 : f64
                affine.store %4, %arg6[%arg7 * 100 + %arg10, %arg9 * 100 + %arg12 - 1099] : memref<800x1200xf64>
              }
            }
          }
        }
      }
    }
  }
  return
}

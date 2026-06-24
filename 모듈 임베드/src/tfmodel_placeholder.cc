// 실제 TFLite 모델 2개 (컴파일 시 바이너리에 임베드).
//
// 1) cascade (메인, 시퀀스): model_tflite[] / model_tflite_len  ← ../tfmodel.h
//    int8 (789,208 bytes, parity_int8 human 0.71 / pose 0.22)
// 2) single (warmup, 단일 프레임): model2_tflite[] / model2_tflite_len  ← ../tfmodel_single.h
//    fp32 (90,896 bytes, 입력 (1,256,5) → person(1,1) + pose(1,9))
//
// 둘 다 C linkage 로 노출 (main.cc 의 extern "C" 선언과 매칭).

extern "C" {
#include "../tfmodel.h"
#include "../tfmodel_single.h"
}

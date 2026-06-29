# Data Description

## Raw Data

원본 데이터는 PNG Port Moresby 해역의 해수 온도 및 염도 시계열 자료입니다. 각 파일은 `Time` 컬럼과 3×3 공간 격자에 해당하는 `Point_R*_C*` 컬럼으로 구성됩니다.

| Variable | File | Depth | Unit |
|---|---|---:|---|
| T0M | `PNG_Port_Moresby_해수온도_0m.csv` | 0 m | °C |
| T10M | `PNG_Port_Moresby_해수온도_10m.csv` | 10 m | °C |
| S0M | `PNG_Port_Moresby_해수염도_0m.csv` | 0 m | PSU |
| S10M | `PNG_Port_Moresby_해수염도_10m.csv` | 10 m | PSU |

## Super-Resolution Data

1km 결과 데이터는 기존 3×3 저해상도 격자를 12×24 고해상도 격자로 확장한 결과입니다. 각 결과 파일은 `Time`과 `Point_R1_C1`부터 `Point_R12_C24`까지 총 288개 공간 격자 컬럼으로 구성됩니다.

## Methods

- **Uniform**: 원본 격자 값을 고해상도 격자에 반복 복제합니다.
- **Bilinear**: 인접 격자 간 선형 보간을 수행합니다.
- **Kriging**: 공간 자기상관성을 고려하여 미관측 격자 값을 추정합니다.
- **DIP**: Bilinear 결과를 입력으로, Kriging 결과를 pseudo GT로 활용하여 CNN 기반 refinement를 수행합니다.

## Caution

본 데이터는 고해상도 실측값이 없는 조건에서 생성된 결과이므로, 해상도 증강 기법 간 비교는 상대적 공간 분포 변화 및 발전량 추정 민감도 분석 관점에서 해석해야 합니다.

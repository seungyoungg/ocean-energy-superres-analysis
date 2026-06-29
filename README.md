# PNG Port Moresby Ocean Data Super-Resolution

PNG Port Moresby 해역의 해수 온도·염도 관측자료를 대상으로, 기존 8×4 km 격자 자료를 1×1 km 격자로 공간 해상도 증강한 코드와 결과 데이터입니다.

본 저장소는 KNU 공유용으로 바로 실행·검토할 수 있도록 원본 데이터, 해상도 증강 결과, 실행 스크립트, 시각화 노트북을 함께 정리한 버전입니다.

## 1. Repository Structure

```text
ocean_energy_superres_knu/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── PNG_Port_Moresby_해수온도_0m.csv
│   │   ├── PNG_Port_Moresby_해수온도_10m.csv
│   │   ├── PNG_Port_Moresby_해수염도_0m.csv
│   │   └── PNG_Port_Moresby_해수염도_10m.csv
│   └── superres_1km/
│       ├── uniform/
│       ├── bilinear/
│       ├── kriging/
│       ├── dip/
│       └── superres_run_summary.csv
├── src/
│   └── run_superres_background.py
├── notebooks/
│   └── ocean_energy_figures_02_45.ipynb
└── docs/
    └── DATA_DESCRIPTION.md
```

## 2. Data Description

### Raw input data

| File | Variable | Depth | Description |
|---|---:|---:|---|
| `PNG_Port_Moresby_해수온도_0m.csv` | Temperature | 0 m | 표층 해수 온도 |
| `PNG_Port_Moresby_해수온도_10m.csv` | Temperature | 10 m | 10 m 수심 해수 온도 |
| `PNG_Port_Moresby_해수염도_0m.csv` | Salinity | 0 m | 표층 해수 염도 |
| `PNG_Port_Moresby_해수염도_10m.csv` | Salinity | 10 m | 10 m 수심 해수 염도 |

각 CSV는 `Time` 컬럼과 공간 격자별 `Point_R*_C*` 컬럼으로 구성됩니다.

### Super-resolution output data

`data/superres_1km/`에는 1×1 km 해상도로 증강된 결과가 방법별로 저장되어 있습니다.

| Directory | Method | Description |
|---|---|---|
| `uniform/` | Uniform interpolation | 저해상도 격자 값을 고해상도 격자에 반복 복제하는 기준선 방법 |
| `bilinear/` | Bilinear interpolation | 인접 격자 간 선형 보간을 통한 공간 연속성 반영 |
| `kriging/` | Kriging interpolation | 공간 자기상관성을 고려한 지리통계 기반 보간 |
| `dip/` | Kriging-supervised DL refinement | Bilinear 결과를 입력으로, Kriging 결과를 pseudo GT로 사용하는 딥러닝 기반 보정 |

출력 파일명은 다음 규칙을 따릅니다.

```text
{VARIABLE}_{METHOD}_1km.csv
```

예: `T0M_kriging_1km.csv`, `S10M_dip_1km.csv`

변수명은 다음을 의미합니다.

| Variable | Description |
|---|---|
| `T0M` | 0 m 해수 온도 |
| `T10M` | 10 m 해수 온도 |
| `S0M` | 0 m 해수 염도 |
| `S10M` | 10 m 해수 염도 |

## 3. Environment Setup

Python 3.9 이상 환경을 권장합니다.

```bash
pip install -r requirements.txt
```

GPU 기반 DIP 실행 시 CUDA 환경의 PyTorch가 필요합니다. CUDA가 없는 경우 CPU로 실행됩니다.

## 4. How to Run

저장소 루트에서 다음 명령어를 실행합니다.

```bash
python src/run_superres_background.py
```

기본적으로 다음 입력·출력 경로를 사용합니다.

```text
Input : data/raw/
Output: data/superres_1km/
```

특정 방법만 실행하려면 다음과 같이 지정합니다.

```bash
python src/run_superres_background.py --methods uniform bilinear kriging
```

DIP만 실행하려면 다음과 같이 실행합니다.

```bash
python src/run_superres_background.py --methods dip --dl-iters 300
```

기존 결과를 덮어쓰려면 `--overwrite` 옵션을 추가합니다.

```bash
python src/run_superres_background.py --overwrite
```

## 5. Visualization

시각화 코드는 아래 노트북에 정리되어 있습니다.

```text
notebooks/ocean_energy_figures_02_45.ipynb
```

노트북은 원본 자료와 1km 해상도 증강 결과를 활용하여 보고서 Figure 2–45에 해당하는 해양 물성 및 발전량 분포를 생성하는 용도입니다.

## 6. Notes

- 본 저장소의 1km 결과는 고해상도 실측 Ground Truth가 없는 조건에서 생성된 해상도 증강 결과입니다.
- 따라서 방법 간 비교는 절대 성능 평가보다는 공간 분포 변화와 발전량 추정 민감도 분석 관점에서 해석하는 것이 적절합니다.
- DIP 결과는 Kriging 결과를 pseudo Ground Truth로 활용한 refinement 결과이므로, 독립적인 고해상도 관측 검증 결과로 해석해서는 안 됩니다.

## 7. Recommended Citation in Reports

본 코드는 PNG Port Moresby 해역 해수 온도·염도 자료의 공간 해상도 증강 및 해양에너지 잠재량 분석을 위해 작성되었으며, 8×4 km 격자 자료를 1×1 km 격자로 확장하기 위해 Uniform, Bilinear, Kriging, Kriging-supervised 딥러닝 기반 refinement 방법을 적용하였습니다.

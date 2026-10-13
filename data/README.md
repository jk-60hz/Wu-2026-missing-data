# 데이터

## 출처

모든 데이터는 **NREL PVDAQ (Photovoltaic Data Acquisition)**의 공개 데이터이다. 받은 곳은 OEDI 공개 S3 버킷 `s3://oedi-data-lake/pvdaq/`(HTTPS: `https://oedi-data-lake.s3.amazonaws.com/pvdaq/`)이고, 다운로드 일자는 2026-10-13이다.
라이선스는 OEDI/PVDAQ 공개 데이터 조건(CC BY 4.0, "NREL PVDAQ" 출처 표기)을 따른다.

| 데이터셋 | 원천 | 내용 |
|---|---|---|
| `maui_15min` | PVDAQ system 2105 (Maui Ocean Center, HI; 2023 Solar Data Prize) | 현장 GHI 일사계 1개 + 스트링 인버터 7개 AC 출력, 15분, 2022-01-10 ~ 2022-03-10 (60일, 5,760 step) |
| `maui_1h` | 위와 동일 | GHI + 인버터 8개, 1시간, 2021-11-01 ~ 2022-04-30 (181일, 4,344 step) |
| `stpete_1h` | PVDAQ St. Petersburg Parks CDP production meters (FL) 1314, 1318, 1325, 1326 | 반경 약 10 km 안의 공원 건물 4곳 AC 출력, 1시간, 2014-12-01 ~ 2015-04-30 (3,624 step) |
| `nola_1h` | PVDAQ New Orleans residential (LA) 1263, 1244, 1250, 1252, 1254, 1255, 1257, 1264, 1265, 1271 | 주택 지붕형 10곳 AC 출력(w_avg), 1시간, 2013-10-01 ~ 2014-02-28 (3,624 step) |

조사했지만 사용하지 못한 소스가 있다. 이 실행 환경의 네트워크 정책이 nrel.gov, DKASC(dkasolarcentre.com.au), PVGIS(re.jrc.ec.europa.eu), Open-Meteo, Open Power System Data, data.go.kr, figshare, zenodo, huggingface 연결을 막았다.
따라서 NREL Solar Power Data for Integration Studies, DKASC, 한국 공공데이터, 논문 원 코드(figshare)는 쓰지 못했다.
AWS S3(OEDI, NSRDB 버킷)만 접근할 수 있었다.

## 재현 방법

```bash
python scripts/download_pvdaq.py maui     # ~200 MB
python scripts/download_pvdaq.py stpete   # ~12k daily CSV
python scripts/download_pvdaq.py nola     # ~28k daily CSV
python scripts/process_pvdaq.py stpete    # 15-s -> 15-min
python scripts/process_pvdaq.py nola      # 5-min -> 15-min
python scripts/build_datasets.py          # -> data/processed/*.parquet + datasets.json
python scripts/make_testsets.py --datasets maui_15min,maui_1h,stpete_1h,nola_1h --seeds 5
```

원본 데이터(`data/raw/`)와 중간 산출물(`*_all.parquet`), 마스크(`data/testsets/`)는 git에서 제외했다. 마스크는 `(dataset, scenario, param, seed)`로 결정적으로 다시 생성된다.
최종 분석용 파일 `data/processed/{maui_15min,maui_1h,stpete_1h,nola_1h}.parquet`(각 200 KB 이하)는 커밋했다.

## 전처리와 품질 판단

- **시간 기준**: 모든 인덱스는 naive UTC 구간 시작 시각이다.
  - Maui 2105: 인버터 시계와 일사계 시계가 서로 달랐다. 각각을 pvlib clear-sky GHI와의 상관이 최대가 되는 오프셋으로 맞췄다(인버터 +10h15m, 일사계 −3h15m. 15분 단위로 탐색했고 상관은 각각 0.90, 0.98).
  - St. Pete: EST(UTC−5)로 보고 +5h 보정했다. 일중 최대 출력 시각이 12시 전후인 것으로 확인했다.
  - NOLA: CST(UTC−6)로 보고 +6h 보정했다.
- **야간**: 인버터는 야간에 기록하지 않는다(커버리지 약 50%). 태양 천정각 ≥ 90°인 시점은 0으로 채웠다. 2 step 이하의 짧은 공백만 선형 보간했고, 그보다 긴 공백이 있는 열은 제외했다. 따라서 남은 데이터는 **결측이 전혀 없는 정답**이다.
- **제외한 채널**:
  - Maui `inv06`: 주간 내내 0.12로 고정된 죽은 채널.
  - Maui `inv05`: 여러 날에 걸친 부분 출력 고장.
  - Maui `inv01`, `inv04`: 15분 데이터 창에서 기록 누락.
  - St. Pete 1321: 시계가 약 3시간 어긋나 있음. 1329: 값이 대부분 0이고 스파이크가 섞임. 나머지 시스템: 해당 창의 커버리지 부족.
  - NOLA 1268, 1273: 비정상(야간 피크, 또는 값이 전부 0).
- **용량(capacity)**: 열마다 관측값의 99.9 백분위수를 명목 용량으로 썼다. %-of-capacity 지표의 분모다.
- **단위**: Maui 인버터 채널의 이름은 "ac_output (kWh)"이지만 5분 샘플 값이라 출력에 비례하는 양으로 취급했다. 모든 지표가 정규화되어 있으므로 단위는 결과에 영향을 주지 않는다.

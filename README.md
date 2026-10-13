# FPRM 결측 복원 기법의 태양광 데이터 검증

Wu et al. (Nat. Commun. 2026)의 **FPRM (Full-Partial Reconstruction Mapping)** 결측치 복원 기법을 Python으로 구현했다.
이를 공개 태양광 실측 데이터(NREL PVDAQ)에 대해 결측 시나리오 7종과 베이스라인 13종으로 벤치마크했다.

- **작업 지시서**: [`PROMPT.md`](PROMPT.md)
- **결과 리포트(결론 먼저)**: [`REPORT.md`](REPORT.md)
- **데이터 설명**: [`data/README.md`](data/README.md)

## 구조

```
paper/        논문 PDF
src/fprm/     FPRM 구현 (embedding, GPR 매핑, E/tau 선택, 모델)
src/baselines/  베이스라인 (보간, clear-sky index, 이웃 스케일링, 직접 회귀, MICE ...)
src/missingness.py  결측 시나리오 S1–S7
src/methods.py      벤치마크 기법 레지스트리
src/solar.py, src/metrics.py, src/datasets.py, src/synthetic.py
scripts/      다운로드, 전처리, 테스트셋 생성, 벤치마크, 합성계 재현, 리포트 그림
configs/      main.yaml (본 실험), ablation.yaml (변형 + 참조 스윕)
tests/        pytest (임베딩, 누설 방지, 마스크, 논문 ecology 재현)
results/      metrics CSV, 집계표, 그림
```

## 빠른 실행

```bash
pip install numpy scipy pandas scikit-learn matplotlib pvlib pyarrow pyyaml pytest
python -m pytest -q tests
python scripts/make_testsets.py --datasets maui_15min,maui_1h,stpete_1h,nola_1h --seeds 5
python scripts/run_benchmark.py --config configs/main.yaml
python scripts/make_report.py
```

분석용 데이터(`data/processed/*.parquet`)는 저장소에 포함되어 있다.
원본부터 다시 만들려면 `data/README.md`의 순서를 따른다.

## FPRM 사용 예

```python
from src.fprm import FPRM
# X: (T, n) 배열, 결측은 NaN. 하나 이상의 열이 완전 관측되어 있어야 함
Y = FPRM(learner="gpr", refs="best").fit_impute(X)
```

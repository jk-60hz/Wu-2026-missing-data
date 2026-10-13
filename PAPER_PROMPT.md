# 작업 지시서: FPRM 태양광 검증 결과를 arXiv 형식 기술 보고서(영문/국문)로 작성

## 0. 목표

이 저장소에서 수행한 **"FPRM(Full-Partial Reconstruction Mapping) 결측 복원 기법의 태양광 데이터 적용 가능성 검증"** 결과를 논문으로 작성한다.
형식은 arXiv에 올라오는 LaTeX 기술 보고서(technical report)이고, **영문판과 국문판 두 편**을 만든다.
목표 수준은 **바로 투고(arXiv 업로드) 가능한 완성본**이다.

- 대상 원 논문: Wu et al., *Dynamics-informed machine learning for recovering extensive missing systems dynamics*, Nature Communications (2026), doi:10.1038/s41467-026-77922-1 (`paper/Wu2026_FPRM_NatCommun.pdf`)
- 저자 표기
  - 영문: **Jongkyu Kim** (jk.kim@60hz.io) & **Claude Opus 5.5**
  - 국문: **김종규** (jk.kim@60hz.io) & **Claude Opus 5.5**
- 소속은 따로 받지 않았다. 이메일 도메인을 근거로 소속을 추정해 넣지 말고, 소속란은 비워 두거나 생략한다. 리포트 끝에 "소속 표기 필요 여부"를 확인 사항으로 남긴다.

막히는 결정은 아래 "기본값"대로 진행하고, 어떤 선택을 했는지 마지막 보고에 적는다.

---

## 1. 입력 자료 (근거는 이것만 사용)

논문의 모든 수치·주장은 아래 파일에서 가져온다. **새 실험을 돌리거나 수치를 지어내지 않는다.**
수치를 옮길 때는 원본 CSV에서 다시 계산하거나 대조해 `REPORT.md`와 일치하는지 확인한다.

| 자료 | 경로 |
|---|---|
| 결과 리포트 (구성·논지의 기준) | `REPORT.md` |
| 원 작업 지시서 (실험 설계 의도) | `PROMPT.md` |
| 데이터 출처·전처리·제외 채널 | `data/README.md`, `data/processed/datasets.json` |
| 구현 | `src/fprm/`, `src/baselines/`, `src/missingness.py`, `src/methods.py`, `src/solar.py`, `src/metrics.py` |
| 실험 설정 | `configs/main.yaml`, `configs/ablation.yaml` |
| 원시 결과 | `results/main/metrics.csv`, `results/ablation/metrics.csv`, `results/synthetic/reproduction.csv`, `results/testsets_summary.csv` |
| 집계표 | `results/tables/*.csv` |
| 그림 | `results/figures/*.png` |
| 원 논문 | `paper/Wu2026_FPRM_NatCommun.pdf` (FPRM 정의, Eq. 6–19, Table 1 수치 인용용) |

---

## 2. 논문 구성 (영문 기준. 국문판은 같은 구조를 번역)

제목(기본값, 더 나은 안이 있으면 바꿔도 됨):
**"Does Dynamics-Informed Imputation Transfer to Solar PV? An Empirical Evaluation of Full-Partial Reconstruction Mapping on Real Photovoltaic Data"**
국문: **"동역학 기반 결측 복원은 태양광 발전 데이터에도 통하는가? — 실측 PV 데이터에서의 FPRM 실증 평가"**

1. **Abstract** (150–250 단어)
   - 문제, 방법, 데이터, 핵심 수치, 결론을 넣는다.
   - 핵심 수치: 임베딩 이득 약 7%, 단일 참조 FPRM은 최강 베이스라인 대비 +12–16%, FPRM-multi는 평균 순위 1위, S6 r=0.9에서 5–14% 우위.
2. **Introduction**
   - PV 데이터 결측 문제의 실무적 중요성과 FPRM의 주장을 소개한다.
   - 이 연구의 질문은 `PROMPT.md` 0장의 세 가지다.
   - 기여(contributions)를 bullet 3–5개로 정리한다.
3. **Related Work**
   - 원 논문 참고문헌 중 관련 항목(S-map, MVCM, Takens, GPR 등)을 다룬다.
   - PV 결측 대치·데이터 품질 관련 연구도 다룬다. **실제로 존재하는 문헌만 인용하고, 서지 정보를 확신할 수 없으면 넣지 않는다.**
   - 웹 검색이 막혀 있으면 원 논문 참고문헌 목록과 확실히 아는 표준 문헌(pvlib, Ineichen clear-sky model, scikit-learn, Rasmussen & Williams GPML, Takens 1981, Kennel 1992 FNN, Fraser & Swinney 1986 AMI, NREL PVDAQ)으로 한정한다.
4. **Method: FPRM**
   - 수식은 원 논문 표기를 따른다: full/partial reconstruction, mapping F, GPR posterior.
   - 본 구현의 세부를 적는다: E·τ 교차검증, fitrgp 유사 GPR 설정, subset-of-data, 야간 중복 행 제거.
   - 논문에 없는 확장(FPRM-multi, backward embedding, CSI 공간)은 명시적으로 구분한다.
   - **알고리즘 박스(algorithm2e 또는 algorithmic) 1개**를 넣는다.
5. **Experimental Setup**
   - 데이터셋 표: 4개 데이터셋, 변수 수, 해상도, 기간, 지정 참조.
   - 전처리: 시계 정합, 야간 처리, 제외 채널과 그 이유.
   - 결측 시나리오 표: S1–S8 정의와 파라미터, 각 시나리오가 대표하는 현실 상황.
   - 베이스라인 표, 평가 지표(주간 한정, %capacity RMSE, ρ, nRMSE, 에너지 오차), 누설 방지 규칙, 통계 검정(paired Wilcoxon).
6. **Results**
   - 6.1 합성계 재현: 논문 수치와 비교한 표, `synthetic_reproduction.png`
   - 6.2 전체 비교: 평균 순위 표, 중앙값 RMSE 표, `s1_mcar_curves.png`, `s2_block_curves.png`
   - 6.3 FPRM vs 최강 베이스라인: family summary 표, 히트맵
   - 6.4 참조가 1개뿐인 경우(S6): S6 표, `s6_oneref_curves.png`, 예시 그림
   - 6.5 실패 사례(S4, S5, S7)와 원인 분석
   - 6.6 Ablation과 참조 스윕(S8)
7. **Discussion**
   - 왜 일반 회귀가 이기는가: 이웃 간 상관이 0.97–0.99로 이미 높아 동역학 정보의 한계 이득이 작다.
   - FPRM이 버리는 정보: 부분 관측 이웃. MICE가 이기는 S6 r=0.5 사례가 근거다.
   - 일사계 참조의 한계, 실무 권장 파이프라인(결측 패턴별 분기), 후속 연구.
8. **Limitations**
   - `REPORT.md` 5.2를 기준으로 한다: 데이터 접근 제약, seed 수, 원 코드 미대조, lag FPRM 미구현, Lorenz 재현 불일치.
9. **Conclusion**
10. **Data and Code Availability**
    - 데이터는 NREL PVDAQ (OEDI S3, CC BY 4.0)이다.
    - 코드는 이 저장소다. 공개 URL이 정해지지 않았으면 `\url{https://github.com/jk-60hz/Wu-2026-missing-data}`를 placeholder로 넣고 확인 사항에 남긴다.
11. **Author Contributions / AI Disclosure**
    - Claude Opus 5.5가 공동 저자이므로 역할 분담을 사실대로 적는다.
      - 김종규: 연구 기획, 지시, 검토.
      - Claude Opus 5.5: 구현, 실험, 분석, 원고 작성.
    - arXiv의 AI 저자 정책(AI를 저자로 올리는 것에 대한 제한 가능성)을 **확인 사항으로 리포트에 명시**한다. 판단은 저자에게 맡긴다.
12. **References** (BibTeX)
13. **Appendix**
    - 전체 시나리오 × 기법 표(`rmse_cap_pivot.csv`), 데이터셋별 결과, 전 데이터셋 예시 그림.
    - 재현 명령어.

분량: 본문 10–14쪽(그림·표 포함)에 부록을 더한다.

---

## 3. 그림과 표

- **본문 그림 (최소 6개)**
  - 파이프라인 개념도 1개: 새로 그린다. TikZ 권장. 내용은 참조 시계열 → 지연 임베딩 → GPR 매핑 → 결측 시계열.
  - `synthetic_reproduction.png`
  - `s1_mcar_curves.png`, `s2_block_curves.png` (subfigure로 묶어도 됨)
  - `fprm_vs_best_heatmap.png`와 `fprm_multi_vs_best_heatmap.png` (나란히 배치하거나 하나는 부록으로)
  - `s6_oneref_curves.png`
  - 예시 시계열 1–2개: `example_nola_1h_S6_r09.png`, `example_nola_1h_S2_24h.png`
- **부록 그림**: 나머지 예시 그림(데이터셋별 S2 24h, S6 r=0.9).
- **그림 수정**
  - 기존 PNG를 그대로 써도 된다.
  - 논문 폭(단일 칼럼 약 6.5 in)에서 글자가 너무 작거나, 그림 안 제목이 캡션과 중복되면 `scripts/make_report.py`를 고쳐 **논문용 버전(PDF 벡터 권장)을 `paper_tex/figures/`에 새로 생성**한다.
  - 그림을 고칠 때는 `dataviz` 스킬 가이드를 따른다.
  - 국문판 그림의 축·범례는 영문 그대로 두어도 된다(기본값). 국문 캡션으로 설명한다.
- **캡션**: 그림이 무엇을 보여주는지와 읽는 법을 2–4문장으로 쓴다(예: "음수 = FPRM 우위, * = p<0.05").
- **표**
  - `booktabs`를 쓴다(`\toprule`, `\midrule`, `\bottomrule`).
  - 각 열의 최고값을 **굵게** 표시한다.
  - 수치는 CSV에서 스크립트로 생성해도 된다(`scripts/make_tables_tex.py` 등). 손으로 옮기면 반드시 대조한다.
- 모든 그림·표는 본문에서 `\ref`로 참조하고, 참조되지 않는 그림이 없게 한다.

---

## 4. LaTeX 형식

- **디렉터리**

  ```
  paper_tex/
    en/main.tex, en/refs.bib
    ko/main.tex, ko/refs.bib (en과 공유해도 됨)
    figures/          (논문용 그림. en/ko가 공유)
    tables/           (자동 생성 표 .tex, 필요시)
    Makefile 또는 build.sh
  ```

- **문서 클래스**: `article` 11pt, A4 또는 letter, 단일 칼럼, 여백 1 in.
  - 쓸 패키지: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`, `cleveref`, `subcaption`, `natbib`(또는 `biblatex`), `algorithm2e`, `xcolor`, `siunitx`(선택), `tikz`.
  - 첫 페이지 상단에 제목, 저자, 이메일, 날짜, abstract. 그다음 keywords.
- **국문판**
  - XeLaTeX + `kotex`(또는 `xetexko`)를 쓴다.
  - 한글 폰트는 설치된 것(예: Noto Sans/Serif CJK KR, 나눔)을 쓰고, 없으면 설치한다.
  - 영문 용어는 첫 등장 시 괄호로 병기한다(예: 지연 임베딩(delay embedding)).
  - 수식·변수명·기법명(FPRM, FPRM-multi, NbrRatio 등)은 영문판과 같게 유지한다.
- **arXiv 업로드 호환성**
  - 각 판을 **업로드용 소스 묶음**으로 만든다: `paper_tex/arxiv_en.tar.gz`, `paper_tex/arxiv_ko.tar.gz`.
  - 묶음에는 main.tex, .bbl, 그림, 스타일 파일이 들어가야 하고 외부 경로 의존이 없어야 한다.
  - 국문판은 arXiv에서 XeLaTeX 컴파일이 필요하다. 지원 여부와 한계를 확인 사항에 적는다. arXiv가 pdfLaTeX를 기본으로 쓰므로, 국문판은 PDF 업로드가 대안일 수 있다.
- **빌드**
  - 이 환경에는 현재 LaTeX이 설치되어 있지 않다.
  - `apt-get install texlive-xetex texlive-latex-extra texlive-lang-korean texlive-science latexmk fonts-noto-cjk` 등으로 설치를 시도한다.
  - 막히면 `tectonic` 등 대안을 시도한다.
  - **두 판 모두 경고 없이(최소한 undefined reference / overfull box 심각 건 없이) PDF로 컴파일**되어야 한다.
  - 최종 PDF는 `paper_tex/en/main.pdf`, `paper_tex/ko/main.pdf`로 둔다.
  - 끝내 컴파일할 수 없으면 그 사실과 원인을 보고한다. 컴파일되지 않은 소스를 "투고 가능"이라고 보고하지 않는다.

---

## 5. 품질 기준

- **사실성**
  - 모든 수치는 `results/` 산출물로 추적 가능해야 한다.
  - 과장하지 않는다. FPRM이 지는 결과도 그대로 쓴다. 이 논문의 핵심 메시지는 "조건부 유효"다.
- **공정성 서술**: 다음을 본문에 명시한다.
  - "최강 베이스라인"은 셀별 사후(oracle) 선택이라 FPRM에 불리한 비교다.
  - 원 논문 코드와 대조하지 못했다.
  - seed를 5개(ablation 3개)로 줄였다.
- **원 논문과의 관계**: 원 논문의 기여를 정확히 소개하고, 원 논문 수치(Table 1, Fig. 3 등)는 출처를 밝혀 인용한다. 원 논문 그림을 복제하지 않는다.
- **용어 일관성**: 시나리오 ID(S1–S8), 기법명, 데이터셋명을 영문판·국문판·`REPORT.md`에서 같게 쓴다.
- **국문판은 기계 번역투가 아닌 자연스러운 학술 한국어**로 쓴다(~이다/~한다 체).
- **검수**: 완성 후 각 판을 처음부터 다시 읽고 다음을 점검해 수정한다.
  - 수치 불일치
  - 참조되지 않은 그림·표
  - 깨진 `\ref`·인용
  - 그림 가독성(컴파일된 PDF를 이미지로 렌더링해 직접 확인)
  - 영문·국문 간 내용 불일치

---

## 6. 진행 방식과 산출물

1. 자료 정독: `REPORT.md`, 원 논문, 결과 CSV.
2. 영문판 개요와 그림·표 목록 확정.
3. 논문용 그림·표 생성.
4. 영문판 작성 → 컴파일 → 검수.
5. 국문판 작성(영문판 확정 후 번역 및 현지화) → 컴파일 → 검수.
6. arXiv 업로드 묶음 생성.

- 단계마다 의미 있는 단위로 커밋하고 지정된 브랜치에 push한다.
- LaTeX 빌드 부산물(`.aux`, `.log` 등)은 `.gitignore`에 넣는다. 최종 PDF와 업로드 묶음은 커밋한다.

**최종 보고**는 대화창에 짧게 한다. 다음을 포함한다.

- 두 PDF의 경로와 쪽수
- 사용한 그림·표 목록
- 확인 사항:
  - 소속 표기
  - 코드 공개 URL
  - arXiv의 AI 공동저자 정책
  - 국문판 arXiv 컴파일 방식(XeLaTeX 소스 vs PDF 업로드)
  - 카테고리 제안: cs.LG primary, eess.SP / physics.data-an cross-list

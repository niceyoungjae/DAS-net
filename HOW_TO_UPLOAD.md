# 📤 GitHub 업로드 가이드 (한국어)

이 폴더를 GitHub에 올리는 방법을 단계별로 안내합니다.

---

## ⚡ 빠른 업로드 (5분 컷)

### 방법 A: GitHub Desktop (가장 쉬움)

1. **GitHub Desktop** 실행
2. **File → Add Local Repository** 클릭
3. **Choose...** → `C:\Users\USER\Desktop\DAS-Net_GitHub` 선택
4. **"This is not a Git repository"** 경고 뜨면 → **"create a repository"** 클릭
5. Name: `DAS-net` 입력 → **Create Repository**
6. **Publish repository** 버튼 클릭
   - ⚠️ **"Keep this code private"** 체크 해제 (public으로!)
   - GitHub에 자동 푸시됨
7. 완료! 브라우저에서 https://github.com/niceyoungjae/DAS-net 확인

### 방법 B: 커맨드 라인 (Git 익숙한 경우)

```bash
cd C:\Users\USER\Desktop\DAS-Net_GitHub

# Git 초기화
git init
git add .
git commit -m "Initial release: DAS-Net code, results, and documentation"

# GitHub에 새 repository 만들기 (https://github.com/new)
# Repository name: DAS-net
# Description: A Lightweight Dynamic Convolution Network with Attention Gates and Deep Supervision for UAV Semantic Segmentation
# Public, NO README/LICENSE/.gitignore (이미 있음)

# 원격 연결 + 푸시
git remote add origin https://github.com/niceyoungjae/DAS-net.git
git branch -M main
git push -u origin main
```

---

## 📋 업로드 전 체크리스트

### ✅ 반드시 확인
- [ ] **개인정보 없음**: 이메일, 전화번호 외 민감 정보 X
- [ ] **manuscript 파일 없음**: docx, pdf 없음 (DAS-Net_Project 폴더 따로)
- [ ] **체크포인트 없음**: .pth 파일 없음 (gitignore에 의해 자동 제외)
- [ ] **데이터 없음**: 실제 이미지 파일 없음
- [ ] **API 키 / 비밀번호 없음**: config에 secret X

### 🟢 포함된 것 (공개 OK)
- 코드 (model, scripts, configs, dataset loaders)
- 결과 CSV (재현성 검증용)
- 논문 figure PNG (저자 본인 결과물)
- README.md, LICENSE, requirements.txt
- 문서 (INSTALLATION, USAGE, REPRODUCIBILITY)

---

## 🎯 GitHub Repository 추천 설정

### Repository Description
```
A Lightweight Dynamic Convolution Network with Attention Gates and Deep Supervision for UAV Semantic Segmentation | Applied Sciences 2026 | 1.66M params, 113 FPS A6000 / 26 FPS Jetson AGX Orin
```

### Topics (Settings → 우측 ⚙️ → Topics 추가)
```
anti-uav
semantic-segmentation
lightweight-network
dynamic-convolution
attention-gate
deep-supervision
unet
jetson
edge-ai
pytorch
real-time
remote-sensing
```

### Website
```
https://www.mdpi.com/journal/applsci  (Applied Sciences journal)
```

또는 paper Accept 후 paper URL 추가

---

## 📤 추가 권장 작업

### 1. Pre-trained Checkpoints 공개 (선택사항)

체크포인트 (~2 GB) 는 GitHub에 직접 못 올림. 대안:

#### 옵션 A: Hugging Face Hub (추천)
```bash
pip install huggingface_hub

# Hugging Face 계정 생성: https://huggingface.co/join
huggingface-cli login

# 업로드
huggingface-cli repo create niceyoungjae/DAS-Net --type model
cd checkpoints/
git lfs install
git clone https://huggingface.co/niceyoungjae/DAS-Net
mv dasnet/ thindyunet/ ... DAS-Net/
cd DAS-Net && git add . && git commit -m "Initial checkpoints" && git push
```

#### 옵션 B: Google Drive 공유 링크
1. Google Drive에 `checkpoints.zip` 업로드 (~2 GB)
2. 공유 → 링크 복사
3. README.md에 추가:
   ```
   ### Pre-trained Checkpoints
   Download from: [Google Drive](your-link-here)
   ```

#### 옵션 C: GitHub Release (작은 파일만)
- 파일 크기 < 2 GB만 가능
- Releases → "Draft a new release" → 체크포인트 zip 첨부

### 2. GitHub Actions (선택사항)
자동 테스트, 자동 deploy 등. 처음에는 안 해도 됨.

### 3. Issue Template (선택사항)
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/question.md`
- `.github/ISSUE_TEMPLATE/reproducibility.md`

---

## 🎓 Paper Accept 후 업데이트할 것

Accept 받으면:

### 1. README.md 업데이트
- "Under review" → "Published in *Applied Sciences*"
- BibTeX citation에 실제 페이지, DOI 추가
- 뱃지 추가: `[![Paper](https://img.shields.io/badge/DOI-10.3390%2Fappxx-blue)](https://doi.org/10.3390/appxx)`

### 2. CITATION.cff 업데이트
```yaml
preferred-citation:
  type: article
  doi: "10.3390/app..."  # 실제 DOI
  volume: ...
  issue: ...
  year: 2026
  pages: ...
```

### 3. GitHub Release 만들기
- "Draft a new release"
- Tag: `v1.0.0`
- Title: `DAS-Net v1.0 — Applied Sciences 2026 Release`
- Body: 논문 abstract + 주요 결과

---

## 🔗 유용한 링크

- **GitHub Desktop 다운로드**: https://desktop.github.com/
- **Git 설치**: https://git-scm.com/downloads
- **GitHub New Repo**: https://github.com/new
- **Hugging Face**: https://huggingface.co/
- **MDPI Author Services**: https://www.mdpi.com/authors

---

## 🚨 자주 발생하는 문제

### "files too large to push"
→ 체크포인트 파일이 포함됐을 가능성
→ `.gitignore`가 제대로 동작하는지 확인
→ `git rm --cached <large-file>` 로 제외

### "push permission denied"
→ GitHub Personal Access Token 필요
→ Settings → Developer settings → Personal access tokens
→ Generate new token (classic) → Scopes: `repo` 체크

### "private repo만 만들 수 있음"
→ Free 계정도 public repo 무제한 가능
→ Repository → Settings → General → Visibility → Change visibility → Public

---

## ✅ 최종 점검

업로드 후 https://github.com/niceyoungjae/DAS-net 방문해서:

- [ ] README.md 잘 보임 (figures 포함)
- [ ] License 표시됨 (MIT)
- [ ] 코드 폴더 구조 정상
- [ ] requirements.txt 보임
- [ ] CITATION.cff "Cite this repository" 버튼 작동
- [ ] Issues 탭 활성화
- [ ] Topics 표시됨

---

## 💡 첫 인상이 중요

GitHub은 reviewer / 면접관 / 박사 지원 시 가장 먼저 보는 곳입니다.

### 좋은 첫 인상 = 좋은 기회
- ⭐ README의 헤더 figure
- ⭐ 명확한 결과 표
- ⭐ 빠른 시작 가이드 (Quick Start)
- ⭐ 실행 가능한 예제 코드
- ⭐ 깔끔한 폴더 구조

→ 이미 모두 갖춰져 있습니다! 🎉

업로드만 하면 됩니다! 🚀

# Biz

소상공인을 위한 공공지원금 정보, 마케팅 팁을 모은 커뮤니티형 플랫폼입니다.

<p align="center">
  <a href="http://biz-fe-roan.vercel.app" target="_blank" rel="noopener noreferrer">
    BIZ 바로가기
  </a>
</p>

## 프로젝트 기간

2025-09-11 ~ 2026-01-30

## 설치 및 실행

### Frontend (Local)

```sh
git clone https://github.com/JangsaCartel/Biz-FE.git
cd Biz-FE

npm install
npm run dev
```

### Backend (Local)

```sh
git clone https://github.com/JangsaCartel/Biz-BE.git
cd Biz-BE

./gradlew clean build

docker run -d -p 8080:8080 --name jc-be-container jc-be
```

### AI (Local)

```sh
git clone https://github.com/JangsaCartel/Biz-AI.git
cd Biz-AI

pip install -r requirements.txt

source .venv/Scripts/activate
python -m uvicorn app.main:app --reload --port 8000 --env-file .env

deactivate
```

## 주요 링크

### Notion 문서

- [🌐 Notion 전체 문서 바로가기](https://minyoung06.notion.site/22ddb7a96218809fb629d23e2e593d49)

### 디자인 시안

- [🖌️ Figma 바로가기](https://www.figma.com/design/tY2YDD1KHi33lSAACC5laO/%EC%9E%A5%EC%82%AC%EB%AD%90%ED%95%98%EB%8B%88-?node-id=2-3&p=f&t=jbpdQ2MFxgZyoyUx-0)

### Swagger

- [⚙️ BE Swagger 바로가기](http://13.158.142.74:8080/swagger-ui.html)

- [🧠 AI Swagger 바로가기](http://13.158.142.74:8000/docs)

## 🛠️ 기술 스택

### Frontend

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Vue.js 3](https://img.shields.io/badge/Vue.js-4FC08D?style=for-the-badge&logo=vue.js&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Sass](https://img.shields.io/badge/Sass-CC6699?style=for-the-badge&logo=sass&logoColor=white)
![Vue Router](https://img.shields.io/badge/Vue%20Router-4FC08D?style=for-the-badge&logo=vue.js&logoColor=white)
![Pinia](https://img.shields.io/badge/Pinia-FFE600?style=for-the-badge&logo=pinia&logoColor=black)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)
![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)
![Prettier](https://img.shields.io/badge/Prettier-F7B93E?style=for-the-badge&logo=prettier&logoColor=black)

### Backend

![Java 17](https://img.shields.io/badge/Java%2017-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![Spring MVC](https://img.shields.io/badge/Spring%20MVC-6DB33F?style=for-the-badge&logo=spring&logoColor=white)
![MyBatis](https://img.shields.io/badge/MyBatis-000000?style=for-the-badge&logo=mybatis&logoColor=white)
![Spring JDBC](https://img.shields.io/badge/Spring%20JDBC-6DB33F?style=for-the-badge&logo=spring&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![HikariCP](https://img.shields.io/badge/HikariCP-000000?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Spring Security](https://img.shields.io/badge/Spring%20Security-6DB33F?style=for-the-badge&logo=springsecurity&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![OAuth2](https://img.shields.io/badge/OAuth2-000000?style=for-the-badge&logo=oauth&logoColor=white)
![Gradle](https://img.shields.io/badge/Gradle-02303A?style=for-the-badge&logo=gradle&logoColor=white)
![Swagger 2](https://img.shields.io/badge/Swagger%202-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)
![SLF4J](https://img.shields.io/badge/SLF4J-000000?style=for-the-badge)
![Log4j2](https://img.shields.io/badge/Log4j2-D22128?style=for-the-badge&logo=apache&logoColor=white)
![Tomcat](https://img.shields.io/badge/Tomcat-F8DC75?style=for-the-badge&logo=apachetomcat&logoColor=black)
![AWS S3](https://img.shields.io/badge/AWS%20S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)

### AI

![Python 3.12](https://img.shields.io/badge/Python%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64?style=for-the-badge&logo=python&logoColor=black)
![Black](https://img.shields.io/badge/Black-000000?style=for-the-badge&logo=python&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

### Infra & DevOps

![AWS EC2](https://img.shields.io/badge/AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

### Design

![Figma](https://img.shields.io/badge/Figma-F24E1E?style=for-the-badge&logo=figma&logoColor=white)

### Tools

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)
![Eclipse](https://img.shields.io/badge/Eclipse-2C2255?style=for-the-badge&logo=eclipseide&logoColor=white)

### Collaboration

![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)

### Documentation & Version Control

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

## 🚀 배포 구조

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS EC2](https://img.shields.io/badge/AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS%20S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](#)

### 개요

> **Frontend** → Vercel **자동 배포**

> **Backend** → Docker 이미지 빌드 → **Docker Hub 푸시** → EC2에서 `docker-compose` **배포**

> **Cache/세션** → **Redis (Docker)**

## 👥 팀원

|               [양민영](https://github.com/Minyoung06)               |               [윤예림](https://github.com/StarWhale0w0)               |               [전영태](https://github.com/JeonYeongtae)               |
| :-----------------------------------------------------------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| <img src="https://github.com/Minyoung06.png" width="200px" alt="_"> | <img src="https://github.com/StarWhale0w0.png" width="200px" alt="_"> | <img src="https://github.com/JeonYeongtae.png" width="200px" alt="_"> |
|                              FullStack                              |                               FullStack                               |                               FullStack                               |

## 역할

| 이름   | 작업한 페이지 / 기능                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 그 외 작업 사항                                                                                                                                                                                                                                                                                                                              |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 양민영 | • 정책 페이지 (외부 API 요청 로직 구현, 정책 리스트 화면 구현)<br>• 정책 상세 페이지 (외부 API 요청으로 상세 내용 가져오기, 화면 구현)<br>• AI 분석 페이지 (게시글 기준으로 핫 토픽 10개 선정 로직 구현, 워드 클라우드 구현, 화면 구현)<br>• AI 채팅 페이지 (Gemini 활용한 챗봇 구현, 기본 질문/자유 질문 응답 구현, 화면 구현)<br>• 알림 기능 (알림 화면 구현, SSE 활용하여 실시간 알림 기능 구현)                                                                         | • CI 프론트에 구축(Github Actions 활용, discord에 알림 자동화)<br>• PR, Issue template 적용<br>• 피그마 디자인 참여<br>• VSCode 프론트엔드 초기 세팅<br>• Eclipse 백엔드 초기 세팅<br>• ERD 구현 및 최적화<br>• AWS EC2에 BE, AI, DB, Redis 설정 및 배포 <br>• Vercel 사용하여 FE 배포<br>• ReadME 작성, Notion 문서화, 시스템 아키텍처 구현 |
| 윤예림 | • 게시판 전반 (자유/정보/지역별 게시판 목록 및 상세 페이지, 댓글 및 좋아요 등 게시판 부가기능, 글쓰기 페이지 화면 구현 및 개발)<br>• 글쓰기 페이지 (게시글 CRUD 관련 API 엔드포인트 구현, 화면 구현)<br>• 공용 컴포넌트 구현 (AppPagination, ModalDialog, RegionDropdowns 등 재사용 가능한 UI 개발)<br>• 지도 페이지 구현 (Kakao Map API 연동, 위치 기반 필터링 및 검색 백엔드 API 연동)<br>• 스케줄러 구현(데이터베이스 주기적 관리 및 삭제 로직 구현, 시스템 자동화 처리) | • 피그마 디자인 참여<br>• Eclipse 백엔드 초기 세팅<br>• ERD 구현 및 최적화<br>• AWS 기초 세팅 <br>• 공용 DB 구현<br>• ReadME 작성, Notion 문서화, 시스템 아키텍처 구현                                                                                                                                                                       |
| 전영태 | • 로그인/회원가입 페이지(Kakao OAuth 2.0 API 연동 및 액세스 토큰 발급 로직, JWT 발급/검증 로직 구현, Bearer Token 인증 방식 설정 및 보안 로직 적용)<br>• 마이 페이지(정보 조회 및 수정, 회원 탈퇴, 보안 관련 로직 개선, 화면 구현)<br>• Kakao Developers 애플리케이션 등록 및 설정(API Key 발급, 플랫폼 및 Redirect URI 구성)                                                                                                                                               | • 피그마 디자인 담당<br>• 공통 UI 컴포넌트 스타일 리팩토링 및 유지보수 추가 구현 <br>• Notion 문서화                                                                                                                                                                                                                                         |

---

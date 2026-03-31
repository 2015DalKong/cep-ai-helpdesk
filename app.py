import streamlit as st
import os

# 필수 모듈
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# [설정] API Key (환경변수 설정)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(
    page_title="이젬코 CEP AI 헬프데스크", 
    page_icon="🤖", 
    layout="wide" # 화면을 넓게 씁니다 (기본값은 'centered')
)
st.title("🤖 이젬코-CEP 매뉴얼 챗봇")
# --- 🎯 [UI 디자인 추가 1] 소개 및 비즈니스 인사이트 ---
st.markdown("""
**진행하였던 프로젝트 솔루션 중 MES, POP 솔루션의 기본 매뉴얼을 이용하여 챗봇을 만들었습니다.**
""")
st.caption("💡 **KJY 포트폴리오 Insight:** 사내규정, 자사 특화의 ERP 매뉴얼, 테이블 레이아웃을 타겟으로 RAG 시스템을 구축할 시, 실무에 즉시 투입 가능한 **강력한 자체 생산성 도우미**를 확보할 수 있습니다.")

# --- 📚 [UI 디자인 추가 2] 모듈 및 질문 예시 (접었다 펼치기 패널) ---
with st.expander("📖 지원 모듈 및 추천 질문 보기", expanded=False): # 처음엔 닫혀있도록 설정 (깔끔하게)
    st.markdown("""
    **[적용 대상 모듈]**
    * 공통/기준정보, 영업관리, 생산관리(MES/POP), 자재관리, 품질관리 등
    
    **[이런 질문을 해보세요!]**
    * "생산계획 프로세스를 알려주세요."
    * "자재 입고 처리는 어떻게 해?"
    * "공정 불량 발생 시 품질 검사 메뉴에서 어떻게 처리해?"
    """)
st.divider() # 시각적으로 깔끔하게 선 하나 그어주기
# ----------------------------------------------------
@st.cache_resource
def build_knowledge_base():
    # 로컬 임베딩 모델 정의 (함수 최상단으로 이동)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    if not os.path.exists("faiss_index"):
        with st.spinner("📚 전체 매뉴얼 지식화 중... (로컬 임베딩 사용)"):
            # data 폴더 내의 모든 PDF 로드
            loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
            texts = text_splitter.split_documents(documents)
            
            # 인덱스 생성 및 로컬 저장
            vector_db = FAISS.from_documents(texts, embeddings)
            vector_db.save_local("faiss_index")
            return vector_db
    else:
        # 이미 생성된 인덱스가 있다면 불러오기
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 1. 지식 베이스 로드
vector_db = build_knowledge_base()

# 2. 체인 설계
template = """이젬코 CEP 솔루션 최고 전문가입니다. 
제공된 매뉴얼 내용을 바탕으로 사용자의 질문에 전문적이고 명확하게 답변하세요.

[답변 원칙]
1. 단순한 '메뉴 목차'나 '페이지 번호'가 아닌, 실제 [업무 로직], [처리 방법], [화면 설명]을 중심으로 답변하세요.
2. 검색된 내용 중 '목차(Table of Contents)'로 보이는 부분은 과감히 무시하고 본문 내용을 우선하세요.
3. 제공된 매뉴얼에서 답변을 찾을 수 없다면, "제공된 매뉴얼에서는 해당 내용을 찾을 수 없습니다."라고 명확히 답변하세요.

매뉴얼 내용:
{context}

질문: {question}
"""
prompt_tmpl = ChatPromptTemplate.from_template(template)

# LLM 설정 (HTTPS 경로 명시)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",  # <- 여기에 -preview 가 반드시 들어가야 합니다!
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # 또는 스튜디오에서 확인하셨던 안정된 버전
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

retriever = vector_db.as_retriever(search_kwargs={"k": 20})

# RAG 파이프라인 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_tmpl
    | llm
    | StrOutputParser()
)
# 3. 채팅 인터페이스
if "messages" not in st.session_state:
    # 텅 빈 배열([]) 대신, 첫 인사말을 미리 넣어둡니다.
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 이젬코 CEP 시스템 전문가입니다. 어떤 업무 로직이 궁금하신가요?\n\n*(💡 아래 입력창에 **'생산계획 프로세스를 알려주세요'** 라고 입력해 보세요!)*"}
    ]

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("예: 자재 입고 처리는 어떻게 해?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("매뉴얼 분석 중..."):
            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

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

st.set_page_config(page_title="CEP 지능형 헬프데스크", layout="wide")
st.title("🤖 이젬코-CEP 매뉴얼 챗봇")

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
template = """당신은 15년 차 경력의 이젬코 CEP 및 SAP B1 솔루션 최고 전문가입니다. 
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
    st.session_state.messages = []

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("이젬코 CEP 업무 로직을 물어보세요."):
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

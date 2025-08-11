import os
import hashlib
import time
from parsing_utils import split_chunks
from collections import deque

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# 기존 LLM imports 유지
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 전역 캐시
_vector_store_cache = {}
_pinecone_client = None

class BufferMemory:
    """기존 BufferMemory - 성능 최적화"""
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
    
    def append(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})
    
    def get_formatted_history(self):
        if not self.history:
            return ""
        return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history])

def get_file_hash(file_path: str) -> str:
    """파일 해시 - 캐싱으로 최적화"""
    try:
        # 파일 수정 시간과 크기로 간단한 해시 생성 (더 빠름)
        stat = os.stat(file_path)
        quick_hash = f"{stat.st_size}_{int(stat.st_mtime)}"
        return hashlib.md5(quick_hash.encode()).hexdigest()
    except:
        return hashlib.md5(file_path.encode()).hexdigest()

def get_qdrant_client():
    """Pinecone 클라이언트로 변경"""
    global _pinecone_client
    if _pinecone_client is None:
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                print("[PINECONE_API_KEY 환경변수가 설정되지 않았습니다]")
                return None
            
            _pinecone_client = Pinecone(api_key=api_key)
            print("[Pinecone 클라이언트 초기화 완료]")
        except Exception as e:
            print(f"[Pinecone 연결 실패: {e}]")
            return None
    return _pinecone_client

def get_llm(tokens=256):
    """LLM 캐싱 - qwen2.5:7b 모델 유지"""
    llm = "qwen2.5:7b-instruct"
    try:
        # 최신 버전에서는 num_predict 사용
        return ChatOllama(
            model=llm, 
            temperature=0.2, 
            num_predict=tokens
        )
    except TypeError:
        try:
            # 구버전에서는 max_tokens 사용
            return ChatOllama(
                model=llm, 
                temperature=0.2, 
                max_tokens=tokens
            )
        except TypeError:
            # 둘 다 안 되면 기본 설정만
            return ChatOllama(
                model=llm, 
                temperature=0.2, 
            )

def get_user_hash(user_id: str) -> str:
    """사용자 ID를 안전한 해시로 변환"""
    return hashlib.md5(user_id.encode()).hexdigest()[:8]

def create_user_index(user_id: str):
    """사용자별 인덱스 생성"""
    pc = get_qdrant_client()  # 함수명 유지하지만 Pinecone 반환
    if not pc:
        return None
    
    user_hash = get_user_hash(user_id)
    index_name = f"user-{user_hash}"
    
    try:
        # 기존 인덱스 확인
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"[새 인덱스 생성: {index_name}]")
            
            pc.create_index(
                name=index_name,
                dimension=1024,  # BGE-M3 차원
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # 인덱스 준비 대기
            time.sleep(3)
            print(f"[인덱스 생성 완료: {index_name}]")
        else:
            print(f"[기존 인덱스 사용: {index_name}]")
        
        return index_name
        
    except Exception as e:
        print(f"[인덱스 생성 실패: {e}]")
        return None

def data_to_vectorstore(file_path: str, user_id: str = "default_user"):
    """벡터스토어 - Pinecone으로 변경"""
    
    if not PINECONE_AVAILABLE:
        print("[Pinecone 사용 불가 - None 반환]")
        return None
    
    # 캐시 확인
    file_hash = get_file_hash(file_path)
    cache_key = f"{user_id}_{file_hash}"
    
    if cache_key in _vector_store_cache:
        print(f"[캐시에서 벡터스토어 로드: {file_path}]")
        return _vector_store_cache[cache_key]
    
    pc = get_qdrant_client()  # 실제로는 Pinecone 클라이언트
    if pc is None:
        return None
    
    # 사용자별 인덱스 생성/확인
    index_name = create_user_index(user_id)
    if not index_name:
        return None
    
    try:
        # Pinecone 인덱스 연결
        index = pc.Index(index_name)
        
        # 파일 해시로 기존 문서 확인
        query_result = index.query(
            vector=[0.0] * 1024,  # 더미 벡터
            filter={"file_hash": file_hash},
            top_k=1,
            include_metadata=True
        )
        
        # 기존 문서가 있으면 벡터스토어만 생성
        if query_result.matches:
            print(f"[기존 문서 사용: {file_hash}]")
            vector_store = PineconeVectorStore(
                index=index,
                embedding=OllamaEmbeddings(model="bge-m3:567m"),
                text_key="text"
            )
            _vector_store_cache[cache_key] = vector_store
            return vector_store
    
    except Exception as e:
        print(f"[기존 문서 확인 실패: {e}]")
    
    # 새 문서 처리
    print(f"[새 문서 임베딩 시작: {file_path}]")
    
    try:
        # 문서 청킹
        documents = split_chunks(file_path)
        if not documents:
            return None
        
        # 메타데이터에 file_hash와 user_id 추가
        for doc in documents:
            doc.metadata.update({
                "file_hash": file_hash,
                "user_id": user_id,
                "file_name": os.path.basename(file_path)
            })
        
        # 벡터스토어 생성 및 문서 추가
        vector_store = PineconeVectorStore(
            index=index,
            embedding=OllamaEmbeddings(model="bge-m3:567m"),
            text_key="text"
        )
        
        print("임베딩 및 저장 중...")
        vector_store.add_documents(documents)
        
        # 캐시에 저장
        _vector_store_cache[cache_key] = vector_store
        print(f"[벡터스토어 캐싱 완료: {len(documents)}개 문서]")
        
        return vector_store
        
    except Exception as e:
        print(f"[벡터스토어 생성 실패: {e}]")
        return None

def smart_determine_params(query: str):
    """개선된 파라미터 결정 - 답변 품질 고려"""
    query_lower = query.lower()
    
    # 복잡한 작업 (더 많은 토큰과 문서 필요)
    if any(keyword in query for keyword in ['보고서', '발표', 'ppt', '분석', '비교', '평가']):
        return 1000, 4096, "복합분석"
    
    # 퀴즈/문제 (적당한 양의 문서, 구조화된 답변)
    elif any(keyword in query for keyword in ['퀴즈']):
        return 1000, 2048, "퀴즈"
    
    # 요약 (전체적인 이해 필요)
    elif any(keyword in query for keyword in ['요약', '정리', '핵심', '간추']):
        return 1000, 1024, "요약"
    
    # 구체적 질문 (관련성 높은 문서 필요)
    elif any(keyword in query for keyword in ['어떻게', '왜', '무엇', '언제', '어디서', '누가']):
        return 1000, 2048, "구체적질문"
    
    # 일반 질문
    else:
        return 100, 512, "일반"

def create_enhanced_prompt(query: str, combined_text: str, history: str, task_type: str):
    """향상된 프롬프트 생성"""
    
    base_context = f"""다음은 사용자와의 대화 기록입니다:
{history}

참고할 문서 내용:
{combined_text}

사용자 질문: {query}
"""

    if task_type == "복합분석":
        return f"""{base_context}

위 문서를 바탕으로 사용자의 요청에 대해 체계적이고 전문적으로 답변해주세요.
- 한국어로 답변합니다.
- 문서의 핵심 내용을 충분히 반영하세요
- 논리적 구조로 답변을 구성하세요
- 구체적인 근거와 예시를 포함하세요
- 문서에 없는 내용은 추측하지 마세요
- 반복되는 말을 하지 마세요

답변:"""

    elif task_type == "퀴즈":
        return f"""{base_context}

문서 내용을 기반으로 퀴즈를 생성해주세요.
- 문서의 핵심 개념과 중요한 정보를 중심으로 구성하세요
- 다양한 유형의 문제를 포함하세요 (객관식, 단답형, 서술형 등)
- 사용자의 요청이 없다면 문제는 5개만 생성합니다.
- 각 문제에 대한 정답과 해설을 제공하세요
- 난이도를 적절히 조절하세요
- 반복되는 말을 하지 마세요

퀴즈:"""

    elif task_type == "요약":
        return f"""{base_context}

문서의 주요 내용을 체계적으로 요약해주세요.
- 핵심 주제와 요점을 명확히 정리하세요
- 중요도에 따라 내용을 구조화하세요
- 구체적인 데이터나 예시가 있다면 포함하세요
- 간결하지만 포괄적으로 정리하세요
- 반복되는 말을 하지 마세요

요약:"""

    elif task_type == "구체적질문":
        return f"""{base_context}

문서를 참조하여 구체적이고 정확하게 답변해주세요.
- 문서에서 관련된 정보를 찾아 근거로 제시하세요
- 단계별로 명확하게 설명하세요
- 문서에 명시되지 않은 부분은 "문서에서 확인할 수 없습니다"라고 명시하세요
- 가능한 한 구체적인 예시나 수치를 포함하세요
- 반복되는 말을 하지 마세요

답변:"""

    else:  # 일반
        return f"""{base_context}

문서를 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
- 문서의 관련 내용을 충분히 활용하세요
- 명확하고 이해하기 쉽게 설명하세요
- 추가적인 맥락이나 배경 정보도 제공하세요
- 문서 범위를 벗어나는 추측은 피하세요
- 답변은 너무 길지 않게 해주세요

답변:"""

def question_answer_with_memory(file_path: str, query: str, memory: BufferMemory, user_id: str = "default_user", tokens=256) -> str:
    """개선된 메인 함수 - user_id 파라미터 추가"""
    
    # 1. 향상된 파라미터 결정
    k, optimized_tokens, task_type = smart_determine_params(query)
    final_tokens = max(tokens, optimized_tokens) if tokens != 256 else optimized_tokens
    
    print(f"[작업 유형: {task_type}, 문서 수: {k}, 토큰: {final_tokens}]")
    
    # 2. 벡터스토어 로드 (user_id 추가)
    vector_store = data_to_vectorstore(file_path, user_id)
    
    # 3. 벡터스토어 실패 시 즉시 폴백
    if vector_store is None:
        print("[벡터스토어 없음 - 직접 파일 읽기]")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)
    
    # 4. 향상된 벡터 검색 (사용자별 필터링 추가)
    try:
        file_hash = get_file_hash(file_path)
        
        # 사용자 문서만 검색
        docs = vector_store.similarity_search(
            query, 
            k=k,
            filter={"file_hash": file_hash, "user_id": user_id}
        )
        
        # 검색 결과가 부족한 경우 추가 검색
        if len(docs) < k//2:
            # 쿼리를 단순화해서 다시 검색
            simple_query = " ".join(query.split()[:3])
            additional_docs = vector_store.similarity_search(
                simple_query, 
                k=k,
                filter={"file_hash": file_hash, "user_id": user_id}
            )
            # 중복 제거하면서 합치기
            seen = set()
            all_docs = []
            for doc in docs + additional_docs:
                doc_hash = hash(doc.page_content[:100])
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    all_docs.append(doc)
                if len(all_docs) >= k:
                    break
            docs = all_docs
        
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 텍스트가 너무 짧은 경우 추가 문서 검색
        if len(combined_text) < 500:
            extra_docs = vector_store.similarity_search(
                "", 
                k=5,
                filter={"file_hash": file_hash, "user_id": user_id}
            )
            for doc in extra_docs:
                if doc not in docs:
                    docs.append(doc)
                    combined_text += "\n\n" + doc.page_content
                if len(combined_text) > 1000:
                    break
        
    except Exception as e:
        print(f"[검색 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)
    
    # 5. 향상된 프롬프트로 LLM 호출
    history = memory.get_formatted_history()
    prompt = create_enhanced_prompt(query, combined_text, history, task_type)
    
    # 6. LLM 호출
    try:
        llm = get_llm(final_tokens)
        answer = llm.invoke(prompt).content
        
        # 메모리 업데이트
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        print(f"[LLM 실패: {e}] - 폴백 모드")
        return handle_fallback_mode(file_path, query, memory, final_tokens, task_type)

def handle_fallback_mode(file_path: str, query: str, memory: BufferMemory, tokens: int, task_type: str = "일반") -> str:
    """개선된 폴백 모드"""
    
    try:
        # 파일 크기에 따라 읽을 양 조절
        file_size = os.path.getsize(file_path)
        
        if file_size > 50000:  # 50KB 이상
            read_size = 15000  # 15KB만 읽기
        elif file_size > 20000:  # 20KB 이상
            read_size = 10000  # 10KB만 읽기
        else:
            read_size = file_size  # 전체 읽기
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(read_size)
        
        history = memory.get_formatted_history()
        
        # 향상된 폴백 프롬프트
        prompt = create_enhanced_prompt(query, content, history, task_type)
        
        # 토큰 수 조절 (폴백 모드에서는 약간 줄임)
        fallback_tokens = min(tokens, 2048)
        
        llm = get_llm(fallback_tokens)
        answer = llm.invoke(prompt).content
        
        memory.append(query, answer)
        return answer
        
    except Exception as e:
        return f"죄송합니다. 문서 처리 중 오류가 발생했습니다: {str(e)}\n\n다시 시도해 주시거나 문서 형식을 확인해 주세요."

def clear_cache():
    """캐시 초기화"""
    global _vector_store_cache, _pinecone_client
    _vector_store_cache.clear()
    _pinecone_client = None
    print("[캐시 초기화 완료]")

def get_cache_stats():
    """캐시 상태 확인"""
    return {
        "vector_stores": len(_vector_store_cache),
        "client_connected": _pinecone_client is not None
    }

# Django 호환성 유지
if __name__ == "__main__":
    # 테스트 코드
    memory = BufferMemory()
    user_id = "test_user_123"
    
    print("=== Pinecone 마이그레이션 테스트 ===")
    
    import time
    start_time = time.time()
    result1 = question_answer_with_memory("temp/sample.txt", "이 문서의 주요 내용은 무엇인가요?", memory, user_id)
    print(f"첫 번째 답변: {result1[:200]}...")
    
    start_time = time.time()
    result2 = question_answer_with_memory("temp/sample.txt", "구체적으로 어떤 기술이 사용되었나요?", memory, user_id)
    print(f"두 번째 답변: {result2[:200]}...")
    
    print(f"\n캐시 상태: {get_cache_stats()}")
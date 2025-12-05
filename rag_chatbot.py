import os
import pandas as pd
import time
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from typing import List
import re

# ===== 알레르기 탐지 클래스 =====

class AllergenDetector:
    """알레르기 유발 성분을 LLM으로 탐지하는 클래스 (100% LLM 기반)"""
    
    # 19가지 법정 알레르기 유발 식품 (참조용)
    ALLERGEN_CATEGORIES = [
        "알류", "우유", "메밀", "땅콩", "대두", "밀", "잣", "호두",
        "게", "새우", "오징어", "고등어", "조개류", "복숭아", "토마토",
        "닭고기", "돼지고기", "쇠고기", "아황산류"
    ]
    
    def __init__(self, llm):
        """
        Args:
            llm: Google Generative AI LLM 인스턴스
        """
        self.llm = llm
    
    def detect(self, ingredients_text: str) -> List[str]:
        """
        LLM을 사용하여 알레르기 유발 성분을 직접 분석하고 탐지.
        
        Args:
            ingredients_text: 재료 텍스트
            
        Returns:
            탐지된 알레르기 성분 리스트
        """
        if pd.isna(ingredients_text) or str(ingredients_text).strip() == "":
            return []
        
        prompt = f"""당신은 식품 알레르기 전문가입니다.
다음 재료 목록을 세밀하게 분석하여 19가지 법정 알레르기 항목에 해당하는 성분이 있는지 모두 찾아주세요.

**19가지 법정 알레르기 항목:**
알류, 우유, 메밀, 땅콩, 대두, 밀, 잣, 호두, 게, 새우, 오징어, 고등어, 조개류, 복숭아, 토마토, 닭고기, 돼지고기, 쇠고기, 아황산류

**재료 목록:**
{ingredients_text}

**분석 지침:**
1. 직접 재료: 명시된 재료가 알레르기 항목인 경우 (예: 계란, 우유, 밀가루)
2. 가공식품 원재료: 가공식품에 포함된 알레르기 성분 분석
   - 마요네즈, 머랭, 케이크 → 알류
   - 치즈, 생크림, 버터, 휘핑크림 → 우유
   - 간장, 된장, 고추장, 쌈장, 두부 → 대두
   - 빵, 빵가루, 파스타, 면류, 튀김가루 → 밀
   - 햄, 베이컨, 소시지 → 돼지고기 (주로)
3. 유사 표현: 다양한 표현 체크
   - "달걀"과 "계란" → 알류
   - "소고기"와 "쇠고기" → 쇠고기
   - "콩"과 "대두" → 대두
4. 소스와 양념: 원재료 추적
   - 크림소스, 화이트소스 → 우유
   - 장류(간장, 된장, 고추장) → 대두
5. 확실한 경우만 포함하고, 애매한 경우는 제외

**답변 형식:**
- 발견된 알레르기 항목만 쉼표로 구분하여 나열
- 없으면 '없음'이라고만 답변
- 설명이나 부가 정보 없이 항목명만 출력

예시 답변: 알류, 우유, 대두, 밀
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            result = result.strip()
            
            if result == "없음" or not result:
                return []
            
            # 쉼표로 구분된 알레르기 항목 파싱
            detected = [item.strip() for item in result.split(",")]
            # 19가지 법정 항목만 필터링
            valid_allergens = [a for a in detected if a in self.ALLERGEN_CATEGORIES]
            return sorted(valid_allergens)
            
        except Exception as e:
            print(f"⚠️  LLM 알레르기 탐지 오류: {e}")
            return []
    
    def detect_batch(self, ingredients_list: List[str], batch_size: int = 10) -> List[List[str]]:
        """
        여러 재료를 배치로 처리하여 속도 향상.
        
        Args:
            ingredients_list: 재료 텍스트 리스트
            batch_size: 한 번에 처리할 재료 수
            
        Returns:
            각 재료의 알레르기 성분 리스트
        """
        results = []
        
        for i in range(0, len(ingredients_list), batch_size):
            batch = ingredients_list[i:i+batch_size]
            
            # 배치 프롬프트 생성
            batch_prompt = """당신은 식품 알레르기 전문가입니다.
다음 여러 레시피의 재료 목록을 분석하여 각각의 알레르기 성분을 찾아주세요.

**19가지 법정 알레르기 항목:**
알류, 우유, 메밀, 땅콩, 대두, 밀, 잣, 호두, 게, 새우, 오징어, 고등어, 조개류, 복숭아, 토마토, 닭고기, 돼지고기, 쇠고기, 아황산류

**분석 지침:**
- 가공식품 원재료도 확인 (마요네즈→알류, 간장→대두, 빵가루→밀 등)
- 확실한 경우만 포함

**재료 목록:**
"""
            for idx, ingredients in enumerate(batch, 1):
                batch_prompt += f"\n[{idx}] {ingredients[:200]}"  # 너무 길면 200자로 제한
            
            batch_prompt += """

**답변 형식 (각 번호마다 한 줄씩):**
[1] 알류, 우유, 대두
[2] 없음
[3] 닭고기, 대두
"""
            
            try:
                # RPM 제한 준수: 10 RPM 이하 유지 (6초 대기)
                time.sleep(6.0)
                
                response = self.llm.invoke(batch_prompt)
                result = response.content if hasattr(response, 'content') else str(response)
                
                # 결과 파싱
                batch_results = []
                for line in result.strip().split('\n'):
                    if line.strip().startswith('['):
                        # [숫자] 이후의 내용 추출
                        content = line.split(']', 1)[1].strip() if ']' in line else ''
                        if content == '없음' or not content:
                            batch_results.append([])
                        else:
                            detected = [item.strip() for item in content.split(',')]
                            valid = [a for a in detected if a in self.ALLERGEN_CATEGORIES]
                            batch_results.append(sorted(valid))
                
                # 배치 크기만큼 결과가 없으면 빈 리스트로 채우기
                while len(batch_results) < len(batch):
                    batch_results.append([])
                
                results.extend(batch_results[:len(batch)])
                
            except Exception as e:
                print(f"⚠️  배치 {i//batch_size + 1} 처리 오류: {e}")
                # 오류 시 더 긴 대기 후 재시도
                time.sleep(10)
                # 오류 시 각 항목에 빈 리스트 추가
                results.extend([[] for _ in batch])
        
        return results


class FeedbackStore:
    """사용자 피드백을 저장하고 관리하는 클래스"""
    
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self.feedbacks = self._load_feedbacks()
    
    def _load_feedbacks(self):
        """저장된 피드백 로드"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"positive": [], "negative": []}
        return {"positive": [], "negative": []}
    
    def _save_feedbacks(self):
        """피드백 저장"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedbacks, f, ensure_ascii=False, indent=2)
    
    def add_feedback(self, query: str, recipe_title: str, recipe_url: str, is_positive: bool):
        """피드백 추가"""
        feedback_entry = {
            "query": query,
            "recipe_title": recipe_title,
            "recipe_url": recipe_url,
            "timestamp": time.time()
        }
        
        if is_positive:
            self.feedbacks["positive"].append(feedback_entry)
            print("\n✅ 긍정적인 피드백이 저장되었습니다. 다음 검색부터 반영됩니다!")
        else:
            self.feedbacks["negative"].append(feedback_entry)
            print("\n❌ 부정적인 피드백이 저장되었습니다. 다음 검색 시 가중치가 낮아집니다.")
        
        self._save_feedbacks()
    
    def get_recipe_score(self, recipe_url: str) -> float:
        """레시피의 피드백 점수 계산 (긍정: +1, 부정: -1)"""
        positive_count = sum(1 for fb in self.feedbacks["positive"] if fb["recipe_url"] == recipe_url)
        negative_count = sum(1 for fb in self.feedbacks["negative"] if fb["recipe_url"] == recipe_url)
        return positive_count - negative_count


class AllergenExtractor:
    """질문에서 알레르기 정보를 추출하는 클래스"""
    
    # 알레르기 키워드 매핑 (19가지 법정 알레르기 유발 식품)
    ALLERGEN_KEYWORDS = {
        "알류": ["알", "알류", "계란", "달걀", "에그", "난"],
        "우유": ["우유", "유제품", "치즈", "버터", "생크림", "요구르트", "모짜렐라", "파마산", "크림"],
        "메밀": ["메밀", "메밀국수", "소바"],
        "땅콩": ["땅콩", "피넛"],
        "대두": ["대두", "두부", "된장", "간장", "콩", "두유", "콩나물"],
        "밀": ["밀", "밀가루", "빵가루", "면", "국수", "파스타", "우동", "라면"],
        "잣": ["잣"],
        "호두": ["호두"],
        "게": ["게", "킹크랩", "대게"],
        "새우": ["새우", "크래미", "새우젓"],
        "오징어": ["오징어", "갑오징어", "한치"],
        "고등어": ["고등어", "삼치", "꽁치"],
        "조개류": ["조개", "조개류", "굴", "홍합", "바지락", "모시조개", "전복", "소라"],
        "복숭아": ["복숭아"],
        "토마토": ["토마토", "방울토마토"],
        "닭고기": ["닭고기", "치킨", "닭", "닭날개", "닭다리", "닭가슴살"],
        "돼지고기": ["돼지고기", "삼겹살", "목살", "등심", "앞다리", "뒷다리", "베이컨", "햄", "돼지"],
        "쇠고기": ["쇠고기", "소고기", "한우", "등심", "안심", "우둔", "양지", "사태", "소"],
        "아황산류": ["아황산", "아황산류", "이산화황"]
    }
    
    @staticmethod
    def extract_from_query(query: str) -> List[str]:
        """질문에서 알레르기 재료 추출 (형식: 알레르기: 알류, 우유 / 질문: ...)"""
        allergens = []
        
        # "알레르기:" 형식으로 시작하는지 확인
        allergen_pattern = r'알레르기\s*:\s*([^/]+)'
        match = re.search(allergen_pattern, query)
        
        if match:
            # 쉼표로 구분된 알레르기 항목 추출
            allergen_text = match.group(1).strip()
            allergen_items = [item.strip() for item in allergen_text.split(',')]
            
            # 19가지 법정 알레르기 항목만 허용
            valid_allergens = list(AllergenExtractor.ALLERGEN_KEYWORDS.keys())
            for item in allergen_items:
                if item in valid_allergens:
                    allergens.append(item)
        
        return allergens
    
    @staticmethod
    def remove_allergen_keywords(query: str, allergens: List[str]) -> str:
        """질문에서 알레르기 관련 키워드 제거 (형식: 알레르기: ... / 질문: ...)"""
        # "알레르기: ... /" 부분 완전 제거
        cleaned_query = re.sub(r'알레르기\s*:[^/]*/\s*', '', query)
        
        # "질문:" 텍스트 제거
        cleaned_query = re.sub(r'질문\s*:\s*', '', cleaned_query)
        
        # 연속된 공백 제거
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query


class FeedbackRetriever(BaseRetriever):
    """피드백과 알레르기를 고려한 Re-ranking Retriever"""
    
    base_retriever: object  # VectorStoreRetriever
    feedback_store: FeedbackStore
    allergen_detector: object  # AllergenDetector
    vectorstore: object  # FAISS VectorStore
    boost_factor: float = 0.3
    user_allergens: List[str] = []
    allergen_penalty_weight: float = 0.8  # 알레르기 페널티 가중치
    
    def set_allergens(self, allergens: List[str]):
        """사용자 알레르기 정보 설정"""
        self.user_allergens = allergens
    
    def _calculate_allergen_similarity(self, doc: Document) -> float:
        """알레르기 재료와의 유사도 계산 (0~1, 높을수록 알레르기 재료 포함 가능성 높음)"""
        if not self.user_allergens:
            return 0.0
        
        # 메타데이터에 저장된 알레르기 정보 우선 사용
        doc_allergens = doc.metadata.get('allergens', [])
        
        if doc_allergens:
            # 메타데이터의 알레르기 정보와 비교
            for allergen in self.user_allergens:
                if allergen in doc_allergens:
                    return 1.0  # 정확한 매칭
            return 0.0  # 알레르기 없음
        
        # 메타데이터가 없는 경우 (레거시 데이터 대비)
        # 알레르기 카테고리명이 직접 포함되어 있는지 체크
        content_lower = doc.page_content.lower()
        for allergen in self.user_allergens:
            if allergen in content_lower:
                return 1.0  # 알레르기 성분 포함
        
        return 0.0
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """피드백과 알레르기를 고려하여 문서 검색 및 Re-ranking"""
        # 기본 검색으로 더 많은 후보 가져오기
        original_k = self.base_retriever.search_kwargs.get('k', 1)
        search_k = max(20, original_k * 20)  # 더 많은 후보 검색
        
        # 유사도 점수와 함께 문서 가져오기
        try:
            # FAISS의 similarity_search_with_score 사용
            candidates_with_scores = self.vectorstore.similarity_search_with_score(query, k=search_k)
        except:
            # 폴백: 일반 검색 사용
            self.base_retriever.search_kwargs = {'k': search_k}
            candidates = self.base_retriever.invoke(query)
            candidates_with_scores = [(doc, 0.0) for doc in candidates]
        
        # 각 문서에 대해 최종 점수 계산 및 알레르기 필터링
        scored_docs = []
        filtered_count = 0
        no_allergen_info_count = 0
        warning_docs = []  # 알레르기 정보 없는 문서들 (경고용)
        
        for doc, base_score in candidates_with_scores:
            recipe_url = doc.metadata.get('source', '')
            allergens = doc.metadata.get('allergens', [])
            
            # 알레르기 체크 (메타데이터 기반)
            allergen_similarity = self._calculate_allergen_similarity(doc)
            has_allergen = allergen_similarity >= 1.0  # 정확한 매칭만 제외
            
            # 알레르기가 있는 청크는 완전히 제외
            if has_allergen:
                filtered_count += 1
                continue
            
            # 알레르기 정보가 없는 경우 경고 플래그 추가
            has_no_allergen_info = (self.user_allergens and not allergens)
            if has_no_allergen_info:
                no_allergen_info_count += 1
            
            # 1. 기본 코사인 유사도 점수 (FAISS는 L2 distance 반환, 작을수록 유사)
            # L2 distance를 유사도로 변환 (거리가 작을수록 점수 높게)
            similarity_score = 1.0 / (1.0 + base_score)
            
            # 2. 피드백 점수
            feedback_score = self.feedback_store.get_recipe_score(recipe_url)
            feedback_boost = feedback_score * self.boost_factor
            
            # 3. 다양성 점수 (랜덤 노이즈 추가로 매번 다른 레시피 추천)
            import random
            diversity_noise = random.uniform(-0.05, 0.05)  # ±5% 랜덤 변동
            
            # 4. 최종 점수 = 기본 유사도 + 피드백 보너스 + 다양성 노이즈
            total_score = similarity_score + feedback_boost + diversity_noise
            
            scored_docs.append({
                'doc': doc,
                'base_score': similarity_score,
                'feedback_score': feedback_score,
                'diversity_noise': diversity_noise,
                'total_score': total_score,
                'has_allergen': False,
                'no_allergen_info': has_no_allergen_info  # 경고 플래그
            })
        
        # 최종 점수로 재정렬
        scored_docs.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 디버깅 정보 출력
        if self.user_allergens:
            print(f"\n📊 알레르기 필터링 결과:")
            print(f"  - 검색된 청크: {len(candidates_with_scores)}개")
            print(f"  - 알레르기 매칭: {filtered_count}개 제외")
            print(f"  - 알레르기 정보 없음: {no_allergen_info_count}개 (⚠️ 경고 포함)")
            print(f"  - 남은 청크: {len(scored_docs)}개")
            
            if scored_docs:
                print(f"\n📊 검색 결과 상위 3개 (알레르기 필터링 후):")
                for i, item in enumerate(scored_docs[:3], 1):
                    title = item['doc'].metadata.get('title', '제목 없음')[:50]
                    allergens = item['doc'].metadata.get('allergens', [])
                    allergen_str = ', '.join(allergens) if allergens else '⚠️ 정보 없음'
                    warning = " [주의필요]" if item.get('no_allergen_info', False) else ""
                    print(f"  {i}. {title}{warning}")
                    print(f"     유사도: {item['base_score']:.3f} | "
                          f"알레르기: {allergen_str} | "
                          f"최종: {item['total_score']:.3f}")
            
            # 필터링된 문서 예시 출력 (상위 3개)
            if filtered_count > 0:
                print(f"\n🚫 필터링된 문서 예시 (상위 3개):")
                filtered_shown = 0
                for doc, base_score in candidates_with_scores:
                    allergen_similarity = self._calculate_allergen_similarity(doc)
                    if allergen_similarity >= 1.0:
                        title = doc.metadata.get('title', '제목 없음')[:50]
                        allergens = doc.metadata.get('allergens', [])
                        matched = [a for a in allergens if a in self.user_allergens]
                        print(f"  {filtered_shown + 1}. {title}")
                        print(f"     알레르기: {', '.join(allergens)} | 매칭: {', '.join(matched)}")
                        filtered_shown += 1
                        if filtered_shown >= 3:
                            break
        
        # 결과가 충분하지 않으면 경고
        if len(scored_docs) < original_k:
            print(f"\n⚠️ 알레르기 조건을 만족하는 레시피가 부족합니다. ({len(scored_docs)}/{original_k}개)")
        
        return [item['doc'] for item in scored_docs[:original_k]]


class VectorStoreBuilder:
    """벡터 저장소를 구축하는 클래스 (100% LLM 기반 알레르기 탐지)"""
    
    def __init__(self, embeddings, persist_directory: str = "faiss_recipe_index"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        # LLM 초기화 (Gemini 2.5 Flash)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )
        # 알레르기 탐지기 초기화 (LLM 전달)
        self.allergen_detector = AllergenDetector(self.llm)
    
    def build_from_csv(self, csv_path: str):
        """CSV 파일에서 벡터 저장소 구축"""
        print(f"레시피 데이터 로딩 중... ({csv_path})")
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        print(f"레시피 개수: {len(df)}")
        
        # 문서 생성
        documents = self._create_documents_from_dataframe(df)
        print(f"생성된 문서 개수: {len(documents)}")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        print(f"분할된 청크 개수: {len(split_docs)}")
        
        # 벡터 저장소 생성 (체크포인트 사용)
        vectorstore = self._build_with_checkpoints(split_docs)
        
        # 저장
        vectorstore.save_local(self.persist_directory)
        print(f"벡터 저장소 저장 완료: {self.persist_directory}")
        
        return vectorstore
    
    def _create_documents_from_dataframe(self, df):
        """DataFrame에서 Document 객체 리스트 생성 (LLM 배치 기반 알레르기 분석)"""
        documents = []
        total_recipes = len(df)
        
        # 체크포인트 파일 경로
        checkpoint_file = "allergen_detection_checkpoint.json"
        
        print("🤖 LLM 배치 처리로 모든 레시피의 알레르기 정보를 추출하는 중...")
        print("⚡ 1000개씩 초대용량 배치 + 2초 대기로 속도 3배 향상!")
        print("⏳ 예상 시간: 하루 약 250,000개 처리 가능 (250 RPD × 1000개/배치)")
        print("💾 진행 상황은 자동으로 저장됩니다 (중단 시 이어서 진행 가능)")
        
        # 기존 체크포인트 로드
        all_allergens = []
        start_idx = 0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    all_allergens = checkpoint_data.get('allergens', [])
                    start_idx = len(all_allergens)
                    print(f"📂 체크포인트 발견: {start_idx}개 레시피 이미 처리됨 (이어서 진행)")
            except Exception as e:
                print(f"⚠️  체크포인트 로드 실패: {e}, 처음부터 시작합니다.")
                all_allergens = []
                start_idx = 0
        
        # 남은 재료 텍스트 수집 (start_idx부터)
        ingredients_list = []
        if start_idx < len(df):
            for idx in range(start_idx, len(df)):
                row = df.iloc[idx]
                ingredients = row['재료']
                if pd.notna(ingredients) and str(ingredients).strip():
                    ingredients_list.append(str(ingredients))
                else:
                    ingredients_list.append("")
        
        if ingredients_list:
            # 배치 처리로 알레르기 탐지
            print(f"\n📦 남은 {len(ingredients_list)}개 레시피를 1000개씩 배치 처리 중...")
            batch_size = 1000
            remaining_allergens = []
            
            for batch_start in range(0, len(ingredients_list), batch_size):
                batch_end = min(batch_start + batch_size, len(ingredients_list))
                batch = ingredients_list[batch_start:batch_end]
                
                # 배치 처리
                batch_results = self.allergen_detector.detect_batch(batch, batch_size=len(batch))
                remaining_allergens.extend(batch_results)
                
                # 진행 상황 표시
                current_total = start_idx + len(remaining_allergens)
                if current_total % 1000 == 0 or batch_end == len(ingredients_list):
                    print(f"진행 중: {current_total}/{total_recipes} 레시피 처리 완료 ({current_total*100//total_recipes}%)")
                
                # 체크포인트 저장 (1000개마다)
                if len(remaining_allergens) % 1000 == 0 or batch_end == len(ingredients_list):
                    temp_allergens = all_allergens + remaining_allergens
                    try:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump({'allergens': temp_allergens, 'total': total_recipes}, f, ensure_ascii=False)
                        print(f"  💾 체크포인트 저장: {len(temp_allergens)}개 처리 완료")
                    except Exception as e:
                        print(f"  ⚠️  체크포인트 저장 실패: {e}")
            
            all_allergens.extend(remaining_allergens)
        else:
            print(f"✅ 모든 레시피({len(all_allergens)}개) 이미 처리 완료!")
        
        # Document 생성
        print("\n📄 Document 객체 생성 중...")
        for idx, row in df.iterrows():
            if (idx + 1) % 1000 == 0 or idx == 0:
                print(f"진행 중: {idx + 1}/{total_recipes} 레시피 처리 중...")
            
            content = (
                f"요리 제목: {row['제목']}\n\n"
                f"재료: {row['재료']}\n\n"
                f"인분: {row.get('인분', '')}\n\n"
                f"소개: {row['인트로']}\n\n"
                f"조리 순서: {row['조리순서']}"
            )
            
            # 배치 처리로 얻은 알레르기 정보 사용
            detected_allergens = all_allergens[idx] if idx < len(all_allergens) else []
            
            # 결과 샘플 출력 (처음 10개만)
            if idx < 10 and detected_allergens:
                print(f"  └─ [{row['제목'][:20]}...] 알레르기: {', '.join(detected_allergens)}")
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "index": row.get('index', ''),
                    "종류별": row.get('종류별', ''),
                    "상황별": row.get('상황별', ''),
                    "재료별": row.get('재료별', ''),
                    "방법별": row.get('방법별', ''),
                    "title": row['제목'],
                    "source": row['url'],
                    "조회수": row.get('조회수', ''),
                    "셰프": row.get('셰프', ''),
                    "servings": row.get('인분', ''),
                    "조리시간": row.get('조리시간', ''),
                    "난이도": row.get('난이도', ''),
                    "ingredients": row['재료'],
                    "인트로": row.get('인트로', ''),
                    "조리순서": row.get('조리순서', ''),
                    "해시태그": row.get('해시태그', ''),
                    "AI리뷰요약": row.get('AI리뷰요약', ''),
                    "allergens": detected_allergens  # 알레르기 정보 추가
                }
            ))
        
        print(f"✅ 총 {total_recipes}개 레시피 로드 완료 (알레르기 정보 포함)!")
        return documents
    
    def _build_with_checkpoints(self, split_docs):
        """체크포인트를 활용하여 벡터 저장소 구축"""
        # 체크포인트 찾기
        checkpoint_files = []
        for filename in os.listdir('.'):
            if filename.startswith(f"{self.persist_directory}_checkpoint_"):
                try:
                    num = int(filename.split('_')[-1])
                    checkpoint_files.append((num, filename))
                except:
                    pass
        
        vectorstore = None
        start_index = 0
        
        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            start_index, checkpoint_path = checkpoint_files[0]
            print(f"\n🔄 체크포인트 발견! {checkpoint_path}에서 재시작합니다.")
            print(f"   이미 처리된 청크: {start_index:,}개")
            print(f"   남은 청크: {len(split_docs) - start_index:,}개")
            vectorstore = FAISS.load_local(
                checkpoint_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("\n🆕 체크포인트가 없습니다. 처음부터 시작합니다.")
        
        # 임베딩 생성
        if start_index < len(split_docs):
            remaining_docs = len(split_docs) - start_index
            print(f"\n임베딩을 생성하고 벡터 저장소를 구축 중입니다...")
            print(f"총 {remaining_docs:,}개의 청크를 처리합니다. (전체: {len(split_docs):,}개)")
            
            batch_size = 1000
            save_interval = 10000
            start_time = time.time()
            
            for i in range(start_index, len(split_docs), batch_size):
                batch = split_docs[i:i+batch_size]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    vectorstore.add_documents(batch)
                
                processed = min(i + batch_size, len(split_docs))
                progress = (processed / len(split_docs)) * 100
                elapsed_time = time.time() - start_time
                processed_in_session = processed - start_index
                
                if processed < len(split_docs) and processed_in_session > 0:
                    estimated_total_time = (elapsed_time / processed_in_session) * remaining_docs
                    remaining_time = estimated_total_time - elapsed_time
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)
                    print(
                        f"진행: {processed:,}/{len(split_docs):,} ({progress:.1f}%) - "
                        f"예상 남은 시간: {remaining_hours}시간 {remaining_minutes}분"
                    )
                else:
                    total_time = elapsed_time
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    print(
                        f"✅ 완료: {processed:,}/{len(split_docs):,} (100%) - "
                        f"총 소요 시간: {hours}시간 {minutes}분"
                    )
                
                # 체크포인트 저장
                if (processed > start_index and 
                    processed % save_interval == 0 and 
                    processed < len(split_docs)):
                    checkpoint_path = f"{self.persist_directory}_checkpoint_{processed}"
                    print(f"💾 중간 저장 중... ({processed:,}개 처리됨) → {checkpoint_path}")
                    try:
                        vectorstore.save_local(checkpoint_path)
                        print(f"✅ 중간 저장 완료!")
                    except Exception as e:
                        print(f"⚠️ 중간 저장 실패: {e}")
        
        return vectorstore


class RagChatbot:
    """RAG 기반 레시피 챗봇 메인 클래스 (100% LLM 기반 알레르기 탐지)"""
    
    def __init__(self, csv_path: str = "data/recipe_main_5.csv", 
                 faiss_index_path: str = "faiss_recipe_index"):
        self.csv_path = csv_path
        self.faiss_index_path = faiss_index_path
        
        # 환경 변수 로드
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 임베딩 모델 로드
        print("로컬 임베딩 모델을 로드 중입니다...")
        self.embeddings = self._load_embeddings()
        
        # LLM 초기화
        print("LLM을 초기화 중입니다...")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        # 서비스 클래스 초기화
        self.allergen_detector = AllergenDetector(self.llm)
        
        # 벡터 저장소 로드 또는 구축
        if os.path.exists(faiss_index_path):
            print("기존 벡터 저장소를 로드 중입니다...")
            self.vectorstore = FAISS.load_local(
                faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("벡터 저장소를 구축합니다...")
            builder = VectorStoreBuilder(self.embeddings, faiss_index_path)
            self.vectorstore = builder.build_from_csv(csv_path)
        
        # Feedback 저장소 및 Retriever 초기화
        print("피드백 저장소를 초기화 중입니다...")
        self.feedback_store = FeedbackStore()
        
        base_retriever = self.vectorstore.as_retriever(search_kwargs={'k': 3})  # 3개 레시피 추천
        self.retriever = FeedbackRetriever(
            base_retriever=base_retriever,
            feedback_store=self.feedback_store,
            allergen_detector=self.allergen_detector,
            vectorstore=self.vectorstore,  # FAISS vectorstore 전달
            boost_factor=0.3,
            allergen_penalty_weight=0.8  # 알레르기 페널티 가중치
        )
        
        # RAG 체인 및 LLM 전용 체인 구성
        self.rag_chain = self._build_rag_chain()
        self.llm_only_chain = self._build_llm_only_chain()
        
        print("\n✅ RAG 챗봇이 준비되었습니다!")
    
    def _load_embeddings(self):
        """임베딩 모델 로드"""
        model_name = "jhgan/ko-sroberta-multitask"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def _format_docs_with_info(self, docs):
        """검색된 문서에 알레르기 정보 추가"""
        formatted_docs = []
        
        for doc in docs:
            content = doc.page_content
            servings = doc.metadata.get('servings', 2)
            title = doc.metadata.get('title', '')
            source = doc.metadata.get('source', '')
            
            # 알레르기 정보 (메타데이터에서 직접 가져오기)
            allergens = doc.metadata.get('allergens', [])
            allergen_text = ", ".join(allergens) if allergens else "없음"
            
            # 메타데이터 추가
            metadata_text = f"\n\n=== 레시피 정보 ===\n"
            metadata_text += f"제목: {title}\n"
            metadata_text += f"출처: {source}\n"
            
            if doc.metadata.get('조회수'):
                metadata_text += f"조회수: {doc.metadata['조회수']}\n"
            if doc.metadata.get('셰프'):
                metadata_text += f"작성자: {doc.metadata['셰프']}\n"
            if doc.metadata.get('조리시간'):
                metadata_text += f"조리시간: {doc.metadata['조리시간']}\n"
            if doc.metadata.get('난이도'):
                metadata_text += f"난이도: {doc.metadata['난이도']}\n"
            
            metadata_text += f"\n인분: {servings}\n"
            metadata_text += f"알레르기 유발 성분: {allergen_text}\n"
            
            cooking_steps = doc.metadata.get('조리순서', '')
            if cooking_steps and cooking_steps != 'nan':
                metadata_text += f"\n조리 순서:\n{cooking_steps}\n"
            
            full_content = content + metadata_text
            formatted_docs.append(full_content)
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_docs)
    
    def _build_rag_chain(self):
        """RAG 체인 구성"""
        prompt_template = """당신은 친절한 요리 전문가입니다. 사용자의 질문에 레시피 데이터베이스를 기반으로 답변해주세요.

**답변 지침:**
1. 검색된 여러 레시피 중에서 질문에 **가장 적합한 레시피 1개**를 선택하여 추천하세요
2. 사용자가 알레르기 정보를 명시한 경우, 해당 재료가 포함되지 않은 레시피만 추천해야 합니다
3. 레시피의 주요 특징, 재료, 조리법을 간단히 설명하세요
4. 알레르기 유발 성분과 영양정보를 반드시 포함하세요
5. 사용자가 알레르기를 명시했다면, 해당 재료가 포함되지 않았음을 확실히 안내하세요
6. **⚠️ 알레르기 정보가 없는 레시피는 "알레르기 정보 미확인" 경고를 반드시 표시하세요**
7. 출처 URL을 제공하세요
8. 친근하고 따뜻한 톤으로 답변하세요

**중요:** 검색된 레시피 중 사용자 질문과 **제목이 정확히 일치**하거나 가장 유사한 레시피를 우선 선택하세요.
예: "무국"을 물었다면 "무나물볶음"보다 "무국 끓이기"를 선택

**검색된 레시피 정보:**
{context}

**사용자 질문:**
{question}

**답변 형식 예시:**
[레시피명]을 추천합니다!
(알레르기 정보가 있었다면) ✅ 이 레시피에는 [알레르기 재료]가 포함되어 있지 않습니다.

[간단한 설명]

**재료:**
- ...

**조리 방법:**
1. ...

**중요 정보:**
- 알레르기: ...
- 영양정보(1인분): ...
- 출처: [URL]

[간단한 조리 설명]

다른 궁금한 점이 있으시면 언제든 물어보세요!

답변:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # 검색과 질문을 분리하는 체인
        def retrieve_with_query(inputs):
            """검색 쿼리로 문서 검색, 원본 질문은 유지"""
            search_query = inputs.get("search_query", inputs.get("question"))
            question = inputs.get("question")
            docs = self.retriever.invoke(search_query)
            return {
                "context": self._format_docs_with_info(docs),
                "question": question
            }
        
        rag_chain = (
            retrieve_with_query
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _build_llm_only_chain(self):
        """LLM만 사용하는 체인 구성 (RAG 없음)"""
        prompt_template = """당신은 친절한 요리 전문가입니다. 사용자의 질문에 답변해주세요.

**답변 지침:**
1. 일반적인 요리 지식을 바탕으로 답변하세요
2. 친근하고 따뜻한 톤으로 답변하세요

**사용자 질문:**
{question}

답변:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        llm_only_chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return llm_only_chain
    
    def run(self, mode='compare'):
        """챗봇 실행
        
        Args:
            mode: 'rag' (RAG만 사용), 'llm' (LLM만 사용), 'compare' (비교 모드)
        """
        print("\n" + "="*60)
        if mode == 'compare':
            print("📊 RAG vs LLM 비교 모드")
            print("="*60)
            print("각 질문에 대해 RAG와 LLM 전용 모드의 답변을 비교합니다.")
        elif mode == 'rag':
            print("🔍 RAG 모드 (레시피 데이터베이스 기반 답변)")
        else:
            print("🤖 LLM 전용 모드 (일반 지식 기반 답변)")
        print("="*60)
        print("\n질문을 입력하세요 (종료하려면 'exit' 입력).\n")
        
        while True:
            try:
                question = input("\n질문: ")
                if question is None:
                    continue
                question = question.strip()
            except EOFError:
                print("\n입력 스트림이 종료되었습니다. 챗봇을 종료합니다.")
                break
            except KeyboardInterrupt:
                print("\n\n챗봇을 종료합니다.")
                break
            
            if question.lower() == 'exit':
                print("챗봇을 종료합니다.")
                break
            
            # 빈 질문 처리
            if not question:
                print("⚠️ 질문을 입력해주세요.")
                continue
            
            # 질문에서 알레르기 정보 추출
            user_allergens = AllergenExtractor.extract_from_query(question)
            search_query = question
            
            if user_allergens:
                self.retriever.set_allergens(user_allergens)
                # 알레르기 키워드를 제거한 깨끗한 검색 쿼리 생성
                search_query = AllergenExtractor.remove_allergen_keywords(question, user_allergens)
                print(f"\n🔍 알레르기 정보 감지: {', '.join(user_allergens)}")
                print(f"→ 해당 재료가 포함되지 않은 레시피를 찾습니다.")
                print(f"→ 검색 쿼리: \"{search_query}\"\n")
            else:
                self.retriever.set_allergens([])
            
            if mode == 'compare':
                # RAG 답변
                print("\n" + "="*60)
                print("🔍 RAG 모드 답변 (레시피 DB 기반):")
                print("="*60)
                start_time = time.time()
                rag_answer = self.rag_chain.invoke({"question": question, "search_query": search_query})
                rag_time = time.time() - start_time
                print(f"\n{rag_answer}")
                print(f"\n⏱️ 응답 시간: {rag_time:.2f}초")
                
                # LLM 전용 답변
                print("\n" + "="*60)
                print("🤖 LLM 전용 모드 답변 (일반 지식 기반):")
                print("="*60)
                start_time = time.time()
                llm_answer = self.llm_only_chain.invoke(question)
                llm_time = time.time() - start_time
                print(f"\n{llm_answer}")
                print(f"\n⏱️ 응답 시간: {llm_time:.2f}초")
                
                # 비교 요약
                print("\n" + "="*60)
                print("📊 비교 요약:")
                print("="*60)
                print(f"RAG 모드: 레시피 DB에서 검색된 실제 레시피 정보 제공")
                print(f"LLM 모드: 일반적인 요리 지식 기반 답변")
                print(f"응답 시간 차이: {abs(rag_time - llm_time):.2f}초")
                
            elif mode == 'rag':
                # RAG만 실행
                start_time = time.time()
                answer = self.rag_chain.invoke({"question": question, "search_query": search_query})
                elapsed_time = time.time() - start_time
                print(f"\n답변: {answer}")
                print(f"\n⏱️ 응답 시간: {elapsed_time:.2f}초")
                
            else:  # llm
                # LLM만 실행
                start_time = time.time()
                answer = self.llm_only_chain.invoke(question)
                elapsed_time = time.time() - start_time
                print(f"\n답변: {answer}")
                print(f"\n⏱️ 응답 시간: {elapsed_time:.2f}초")
            
            # 피드백 수집 (RAG 모드 또는 비교 모드일 때만)
            if mode in ['rag', 'compare']:
                print("\n" + "="*60)
                print("이 답변이 도움이 되었나요?")
                print("👍 좋아요 (1) | 👎 별로에요 (2) | ⏭️  건너뛰기 (Enter)")
                feedback_input = input("선택: ").strip()
                
                if feedback_input in ['1', '2']:
                    docs = self.retriever.invoke(search_query)
                    if docs:
                        doc = docs[0]
                        recipe_title = doc.metadata.get('title', '제목 없음')
                        recipe_url = doc.metadata.get('source', '')
                        is_positive = (feedback_input == '1')
                        self.feedback_store.add_feedback(question, recipe_title, recipe_url, is_positive)
                else:
                    print("피드백을 건너뛰었습니다.")


def main():
    """메인 함수"""
    print("\n" + "="*60)
    print("🍳 레시피 챗봇")
    print("="*60)
    print("\n모드를 선택하세요:")
    print("1. 비교 모드 (RAG vs LLM 전용) - 기본값")
    print("2. RAG 모드 (레시피 DB 기반)")
    print("3. LLM 전용 모드 (일반 지식 기반)")
    
    mode_input = input("\n선택 (1-3, Enter=비교모드): ").strip()
    
    if mode_input == '2':
        mode = 'rag'
    elif mode_input == '3':
        mode = 'llm'
    else:
        mode = 'compare'
    
    # 100% LLM 기반 알레르기 탐지
    print(f"\n⚙️ 알레르기 탐지 모드: 100% LLM 기반 분석 (최고 정확도)")

    
    chatbot = RagChatbot(
        csv_path="data/recipe_main_5.csv",
        faiss_index_path="faiss_recipe_index"
    )
    chatbot.run(mode=mode)


if __name__ == "__main__":
    main()

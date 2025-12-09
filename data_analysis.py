import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("📊 레시피 데이터 분석 리포트")
print("=" * 80)

# 1. 기본 데이터 정보
print("\n[1] 기본 데이터 정보")
print("-" * 80)

df = pd.read_csv('data/recipe_main_5.csv', encoding='utf-8', low_memory=False)
print(f"✅ 총 레시피 수: {len(df):,}개")
print(f"✅ 총 컬럼 수: {len(df.columns)}개")
print(f"\n📋 컬럼 목록:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 2. 결측치 분석
print("\n\n[2] 결측치 분석")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '결측 비율(%)': missing_pct
})
missing_df = missing_df[missing_df['결측치 수'] > 0].sort_values('결측치 수', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("✅ 결측치 없음")

# 3. 카테고리별 분포
print("\n\n[3] 카테고리별 레시피 분포")
print("-" * 80)

categories = ['종류별', '상황별', '재료별', '방법별']
for cat in categories:
    if cat in df.columns:
        print(f"\n📌 {cat} 분포:")
        value_counts = df[cat].value_counts().head(10)
        for idx, (value, count) in enumerate(value_counts.items(), 1):
            pct = (count / len(df) * 100)
            bar = '█' * int(pct / 2)
            print(f"  {idx:2d}. {str(value):20s} │ {count:6,}개 ({pct:5.2f}%) {bar}")

# 4. 난이도 분포
print("\n\n[4] 난이도 분포")
print("-" * 80)
if '난이도' in df.columns:
    difficulty = df['난이도'].value_counts()
    for level, count in difficulty.items():
        pct = (count / len(df) * 100)
        bar = '█' * int(pct / 5)
        print(f"  {str(level):10s} │ {count:6,}개 ({pct:5.2f}%) {bar}")

# 5. 조리시간 분석
print("\n\n[5] 조리시간 분석")
print("-" * 80)
if '조리시간' in df.columns:
    time_counts = df['조리시간'].value_counts().head(15)
    for time, count in time_counts.items():
        pct = (count / len(df) * 100)
        bar = '█' * int(pct / 2)
        print(f"  {str(time):20s} │ {count:6,}개 ({pct:5.2f}%) {bar}")

# 6. 인분 분석
print("\n\n[6] 인분 분석")
print("-" * 80)
if '인분' in df.columns:
    servings = df['인분'].value_counts().head(10)
    for serving, count in servings.items():
        pct = (count / len(df) * 100)
        bar = '█' * int(pct / 2)
        print(f"  {str(serving):10s} │ {count:6,}개 ({pct:5.2f}%) {bar}")

# 7. 조회수 분석
print("\n\n[7] 조회수 통계")
print("-" * 80)
if '조회수' in df.columns:
    views = pd.to_numeric(df['조회수'], errors='coerce')
    print(f"  평균 조회수:   {views.mean():,.0f}회")
    print(f"  중앙값:        {views.median():,.0f}회")
    print(f"  최소 조회수:   {views.min():,.0f}회")
    print(f"  최대 조회수:   {views.max():,.0f}회")
    print(f"  표준편차:      {views.std():,.0f}회")
    
    # 조회수 구간별 분포
    print(f"\n  📊 조회수 구간별 분포:")
    max_view = views.max()
    if max_view <= 1000:
        bins = [0, 200, 400, 600, 800, max_view + 1]
        labels = ['~200', '200~400', '400~600', '600~800', '800~']
    else:
        bins = [0, 1000, 5000, 10000, 50000, 100000, max_view + 1]
        labels = ['~1천', '1천~5천', '5천~1만', '1만~5만', '5만~10만', '10만~']
    
    views_binned = pd.cut(views, bins=bins, labels=labels)
    for label, count in views_binned.value_counts().sort_index().items():
        pct = (count / len(views) * 100)
        bar = '█' * int(pct / 2)
        print(f"    {str(label):10s} │ {count:6,}개 ({pct:5.2f}%) {bar}")

# 8. 셰프별 레시피 수
print("\n\n[8] 셰프별 레시피 수 (상위 15명)")
print("-" * 80)
if '셰프' in df.columns:
    chefs = df['셰프'].value_counts().head(15)
    for idx, (chef, count) in enumerate(chefs.items(), 1):
        pct = (count / len(df) * 100)
        bar = '█' * int(pct / 2)
        print(f"  {idx:2d}. {str(chef):30s} │ {count:5,}개 ({pct:5.2f}%) {bar}")

# 9. 재료 분석
print("\n\n[9] 재료 분석")
print("-" * 80)
if '재료' in df.columns:
    # 재료 길이 통계
    ingredient_lengths = df['재료'].str.len()
    print(f"  평균 재료 텍스트 길이: {ingredient_lengths.mean():.0f}자")
    print(f"  중앙값:               {ingredient_lengths.median():.0f}자")
    print(f"  최소 길이:            {ingredient_lengths.min():.0f}자")
    print(f"  최대 길이:            {ingredient_lengths.max():.0f}자")
    
    # 재료 개수 추정 (쉼표 기준)
    ingredient_counts = df['재료'].str.count(',') + 1
    print(f"\n  평균 재료 개수 (추정): {ingredient_counts.mean():.1f}개")
    print(f"  중앙값:               {ingredient_counts.median():.0f}개")
    print(f"  최소 개수:            {ingredient_counts.min():.0f}개")
    print(f"  최대 개수:            {ingredient_counts.max():.0f}개")

# 10. 알레르기 분석 (FAISS 인덱스에서)
print("\n\n[10] 알레르기 정보 분석")
print("-" * 80)

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("  FAISS 인덱스 로딩 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embedments': True}
    )
    
    vectorstore = FAISS.load_local(
        "faiss_recipe_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 샘플 문서에서 알레르기 정보 수집
    print("  알레르기 정보 수집 중...")
    sample_docs = vectorstore.similarity_search("", k=1000)
    
    allergen_counter = Counter()
    docs_with_allergens = 0
    docs_without_allergens = 0
    
    for doc in sample_docs:
        allergens = doc.metadata.get('allergens', [])
        if allergens:
            docs_with_allergens += 1
            for allergen in allergens:
                allergen_counter[allergen] += 1
        else:
            docs_without_allergens += 1
    
    print(f"\n  ✅ 알레르기 정보 있는 문서: {docs_with_allergens}개")
    print(f"  ⚠️  알레르기 정보 없는 문서: {docs_without_allergens}개")
    
    if allergen_counter:
        print(f"\n  📊 알레르기 성분별 빈도 (상위 19개):")
        for idx, (allergen, count) in enumerate(allergen_counter.most_common(19), 1):
            pct = (count / len(sample_docs) * 100)
            bar = '█' * int(pct / 2)
            print(f"    {idx:2d}. {allergen:10s} │ {count:5,}회 ({pct:5.2f}%) {bar}")

except Exception as e:
    print(f"  ⚠️  FAISS 인덱스 분석 실패: {e}")
    print("  (벡터 저장소가 아직 구축되지 않았을 수 있습니다)")

# 11. 해시태그 분석
print("\n\n[11] 해시태그 분석 (상위 20개)")
print("-" * 80)
if '해시태그' in df.columns:
    all_hashtags = []
    for tags in df['해시태그'].dropna():
        if isinstance(tags, str):
            # #으로 시작하는 단어들 추출
            hashtags = [tag.strip() for tag in str(tags).split() if tag.startswith('#')]
            all_hashtags.extend(hashtags)
    
    if all_hashtags:
        hashtag_counter = Counter(all_hashtags)
        for idx, (tag, count) in enumerate(hashtag_counter.most_common(20), 1):
            pct = (count / len(all_hashtags) * 100)
            bar = '█' * int(pct)
            print(f"  {idx:2d}. {tag:20s} │ {count:5,}회 ({pct:5.2f}%) {bar}")
    else:
        print("  해시태그 정보 없음")

# 12. AI 리뷰 요약 분석
print("\n\n[12] AI 리뷰 요약 정보")
print("-" * 80)
if 'AI리뷰요약' in df.columns:
    has_review = df['AI리뷰요약'].notna().sum()
    no_review = df['AI리뷰요약'].isna().sum()
    pct_has = (has_review / len(df) * 100)
    pct_no = (no_review / len(df) * 100)
    
    print(f"  ✅ AI 리뷰 있음: {has_review:,}개 ({pct_has:.2f}%)")
    print(f"  ❌ AI 리뷰 없음: {no_review:,}개 ({pct_no:.2f}%)")

# 13. 피드백 데이터 분석
print("\n\n[13] 사용자 피드백 분석")
print("-" * 80)

try:
    with open('feedback_data.json', 'r', encoding='utf-8') as f:
        feedback_data = json.load(f)
    
    positive_count = len(feedback_data.get('positive', []))
    negative_count = len(feedback_data.get('negative', []))
    total_feedback = positive_count + negative_count
    
    if total_feedback > 0:
        pos_pct = (positive_count / total_feedback * 100)
        neg_pct = (negative_count / total_feedback * 100)
        
        print(f"  총 피드백 수: {total_feedback}개")
        print(f"  👍 긍정: {positive_count}개 ({pos_pct:.1f}%)")
        print(f"  👎 부정: {negative_count}개 ({neg_pct:.1f}%)")
        
        # 피드백 많은 레시피
        if positive_count > 0:
            print(f"\n  📌 긍정 피드백 받은 레시피:")
            positive_recipes = Counter([fb['recipe_title'] for fb in feedback_data['positive']])
            for idx, (recipe, count) in enumerate(positive_recipes.most_common(5), 1):
                print(f"    {idx}. {recipe[:50]} ({count}회)")
    else:
        print("  ⚠️  피드백 데이터 없음")
        
except Exception as e:
    print(f"  ⚠️  피드백 파일 없음 또는 읽기 실패: {e}")

# 14. 데이터 품질 평가
print("\n\n[14] 데이터 품질 평가")
print("-" * 80)

quality_score = 0
max_score = 0

# 필수 컬럼 완성도
required_cols = ['제목', '재료', '조리순서', 'url']
for col in required_cols:
    max_score += 1
    if col in df.columns:
        completeness = (df[col].notna().sum() / len(df) * 100)
        print(f"  {col:15s} 완성도: {completeness:6.2f}%")
        if completeness > 95:
            quality_score += 1
        elif completeness > 80:
            quality_score += 0.7
        elif completeness > 50:
            quality_score += 0.5

# 메타데이터 풍부도
metadata_cols = ['난이도', '조리시간', '인분', '셰프', '조회수']
for col in metadata_cols:
    max_score += 1
    if col in df.columns:
        completeness = (df[col].notna().sum() / len(df) * 100)
        if completeness > 80:
            quality_score += 1
        elif completeness > 50:
            quality_score += 0.7

final_quality = (quality_score / max_score * 100)
print(f"\n  📊 전체 데이터 품질 점수: {final_quality:.1f}/100점")

if final_quality >= 90:
    grade = "A+ (우수)"
elif final_quality >= 80:
    grade = "A (양호)"
elif final_quality >= 70:
    grade = "B+ (보통)"
else:
    grade = "B (개선 필요)"

print(f"  🏆 등급: {grade}")

# 요약
print("\n\n" + "=" * 80)
print("📊 분석 요약")
print("=" * 80)
print(f"✅ 총 {len(df):,}개의 레시피 데이터 분석 완료")
print(f"✅ {len(df.columns)}개의 컬럼 확인")
print(f"✅ 데이터 품질: {final_quality:.1f}점 ({grade})")
print("=" * 80)

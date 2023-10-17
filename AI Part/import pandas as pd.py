import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 임의의 사용자-책 평점 데이터
data = {
    'User': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4'],
    'Book': ['Book1', 'Book2', 'Book2', 'Book3', 'Book1', 'Book3', 'Book2', 'Book4'],
    'Rating': [5, 4, 3, 2, 4, 5, 4, 5]
}

df = pd.DataFrame(data)

# Pivot table을 사용하여 사용자-책 평점 매트릭스 생성
user_book_rating = df.pivot_table(index='User', columns='Book', values='Rating').fillna(0)

# 코사인 유사도 계산
item_similarity = cosine_similarity(user_book_rating.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_book_rating.columns, columns=user_book_rating.columns)

# 아이템 기반 추천
def get_book_recommendations(book_name, user_rating):
    similar_score = item_similarity_df[book_name] * user_rating
    similar_score = similar_score.sort_values(ascending=False)
    
    return similar_score

print(get_book_recommendations('Book1', 5))


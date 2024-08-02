import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd
import re
from konlpy.tag import Okt
from google.colab import files
import io

# ELECTRA 모델 및 토크나이저 로드
model_name = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Okt 형태소 분석기 초기화
okt = Okt()

# 불용어 리스트
POSITIVE_STOPWORDS = {'아', '진짜', '근데', '다', '전', '저'}
NEGATIVE_STOPWORDS = {'샵검색', '웃음', '다', '다들', '사진', '잘', '이제', '진짜'}

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    morphs = okt.pos(text, stem=True)
    selected_words = [word for word, pos in morphs if pos in ['Noun', 'Adjective', 'Verb', 'Adverb']]
    return ' '.join(selected_words)

def analyze_sentiment_batch(texts, batch_size=16):
    sentiments = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="감정 분석 중"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        batch_sentiments = probabilities[:, 1].cpu().numpy()
        sentiments.extend(batch_sentiments.tolist())
    return sentiments

def parse_kakao_chat(file_content):
    chat_data = []
    current_date = None
    for line in file_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('---------------'):
            current_date = line.strip('-').strip()
        elif line.startswith('[') and '] [' in line:
            try:
                parts = line.split('] [', 1)
                name = parts[0][1:]
                time_message = parts[1]
                time, message = time_message.split(']', 1)
                chat_data.append({'date': current_date, 'time': time, 'name': name, 'message': message.strip()})
            except:
                print(f"파싱 오류 (무시됨): {line}")
    return pd.DataFrame(chat_data)

def get_sentiment_stats(df, max_messages=70000):
    df = df.head(max_messages) if len(df) > max_messages else df
    df['sentiment'] = analyze_sentiment_batch(df['message'].tolist())
    return df.groupby('name')['sentiment'].agg(['mean', 'sum']).sort_values('sum', ascending=False)

def get_top_words(df, sentiment, top_n=10):
    threshold = 0.5
    condition = df['sentiment'] > threshold if sentiment == 'positive' else df['sentiment'] <= threshold
    stopwords = POSITIVE_STOPWORDS if sentiment == 'positive' else NEGATIVE_STOPWORDS
    words = df[condition]['message'].apply(preprocess_text).str.split(expand=True).stack()
    word_counts = words[~words.isin(stopwords)].value_counts()
    return word_counts.head(top_n)

def get_messages_with_word(df, word, sentiment, limit=5):
    threshold = 0.5
    condition = (df['sentiment'] > threshold) if sentiment == 'positive' else (df['sentiment'] <= threshold)
    messages = df[condition & df['message'].str.contains(word, case=False, na=False)]
    return messages[['name', 'message']].head(limit)

try:
    uploaded = files.upload()

    for filename, content in uploaded.items():
        print(f'파일 "{filename}" 처리 중...')
        file_content = io.StringIO(content.decode("utf-8")).read()

        df = parse_kakao_chat(file_content)
        if df.empty:
            print("파싱된 데이터가 없습니다.")
            continue

        print("\n파싱된 데이터 정보:")
        print(df.info())
        print("\n파싱된 데이터 샘플:")
        print(df.head())

        chat_counts = df['name'].value_counts()
        sentiment_stats = get_sentiment_stats(df)

        print(f"\n가장 많은 채팅을 한 사람: {chat_counts.index[0]}")
        print(f"\n가장 긍정적인 표현을 한 사람: {sentiment_stats.index[0]}")
        print(f"가장 부정적인 표현을 한 사람: {sentiment_stats.index[-1]}")

        pos_words = get_top_words(df, 'positive')
        neg_words = get_top_words(df, 'negative')

        print("\n상위 긍정적 단어 (불용어 제외):")
        print(pos_words)
        print("\n상위 부정적 단어 (불용어 제외):")
        print(neg_words)

        print("\n긍정적 단어가 포함된 메시지 예시:")
        for word in pos_words.index[:5]:
            print(f"\n'{word}' 단어가 포함된 메시지:")
            print(get_messages_with_word(df, word, 'positive'))

        print("\n부정적 단어가 포함된 메시지 예시:")
        for word in neg_words.index[:5]:
            print(f"\n'{word}' 단어가 포함된 메시지:")
            print(get_messages_with_word(df, word, 'negative'))

except Exception as e:
    print(f"오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
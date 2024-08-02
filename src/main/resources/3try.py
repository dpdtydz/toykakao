!pip install transformers konlpy sentencepiece

import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import logging
import os
import unicodedata
from shutil import copyfile
from tqdm import tqdm
from google.colab import files
import io
from sklearn.metrics import classification_report

class KoBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, *args, **kwargs):
        super().__init__(vocab_file, *args, **kwargs)
        self.token2idx = dict(self.vocab)
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

# ELECTRA 모델 및 토크나이저 로드
model_name = "monologg/kobert"
tokenizer = KoBertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name, from_pt=True)

# Okt 형태소 분석기 초기화
from konlpy.tag import Okt
okt = Okt()
# 설정
SEQ_LEN = 64
BATCH_SIZE = 32
DATA_COLUMN = "message"
LABEL_COLUMN = "sentiment"

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

def convert_data(data_df):
    global tokenizer

    tokens, masks, segments = [], [], []

    for i in tqdm(range(len(data_df))):
        token = tokenizer.encode(data_df[DATA_COLUMN][i], truncation=True, padding='max_length', max_length=SEQ_LEN)

        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        masks.append(mask)
        segments.append(segment)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)

    return [tokens, masks, segments]

# 모델 구성
token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
bert_outputs = bert_outputs[1]

sentiment_drop = tf.keras.layers.Dropout(0.5)(bert_outputs)
sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(sentiment_drop)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)

# 옵티마이저 및 모델 컴파일
opt = tf.keras.optimizers.Adam(learning_rate=5.0e-5)
sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# 파일 업로드 및 데이터 처리
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

    # 데이터 변환
    data_x = convert_data(df)

    # 감정 분석
    predictions = sentiment_model.predict(data_x)
    df['sentiment'] = predictions

    # 결과 출력
    print("\n감정 분석 결과:")
    print(df[['name', 'message', 'sentiment']].head(20))

    # 사용자별 평균 감정 점수
    user_sentiment = df.groupby('name')['sentiment'].mean().sort_values(ascending=False)
    print("\n사용자별 평균 감정 점수:")
    print(user_sentiment)

    # 가장 긍정적인 메시지
    most_positive = df.loc[df['sentiment'].idxmax()]
    print("\n가장 긍정적인 메시지:")
    print(f"{most_positive['name']}: {most_positive['message']} (점수: {most_positive['sentiment'][0]})")

    # 가장 부정적인 메시지
    most_negative = df.loc[df['sentiment'].idxmin()]
    print("\n가장 부정적인 메시지:")
    print(f"{most_negative['name']}: {most_negative['message']} (점수: {most_negative['sentiment'][0]})")

# 개별 메시지 감정 분석 함수
def analyze_message(message):
    data_x = convert_data(pd.DataFrame([{'message': message}]))
    prediction = sentiment_model.predict(data_x)[0][0]

    if prediction > 0.5:
        print(f"긍정적인 메시지입니다. (점수: {prediction:.2f})")
    else:
        print(f"부정적인 메시지입니다. (점수: {prediction:.2f})")

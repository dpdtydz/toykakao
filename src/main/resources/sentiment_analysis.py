from transformers import pipeline

# 감정 분석 파이프라인 로드
classifier = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = classifier(text)
    return result[0]['label']

if __name__ == "__main__":
    import sys
    text = sys.argv[1]
    print(analyze_sentiment(text))

import os
import re
import whisper
import json
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from .models import Lecture, Question
load_dotenv()


def transcribe_with_whisper(audio_path, model_size="small"):
    try:
        warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        if not os.path.exists(audio_path):
            print(f"❌ 找不到音訊檔案：{audio_path}")
            return None
        model = whisper.load_model(model_size)
        print("✅ Whisper 轉錄開始")
        result = model.transcribe(audio_path, fp16=False)
        return result["text"]
    except Exception as e:
        print(f"❌ Whisper 轉錄錯誤: {e}")
        return None


def dynamic_split(text, min_length=300, max_length=1000):
    text = text.strip()
    if len(text) <= max_length:
        return [text]
    paragraphs = re.split(r'(?<=[。！？])\s*', text)
    chunks, temp = [], ""
    for para in paragraphs:
        if len(temp) + len(para) <= max_length:
            temp += para
        else:
            if len(temp) >= min_length:
                chunks.append(temp.strip())
                temp = para
            else:
                temp += para
    if temp:
        chunks.append(temp.strip())
    return chunks


def create_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not api_key or api_key.strip().upper() == "EMPTY":
        raise ValueError("❌ 請設定 OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=api_base)


def generate_summary_for_chunk(client, chunk, chunk_index, total_chunks):
    prompt = [
        {"role": "system", "content": f"""你是一位專業的繁體中文課程摘要設計師。
請針對第 {chunk_index + 1} 段課程內容進行重點摘要，包含：
- 簡潔內容概述（50–80字）
- 2–4 個學習要點，使用條列式
總字數控制在 150 字內。"""},
        {"role": "user", "content": chunk}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 段落摘要錯誤: {e}")
        return f"第 {chunk_index + 1} 段摘要失敗"


def combine_summaries(client, summaries):
    combined = "\n\n".join([f"段落 {i+1}：{s}" for i, s in enumerate(summaries)])
    prompt = [
        {"role": "system", "content": """你是教育設計專家，請將下列分段摘要統整為完整課程摘要，格式如下：

【課程概述】：說明整體課程內容與重要性（150–200字）
【學習重點】：列出本課程的 4–5 個學習目標（條列）
【完成後收穫】：簡述學生完成課程後能具備的能力（60字內）

請使用繁體中文，避免重複敘述，控制總字數在 400 字內。"""},
        {"role": "user", "content": combined}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 總結錯誤: {e}")
        return "整合摘要失敗"


def generate_quiz(client, summary, count=3):
    prompt = [
        {
            "role": "system",
            "content": f"""你是一位課程出題 AI，請根據以下課程摘要產生 {count} 題選擇題，格式如下：
[
  {{
    "concept": "學習概念",
    "question": "問題內容",
    "options": {{
      "A": "選項 A",
      "B": "選項 B",
      "C": "選項 C",
      "D": "選項 D"
    }},
    "answer": "B",
    "explanation": "正確答案解析"
  }},
  ...
]
請用 JSON 格式回覆，不要有其他文字說明。"""
        },
        {"role": "user", "content": summary}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.5,
            max_tokens=1500
        )
        return json.loads(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ 選擇題產生失敗：{e}")
        return []


def generate_tf_questions(client, summary, count):
    prompt = [
        {
            "role": "system",
            "content": f"""請根據以下課程摘要，設計 {count} 題是非題（True/False），格式如下：
[
  {{
    "concept": "學習概念",
    "question": "問題內容",
    "answer": "True",
    "explanation": "正確答案解析"
  }},
  ...
]
請回傳 JSON 格式資料，不要有其他文字說明。"""
        },
        {"role": "user", "content": summary}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            temperature=0.5
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ 是非題產生失敗：{e}")
        return []


def parse_and_store_questions(summary, quiz_data, lecture, question_type):
    for item in quiz_data:
        question_text = item.get('question', '').strip()
        explanation = item.get('explanation', '').strip()
        answer = item.get('answer', '').strip()

        if question_type == 'mcq':
            options = item.get('options', {})
            Question.objects.create(
                lecture=lecture,
                question_text=question_text,
                option_a=options.get('A'),
                option_b=options.get('B'),
                option_c=options.get('C'),
                option_d=options.get('D'),
                correct_answer=answer,
                explanation=explanation,
                question_type='mcq'
            )
        elif question_type == 'tf':
                Question.objects.create(
                    lecture=lecture,
                    question_text=item.get('question', '').strip(),
                    correct_answer=item.get('answer'),
                    explanation=item.get('explanation', '').strip(),
                    question_type='tf'
            )
        else:
            print(f"⚠️ 未知題型：{question_type}")


def process_audio_and_generate_quiz(lecture_id, num_mcq=3, num_tf=0):
    lecture = Lecture.objects.get(id=lecture_id)
    client = create_openai_client()

    print("🎧 開始語音轉錄")
    transcript = transcribe_with_whisper(lecture.audio_file.path)
    if not transcript:
        return
    lecture.transcript = transcript
    lecture.save()

    print("📝 開始摘要處理")
    chunks = dynamic_split(transcript)
    summaries = [generate_summary_for_chunk(client, c, i, len(chunks)) for i, c in enumerate(chunks)]

    final_summary = combine_summaries(client, summaries)
    lecture.summary = final_summary
    lecture.save()

    print("🧠 開始產生考題")

    if num_mcq > 0:
        mcq_data = generate_quiz(client, final_summary, num_mcq)
        if mcq_data:
            parse_and_store_questions(final_summary, mcq_data, lecture, 'mcq')
        else:
            print("⚠️ 沒有回傳 MCQ 題目")

    if num_tf > 0:
        tf_data = generate_tf_questions(client, final_summary, num_tf)
        if tf_data:
            parse_and_store_questions(final_summary, tf_data, lecture, 'tf')
        else:
            print("⚠️ 沒有回傳 TF 題目")

def process_transcript_and_generate_quiz(lecture, client=None, num_mcq=3, num_tf=0):
    if not client:
        client = create_openai_client()

    transcript = lecture.transcript
    if not transcript:
        print("❌ 無轉錄內容，無法生成摘要與題目")
        return

    print("📝 開始摘要處理")
    chunks = dynamic_split(transcript)
    summaries = [generate_summary_for_chunk(client, c, i, len(chunks)) for i, c in enumerate(chunks)]

    final_summary = combine_summaries(client, summaries)
    lecture.summary = final_summary
    lecture.save()

    print("🧠 開始產生考題")

    if num_mcq > 0:
        mcq_data = generate_quiz(client, final_summary, num_mcq)
        if mcq_data:
            parse_and_store_questions(final_summary, mcq_data, lecture, 'mcq')
        else:
            print("⚠️ 沒有回傳 MCQ 題目")

    if num_tf > 0:
        tf_data = generate_tf_questions(client, final_summary, num_tf)
        if tf_data:
            parse_and_store_questions(final_summary, tf_data, lecture, 'tf')
        else:
            print("⚠️ 沒有回傳 TF 題目")
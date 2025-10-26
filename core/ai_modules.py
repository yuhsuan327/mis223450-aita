import os
import re
import json
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from .models import Lecture, Question

load_dotenv()


def create_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not api_key or api_key.strip().upper() == "EMPTY":
        raise ValueError("❌ 請設定 OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=api_base)


def transcribe_with_whisper(audio_path):
    try:
        if not os.path.exists(audio_path):
            print(f"❌ 找不到音訊檔案：{audio_path}")
            return None
        print("✅ Whisper API 轉錄開始")
        client = create_openai_client()
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return response.text
    except Exception as e:
        print(f"❌ Whisper API 轉錄錯誤: {e}")
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


# 🆕 改善版 generate_quiz 加入自動重試與解析處理
def safe_json_parse(raw: str):
    try:
        return normalize_mcq_payload(json.loads(raw))
    except Exception:
        pass

    m = re.search(r'```json\s*([\s\S]*?)```', raw, re.IGNORECASE)
    if not m:
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
    if m:
        candidate = m.group(1)
        candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return normalize_mcq_payload(json.loads(candidate))
        except Exception as e:
            print("❌ JSON 區塊解析仍失敗：", e)
    print("⚠️ MCQ 原始回應（截斷 500 字）：", raw[:500])
    raise ValueError("模型未回傳合法 JSON")


def normalize_mcq_payload(data):
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON 結構不符合預期")

    cleaned = []
    for it in items:
        try:
            concept = it["concept"].strip()
            question = it["question"].strip()
            options = it["options"]
            answer = it["answer"].strip().upper()
            explanation = it.get("explanation", "").strip()
            if not all(k in options for k in ["A", "B", "C", "D"]):
                continue
            if answer not in {"A", "B", "C", "D"}:
                continue
            cleaned.append({
                "concept": concept,
                "question": question,
                "options": {
                    "A": str(options["A"]),
                    "B": str(options["B"]),
                    "C": str(options["C"]),
                    "D": str(options["D"]),
                },
                "answer": answer,
                "explanation": explanation,
            })
        except Exception:
            continue
    return cleaned


def generate_quiz_with_retry(client, summary, count=3, retries=1):
    system = f"""你是一位課程出題 AI，請根據以下課程摘要產生 {count} 題選擇題。
嚴格且只輸出 JSON「陣列」，不要任何說明、不要加 ```json。每一題物件必須包含：
concept, question, options(A/B/C/D), answer(只能是 A/B/C/D), explanation。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": summary}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        raw = response.choices[0].message.content or ""
        return safe_json_parse(raw)
    except Exception as e:
        print("🔁 第一次解析失敗，改用更嚴格提示重試：", e)

    hard_prompt = summary + f"\n\n請嚴格輸出 {count} 題選擇題，只輸出 JSON 陣列，格式嚴謹。不要有任何說明、```json、前後文字。"
    try:
        messages = [
            {"role": "system", "content": "你只會輸出 JSON。"},
            {"role": "user", "content": hard_prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        raw = response.choices[0].message.content or ""
        return safe_json_parse(raw)
    except Exception as e:
        print("❌ 重試仍失敗：", e)
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
        mcq_data = generate_quiz_with_retry(client, final_summary, num_mcq)
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
        mcq_data = generate_quiz_with_retry(client, final_summary, num_mcq)
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
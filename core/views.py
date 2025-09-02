from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Count, Q
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.views.decorators.http import require_POST
import json
from django.views.decorators.csrf import csrf_exempt
import io, uuid

from .models import Lecture, Question, Student, Submission, Course, Profile
from .forms import (
    UploadLectureForm,
    CourseForm,
    CustomUserCreationForm
)
from .ai_modules import process_audio_and_generate_quiz
import os
from django.conf import settings
import re

TEACHER_CODE = os.getenv('TEACHER_CODE', 'teach2024')  # 預設值仍保留

# ---------- 講次相關 ----------

def upload_lecture(request):
    if request.method == 'POST':
        course_id = request.POST.get('course')
        audio_file = request.FILES.get('audio')

        if course_id and audio_file:
            course = Course.objects.get(id=course_id)
            lecture = Lecture.objects.create(course=course, audio_file=audio_file)
            process_audio_and_generate_quiz(lecture.id)
            return redirect('lecture_detail', lecture.id)
    courses = Course.objects.all()
    return render(request, 'upload.html', {'courses': courses})

def upload_lecture_for_course(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file:
            lecture = Lecture.objects.create(course=course, audio_file=audio_file, title=request.POST.get('title'))
            process_audio_and_generate_quiz(lecture.id)
            return redirect('lecture_detail', lecture.id)
    return render(request, 'upload.html', {'course': course})

def lecture_detail(request, lecture_id):
    lecture = get_object_or_404(Lecture, pk=lecture_id)
    questions = Question.objects.filter(lecture=lecture)
    return render(request, 'lecture_detail.html', {'lecture': lecture, 'questions': questions})

def lecture_list(request):
    query = request.GET.get('q', '')
    lectures = Lecture.objects.all()

    if query:
        lectures = lectures.filter(Q(summary__icontains=query))

    lectures = lectures.order_by('-id')
    for lec in lectures:
        lec.is_ready = bool(lec.summary and lec.question_set.exists())

    # ✅ 加上學生作答狀態
    student = None
    answered_lecture_ids = set()
    if request.user.is_authenticated and hasattr(request.user, 'profile') and request.user.profile.role == 'student':
        try:
            student = Student.objects.get(user=request.user)
            answered_lecture_ids = set(
                Submission.objects.filter(student=student).values_list('question__lecture', flat=True)
            )
        except Student.DoesNotExist:
            pass

    paginator = Paginator(lectures, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'lecture_list.html', {
        'page_obj': page_obj,
        'query': query,
        'answered_lecture_ids': answered_lecture_ids,
    })

def delete_lecture(request, lecture_id):
    lecture = get_object_or_404(Lecture, id=lecture_id)
    lecture.delete()
    return redirect('lecture_list')

# ---------- 測驗相關 ----------

def quiz(request, lecture_id):
    lecture = get_object_or_404(Lecture, pk=lecture_id)
    questions = Question.objects.filter(lecture=lecture)
    
    if request.method == 'POST':
        student = request.user.profile  # 假設有登入並有 profile
        for question in questions:
            student_answer = request.POST.get(str(question.id))
            correct = student_answer == question.correct_answer
            Submission.objects.create(
                student=student,
                question=question,
                student_answer=student_answer,
                is_correct=correct
            )
        return redirect('lecture_detail', lecture_id)
    
    return render(request, 'quiz.html', {'lecture': lecture, 'questions': questions})

# ---------- 學生報告 ----------

@login_required
def student_report(request):
    student = get_object_or_404(Student, user=request.user)
    submissions = Submission.objects.filter(student=student)
    total = submissions.count()
    correct = submissions.filter(is_correct=True).count()
    accuracy = (correct / total * 100) if total else 0
    wrong = submissions.filter(is_correct=False).values('question__question_text').annotate(count=Count('id')).order_by('-count')[:5]
    wrong_count = total - correct


    return render(request, 'student_report.html', {
        'student': student,
        'total': total,
        'correct': correct,
        'wrong_count': wrong_count,
        'accuracy': round(accuracy, 2),
        'wrong': wrong,
    })

def student_weakness_report(request, student_id):
    student = get_object_or_404(Student, user=request.user)
    weaknesses = Submission.objects.filter(student=student, is_correct=False).values('question__concept').annotate(count=Count('id')).order_by('-count')[:5]

    return render(request, 'student_weakness_report.html', {
        'student': student,
        'weaknesses': weaknesses,
    })

# ---------- 課程相關 ----------

def create_course(request):
    if request.method == 'POST':
        form = CourseForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('course_list')
    else:
        form = CourseForm()
    return render(request, 'create_course.html', {'form': form})

def course_list(request):
    courses = Course.objects.all().order_by('-date')
    return render(request, 'course_list.html', {'courses': courses})

def course_detail(request, course_id=None):
    course = get_object_or_404(Course, pk=course_id)

    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        lecture_title = request.POST.get('lecture_title', '').strip()

        # 嘗試安全轉換為整數（預設為 0）
        try:
            num_mcq = int(request.POST.get('num_mcq', 0))
            num_tf = int(request.POST.get('num_tf', 0))
        except ValueError:
            messages.error(request, "❌ 題目數量格式錯誤，請輸入數字。")
            return redirect('course_detail', course_id=course.id)

        # 檢查必要欄位
        if not lecture_title:
            messages.warning(request, "⚠ 請輸入單元名稱。")
            return redirect('course_detail', course_id=course.id)

        if not audio_file:
            messages.warning(request, "⚠ 請選擇要上傳的音檔。")
            return redirect('course_detail', course_id=course.id)

        # 建立講次並觸發 AI 處理
        lecture = Lecture.objects.create(
            course=course,
            audio_file=audio_file,
            title=lecture_title
        )
        print("🧠 呼叫 AI 處理開始")

        process_audio_and_generate_quiz(
            lecture.id,
            num_mcq=num_mcq,
            num_tf=num_tf
        )

        #messages.success(request, f"✅ 成功建立講次《{lecture_title}》並開始產生題目。")
        return redirect('lecture_detail', lecture.id)

    lectures = Lecture.objects.filter(course=course).order_by('-date')
    return render(request, 'course_detail.html', {
        'course': course,
        'lectures': lectures,
    })

# ---------- 使用者角色登入導向 ----------

@login_required
def dashboard(request):
    profile = request.user.profile
    if profile.role == 'teacher':
        return redirect('course_list')
    elif profile.role == 'student':
        return redirect('lecture_list')
    else:
        return redirect('login')

# ---------- 註冊 ----------

TEACHER_CODE = "teach2024"  # 教師驗證碼，可改為環境變數更安全

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        role = request.POST.get('role')
        teacher_code = request.POST.get('teacher_code')

        if form.is_valid():
            user = form.save()

            # 教師驗證邏輯
            if role == 'teacher':
                if teacher_code != TEACHER_CODE:
                    messages.error(request, '教師驗證碼錯誤')
                    user.delete()
                    return render(request, 'register.html', {'form': form})

            # 建立 profile，避免 signal 重複建立衝突
            if hasattr(user, 'profile'):
                user.profile.role = role
                user.profile.save()
            else:
                Profile.objects.create(user=user, role=role)

            login(request, user)
            return redirect('dashboard')  # 登入後導向首頁

    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


# 編輯課程
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST

@login_required
def edit_course(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    if request.method == 'POST':
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            return redirect('course_list')
    else:
        form = CourseForm(instance=course)
    return render(request, 'edit_course.html', {'form': form, 'course': course})

# 刪除課程
@require_POST
@login_required
def delete_course(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    course.delete()
    return redirect('course_list')

from django.shortcuts import render, redirect
from .forms import CourseForm
from core.decorators import teacher_required

@teacher_required
def create_course(request):
    if request.method == 'POST':
        form = CourseForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('course_list')
    else:
        form = CourseForm()  # ← 加這一行才有 GET 回應

    return render(request, 'create_course.html', {'form': form})  # ← 確保最後有 return

@login_required
def dashboard(request):
    profile = request.user.profile
    context = {}

    if profile.role == 'student':
        try:
            student = Student.objects.get(user=request.user)
            context['student_id'] = student.id
        except Student.DoesNotExist:
            context['student_id'] = None

    return render(request, 'dashboard.html', context)

@login_required
def quiz(request, lecture_id):
    lecture = get_object_or_404(Lecture, pk=lecture_id)
    questions = Question.objects.filter(lecture=lecture)

    try:
        student = Student.objects.get(user=request.user)  # ✅ 改這裡，不是用 profile
    except Student.DoesNotExist:
        return HttpResponse("❌ 找不到對應的學生資料，請聯絡管理員。")

    # ✅ 防止學生重複作答
    if Submission.objects.filter(student=student, question__lecture=lecture).exists():
        return HttpResponse("⚠️ 你已經完成這份測驗，請勿重複作答。")

    # ✅ 如果是提交作答
    if request.method == 'POST':
        for question in questions:
            student_answer = request.POST.get(str(question.id))
            correct = student_answer == question.correct_answer
            Submission.objects.create(
                student=student,
                question=question,
                student_answer=student_answer,
                is_correct=correct
            )
        return redirect('lecture_detail', lecture_id)

    # ✅ 顯示測驗表單
    return render(request, 'quiz.html', {
        'lecture': lecture,
        'questions': questions
    })



from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden

@login_required
def edit_summary(request, lecture_id):
    lecture = get_object_or_404(Lecture, pk=lecture_id)

    if request.user.profile.role != 'teacher':
        return HttpResponseForbidden("只有老師可以編輯摘要")

    if request.method == 'POST':
        summary = request.POST.get('summary')
        lecture.summary = summary
        lecture.save()
        messages.success(request, "✅ 摘要已成功更新")
        return redirect('lecture_list')  # 導回課程總覽
    return render(request, 'edit_summary.html', {'lecture': lecture})

@login_required
def view_student_report_by_teacher(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    submissions = Submission.objects.filter(student=student)

    # 📊 基本統計
    total = submissions.count()
    correct = submissions.filter(is_correct=True).count()
    wrong_count = total - correct
    accuracy = round((correct / total * 100), 2) if total else 0

    # ❗ 常錯題（最多五題）
    wrong = (
        submissions
        .filter(is_correct=False)
        .values('question__question_text')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    # 📈 講次正確率資料
    stats = (
        submissions
        .values('question__lecture__title')
        .annotate(
            total=Count('id'),
            correct=Count('id', filter=Q(is_correct=True))
        )
        .order_by('question__lecture__date')
    )
    labels = [s['question__lecture__title'] for s in stats]
    data = [round(s['correct'] / s['total'] * 100, 2) for s in stats]

    # 💡 學習建議
    avg_accuracy = sum(data) / len(data) if data else 0
    if avg_accuracy < 60:
        suggestion = "你的整體正確率偏低，建議加強基本練習。"
    elif avg_accuracy < 85:
        suggestion = "表現尚可，可針對錯誤單元加強複習。"
    else:
        suggestion = "表現優異，請持續保持！"

    return render(request, 'progress_report.html', {
        'student': student,
        'total': total,
        'correct': correct,
        'wrong_count': wrong_count,
        'accuracy': accuracy,
        'wrong': wrong,
        'labels': labels,
        'data': data,
        'suggestion': suggestion,
    })

@login_required
def all_submissions(request):
    if not request.user.profile.role == 'teacher':
        return HttpResponseForbidden("你沒有權限查看此頁面")

    submissions = Submission.objects.select_related('student', 'question', 'question__lecture')
    return render(request, 'all_submissions.html', {
        'submissions': submissions,
    })

@login_required
def lecture_submissions(request, lecture_id):
    lecture = get_object_or_404(Lecture, id=lecture_id)
    submissions = Submission.objects.filter(question__lecture=lecture)

    # 對每個學生統計該講次的作答情況
    students_data = []
    for student in Student.objects.all():
        student_subs = submissions.filter(student=student)
        total = student_subs.count()
        correct = student_subs.filter(is_correct=True).count()
        if total > 0:
            students_data.append({
                'student': student,
                'total': total,
                'correct': correct,
                'accuracy': round(correct / total * 100, 2)
            })

    return render(request, 'lecture_submissions.html', {
        'lecture': lecture,
        'students_data': students_data
    })



# 查看某學生的所有講次作答紀錄
@login_required
def student_submissions(request, student_id):
    student = get_object_or_404(Student, pk=student_id)
    submissions = Submission.objects.filter(student=student).select_related('question__lecture__course')

    summary = {}
    for s in submissions:
        lec = s.question.lecture
        if lec.id not in summary:
            summary[lec.id] = {
                'lecture': lec,
                'total': 0,
                'correct': 0
            }
        summary[lec.id]['total'] += 1
        if s.is_correct:
            summary[lec.id]['correct'] += 1

    result = []
    for item in summary.values():
        total = item['total']
        correct = item['correct']
        accuracy = round((correct / total) * 100, 2) if total > 0 else 0
        result.append({
            'lecture': item['lecture'],
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        })

    return render(request, 'student_submissions.html', {
        'student': student,
        'submissions': result
    })

@login_required
def student_directory(request):
    students = Student.objects.select_related('user').all()
    return render(request, 'student_directory.html', {'students': students})

# ---------- 題目解析頁 ----------


@login_required
def submission_result(request, lecture_id):
    student = get_object_or_404(Student, user=request.user)
    lecture = get_object_or_404(Lecture, id=lecture_id)
    questions = Question.objects.filter(lecture=lecture)
    submissions = Submission.objects.filter(student=student, question__lecture=lecture)

    result = []
    for q in questions:
        sub = submissions.filter(question=q).first()
        result.append({
            'question': q,
            'student_answer': sub.student_answer if sub else None,
            'is_correct': sub.is_correct if sub else None,
        })

    return render(request, 'submission_result.html', {
        'lecture': lecture,
        'results': result
    })

@login_required
def edit_lecture_title(request, lecture_id):
    if request.user.profile.role != 'teacher':
        return HttpResponseForbidden("只有老師可以修改單元名稱")
    
    lecture = get_object_or_404(Lecture, id=lecture_id)

    if request.method == 'POST':
        new_title = request.POST.get('title')
        lecture.title = new_title
        lecture.save()
        #messages.success(request, '單元名稱已更新！')
        return redirect('course_detail', course_id=lecture.course.id)

    # 👇 這段要保留，允許 GET 請求時顯示編輯表單
    return render(request, 'edit_lecture_title.html', {'lecture': lecture})





@login_required
def progress_report(request):
    student = get_object_or_404(Student, user=request.user)
    submissions = Submission.objects.filter(student=student)

    # 📊 基本統計
    total = submissions.count()
    correct = submissions.filter(is_correct=True).count()
    wrong_count = total - correct
    accuracy = round((correct / total * 100), 2) if total else 0

    # ❗ 常錯題（最多五題）
    wrong = (
        submissions
        .filter(is_correct=False)
        .values('question__question_text')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    # 📈 講次正確率資料
    stats = (
        submissions
        .values('question__lecture__title')
        .annotate(
            total=Count('id'),
            correct=Count('id', filter=Q(is_correct=True))
        )
        .order_by('question__lecture__date')
    )
    labels = [s['question__lecture__title'] for s in stats]
    data = [round(s['correct'] / s['total'] * 100, 2) for s in stats]

    # 💡 學習建議
    avg_accuracy = sum(data) / len(data) if data else 0
    if avg_accuracy < 60:
        suggestion = "你的整體正確率偏低，建議加強基本練習。"
    elif avg_accuracy < 85:
        suggestion = "表現尚可，可針對錯誤單元加強複習。"
    else:
        suggestion = "表現優異，請持續保持！"

    return render(request, 'progress_report.html', {
        'student': student,
        'total': total,
        'correct': correct,
        'wrong_count': wrong_count,
        'accuracy': accuracy,
        'wrong': wrong,
        'labels': labels,
        'data': data,
        'suggestion': suggestion,
    })


import tempfile
from django.http import JsonResponse
from django.core.files import File
from pydub import AudioSegment
from .models import Course, Lecture
from .ai_modules import process_audio_and_generate_quiz

@require_POST
def record_and_process(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    audio_file = request.FILES.get('audio_data')
    lecture_title = request.POST.get('lecture_title', '').strip()

    # 題目數量
    try:
        num_mcq = int(request.POST.get('num_mcq', 0))
        num_tf = int(request.POST.get('num_tf', 0))
    except ValueError:
        return JsonResponse({'error': '題目數量格式錯誤'}, status=400)

    if not audio_file or not lecture_title:
        return JsonResponse({'error': '缺少音檔或標題'}, status=400)

    # 儲存 .webm 到暫存檔
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
        for chunk in audio_file.chunks():
            temp_webm.write(chunk)
        webm_path = temp_webm.name

    # 轉換為 .wav
    wav_path = webm_path.replace('.webm', '.wav')
    AudioSegment.from_file(webm_path, format='webm').export(wav_path, format='wav')

    # 建立 Lecture 並儲存音檔
    with open(wav_path, 'rb') as wav_file:
        lecture = Lecture.objects.create(course=course, title=lecture_title)
        lecture.audio_file.save(f"{lecture_title}.wav", File(wav_file))

    # 執行 AI 分析
    process_audio_and_generate_quiz(
        lecture.id,
        num_mcq=num_mcq,
        num_tf=num_tf  # ✅ 加入是非題
    )

    return JsonResponse({'success': True})

def submission_detail(request, lecture_id, student_id):
    lecture = get_object_or_404(Lecture, id=lecture_id)
    student = get_object_or_404(Student, id=student_id)
    submissions = Submission.objects.filter(student=student, question__lecture=lecture).select_related('question')

    return render(request, 'submission_result.html', {
        'lecture': lecture,
        'student': student,
        'results': submissions,
    })



from .ai_modules import process_transcript_and_generate_quiz, process_audio_and_generate_quiz,transcribe_with_whisper
import uuid
@csrf_exempt
def live_chunk_upload(request):
    if request.method == 'POST':
        audio_chunk = request.FILES.get('audio_chunk')
        lecture_title = request.POST.get('lecture_title')
        course_id = request.POST.get('course_id')

        if not (audio_chunk and lecture_title and course_id):
            return JsonResponse({'error': '缺少必要欄位'}, status=400)

        lecture, _ = Lecture.objects.get_or_create(title=lecture_title, course_id=course_id)

        # ✅ 注意：Windows 下不要 delete=True，Whisper會讀不到檔案
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            for chunk in audio_chunk.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name  # 不要用 tmp.name 以外的東西

        print("✅ Whisper 轉錄開始")
        try:
            text = transcribe_with_whisper(tmp_path)
            print("✅ Whisper 轉錄成功")
        except Exception as e:
            print("❌ Whisper 轉錄錯誤:", e)
            text = ""

        if text:
            lecture.transcript = (lecture.transcript or '') + '\n' + text
            lecture.save()

        return JsonResponse({'transcript': text, 'lecture_id': lecture.id})
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def finalize_transcript_summary_quiz(request, lecture_id):
    lecture = Lecture.objects.get(id=lecture_id)

    if not lecture.transcript:
        return JsonResponse({"error": "Transcript not found"}, status=404)

    # ✅ 接收前端傳來的題目數量
    try:
        data = json.loads(request.body)
        num_mcq = int(data.get("num_mcq", 3))
        num_tf = int(data.get("num_tf", 0))
    except Exception as e:
        print("❌ 題目數量解析錯誤", e)
        num_mcq, num_tf = 3, 0  # 預設值

    process_transcript_and_generate_quiz(lecture, num_mcq=num_mcq, num_tf=num_tf)
    return JsonResponse({"status": "ok"})
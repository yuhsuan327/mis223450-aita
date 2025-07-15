from django.contrib.auth.models import User
from django.db import models

class Course(models.Model):
    name = models.CharField(max_length=200)
    date = models.DateField(null=True, blank=True)  # 新增：日期欄位
    description = models.TextField(blank=True)       # 新增：簡介欄位

    def __str__(self):
        return self.name

class Lecture(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, null=True)
    audio_file = models.FileField(upload_to='lectures/')
    transcript = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    quiz_generated = models.BooleanField(default=False)

class Question(models.Model):
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE)
    question_text = models.TextField()
    option_a = models.CharField(max_length=200, blank=True, null=True)
    option_b = models.CharField(max_length=200, blank=True, null=True)
    option_c = models.CharField(max_length=200, blank=True, null=True)
    option_d = models.CharField(max_length=200, blank=True, null=True)
    correct_answer = models.CharField(max_length=200)  # 支援文字答案
    explanation = models.TextField()
    concept = models.CharField(max_length=100, default="未分類")
    question_type = models.CharField(
        max_length=20,
        choices=[
            ('mcq', '選擇題'),
            ('tf', '是非題'),
        ],
        default='mcq'
    )

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

class Submission(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    student_answer = models.CharField(max_length=1)
    is_correct = models.BooleanField()
    submitted_at = models.DateTimeField(auto_now_add=True)

#class Profile(models.Model):
#    user = models.OneToOneField(User, on_delete=models.CASCADE)
#    role = models.CharField(max_length=10, choices=[('teacher', '老師'), ('student', '學生')])

#    def __str__(self):
#        return f"{self.user.username} - {self.role}"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=[('teacher', '老師'), ('student', '學生')])

    def is_teacher(self):
        return self.role == 'teacher'

    def is_student(self):
        return self.role == 'student'

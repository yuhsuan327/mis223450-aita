from django.contrib import admin
from .models import Course, Lecture, Question, Student, Submission, Profile

admin.site.register(Course)
admin.site.register(Lecture)
admin.site.register(Question)
admin.site.register(Student)
admin.site.register(Submission)
admin.site.register(Profile)
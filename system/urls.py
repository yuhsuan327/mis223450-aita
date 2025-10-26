from django.contrib import admin
from django.urls import path
from core import views
from django.contrib.auth import views as auth_views
from core.views import register
from django.contrib.auth.views import LogoutView



urlpatterns = [
    path('admin/', admin.site.urls),
    #path('', views.upload_lecture, name='upload_lecture'),
    path('', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('lecture/<int:lecture_id>/', views.lecture_detail, name='lecture_detail'),
    path('lecture/<int:lecture_id>/quiz/', views.quiz, name='quiz'),
    path("student/report/", views.student_report, name="student_report"),
    path('student/<int:student_id>/weakness/', views.student_weakness_report, name='student_weakness_report'),
    path('lectures/', views.lecture_list, name='lecture_list'),
    path('lecture/<int:lecture_id>/delete/', views.delete_lecture, name='delete_lecture'),
    path('courses/new/', views.create_course, name='create_course'),
    path('course/<int:course_id>/', views.course_detail, name='course_detail'),
    path('lectures/', views.lecture_list, name='lecture_list'),
    path('course/', views.course_list, name='course_list'),
    path('courses/<int:course_id>/upload/', views.upload_lecture_for_course, name='upload_lecture_for_course'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', LogoutView.as_view(template_name='logout.html'), name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),  # 登入後的主頁
    path('register/', views.register, name='register'),
    path('course/<int:course_id>/edit/', views.edit_course, name='edit_course'),
    path('course/<int:course_id>/delete/', views.delete_course, name='delete_course'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('lecture/<int:lecture_id>/edit_summary/', views.edit_summary, name='edit_summary'),
    path('submissions/', views.all_submissions, name='all_submissions'),
    path('lecture/<int:lecture_id>/submissions/', views.lecture_submissions, name='lecture_submissions'),
    path('student/<int:student_id>/submissions/', views.student_submissions, name='student_submissions'),
    path('student/directory/', views.student_directory, name='student_directory'),
    path('lecture/<int:lecture_id>/result/', views.submission_result, name='submission_result'),
    path('lecture/<int:lecture_id>/edit_title/', views.edit_lecture_title, name='edit_lecture_title'),
    path('progress/', views.progress_report, name='progress_report'),
    path('course/<int:course_id>/record/', views.record_and_process, name='record_and_process'),
    path('lecture/<int:lecture_id>/student/<int:student_id>/', views.submission_detail, name='submission_detail'),
    #path('progress/<int:student_id>/', views.student_progress_report_admin, name='student_progress_report_admin'),
    path('student/<int:student_id>/report/', views.view_student_report_by_teacher, name='teacher_student_report'),
    path("api/live_chunk_upload/", views.live_chunk_upload, name="live_chunk_upload"),
    path("api/finalize_transcript_summary_quiz/<int:lecture_id>/", views.finalize_transcript_summary_quiz, name="finalize_transcript_summary_quiz"),

]

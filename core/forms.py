from django import forms
from .models import Lecture
from .models import Course
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile

class UploadLectureForm(forms.ModelForm):
    class Meta:
        model = Lecture
        fields = ['course', 'audio_file']

class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ['name', 'date', 'description']
        labels = {
            'name': '課程名稱',
            'date': '課程日期',
            'description': '課程簡介說明',
        }
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '請輸入課程名稱',
            }),
            'date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '選填：簡單說明課程內容',
                'rows': 3,
            }),
        }

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def clean_username(self):
        username = self.cleaned_data.get("username")
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("此帳號已被使用，請更換一個")
        return username

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control'})

class LectureForm(forms.ModelForm):
    class Meta:
        model = Lecture
        fields = ['title']
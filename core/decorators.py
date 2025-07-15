from django.shortcuts import redirect
from functools import wraps
from django.contrib.auth.decorators import login_required

def teacher_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or request.user.profile.role != 'teacher':
            return redirect('dashboard')  # 無權限者導回主頁
        return view_func(request, *args, **kwargs)
    return wrapper

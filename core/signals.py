# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Profile, Student

#@receiver(post_save, sender=User)
#def create_profile(sender, instance, created, **kwargs):
#    if created:
#        Profile.objects.create(user=instance)

#@receiver(post_save, sender=User)
#def create_profile_and_related(sender, instance, created, **kwargs):
#    if created:
        # 若 Profile 已透過 views.register 創建，這邊跳過
#        if hasattr(instance, 'profile'):
#            role = instance.profile.role
#        else:
#            role = 'student'  # 預設為學生

#       profile = Profile.objects.create(user=instance, role=role)

#        if role == 'student':
#            Student.objects.create(user=instance, name=instance.username, email=instance.email)


from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Profile

#@receiver(post_save, sender=User)
#def create_user_profile(sender, instance, created, **kwargs):
#    if created and not hasattr(instance, 'profile'):
#        Profile.objects.create(user=instance, role='student')  # 預設學生

@receiver(post_save, sender=User)
def create_profile_and_student(sender, instance, created, **kwargs):
    if created:
        profile = Profile.objects.create(user=instance, role='student')

        # 自動建立 Student
        Student.objects.create(user=instance, name=instance.username, email=instance.email)

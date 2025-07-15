# core/templatetags/quiz_tags.py
from django import template

register = template.Library()

@register.filter
def get_option(question, key):
    return getattr(question, f"option_{key.lower()}")

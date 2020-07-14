from django import forms


class SignInForm(forms.Form):
    login = forms.CharField(max_length=100)
    password = forms.CharField(max_length=100)

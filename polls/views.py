# Create your views here.
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import loader

from polls.forms import *


def index(request):
    template = loader.get_template('index.html')
    response_body = template.render({'current_user': request.user})
    return HttpResponse(response_body)


def sign_up(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            login(request, user)
            return redirect("index")

        else:
            for msg in form.error_messages:
                print(form.error_messages[msg])

            return render(request=request,
                          template_name="register.html",
                          context={"form": form, 'current_user': request.user})
    form = UserCreationForm
    return render(request=request,
                  template_name="register.html",
                  context={"form": form, 'current_user': request.user})


def sign_in(request):
    if request.method == 'POST':
        form = SignInForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_pwd = form.cleaned_data.get('password')
            user = authenticate(username=username, password=raw_pwd)
            login(request, user)
            return redirect('index')
    else:
        form = SignInForm()
    return render(request, 'login.html', {'form': form, 'current_user': request.user})


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("logged_out")


def login_request(request):
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}")
                return redirect('/')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request=request,
                  template_name="login.html",
                  context={"form": form, 'current_user': request.user})

# Create your views here.
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import loader

from backend.nnb import *
from backend.tools import *
from polls.models import NeuralNetworks


def index(request):
    template = loader.get_template('index.html')
    response_body = template.render({'current_user': request.user})
    return HttpResponse(response_body)


def sign_up(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        print(form.is_valid())
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data['password1']
            login(request, user)
            return redirect("index")
        else:
            for msg in form.error_messages:
                print(form.error_messages[msg])

            return render(request=request,
                          template_name="register.html",
                          context={"form": form, 'current_user': request.user})
    form = UserCreationForm()
    return render(request=request,
                  template_name="register.html",
                  context={"form": form, 'current_user': request.user})


@login_required
def logout_request(request):
    usr = request.user
    logout(request)
    messages.info(request, "Logged out successfully!")
    return render(request=request,
                  template_name="logout.html",
                  context={'old_user': usr, 'current_user': request.user})


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


def search_nnb(request):
    query_results = NeuralNetworks.objects.all()
    if request.POST.get('Find'):
        print(request.POST.get('Func'))
        query_results = NeuralNetworks.objects.filter(problem=request.POST.get('Func'))
    return render(request=request,
                  template_name="search_nnb.html",
                  context={'current_user': request.user, 'query_results': query_results})


def create_nnb(request):
    if request.POST.get('build'):
        print('click')
        expr = request.POST.get('expression')
        ls = request.POST.get('linspace')
        t = request.POST.get('time')
        neurons, acts = get_random_model_scheme()
        nnb = NeuralNetworkDesigner(neurons, acts, ls, translate_pythonic(expr))
        res = nnb.simulated_annealing(t)
        print(res)
        new_record = NeuralNetworks(creator=request.user, problem=normalize_pythonic(expr), mse=res[0],
                                    neuron_list=res[1],
                                    activation_list=res[2])
        new_record.save()
        return render(request=request,
                      template_name="result.html",
                      context={'current_user': request.user, 'data': nnb.data})
    return render(request=request,
                  template_name="configure.html",
                  context={'current_user': request.user})

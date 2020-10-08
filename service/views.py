# Create your views here.
import base64
import io
import pickle

import PIL
import pylab
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.db import connection
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import loader
from matplotlib.pyplot import barh, yticks, xlabel, ylabel, title, subplots_adjust
from numpy.ma import arange

from backend.nnb import *
from backend.tools import *
from service.models import NeuralNetworks, Graphs


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
    query_results = Graphs.objects.select_related('nnb_id')
    if request.POST.get('Find'):
        print(request.POST.get('Func'))
        query_results = Graphs.objects.select_related('nnb_id').filter(nnb_id__problem=request.POST.get('Func'))
    return render(request=request,
                  template_name="search_nnb.html",
                  context={'current_user': request.user, 'query_results': query_results})


def create_nnb(request):
    if request.POST.get('build'):
        expr, ls, t, initial_temp, scale, rs_step, neurons, acts = get_parameters_configuration(request)
        neurons, acts, rnd_model = choose_model_creation(acts, neurons)
        new_nnb = NeuralNetworkDesigner(neurons, acts, ls, translate_pythonic(expr))
        new_nnb.data['random_model'] = rnd_model
        setup_initial_parameters(new_nnb, scale, initial_temp, rs_step)
        res = new_nnb.simulated_annealing(t) if len(
            rs_step) == 0 and not rs_step.isdigit() else new_nnb.simulated_annealing(t, step_limit=int(rs_step))
        new_record = NeuralNetworks(creator=request.user, problem=normalize_pythonic(expr), mse=res[0],
                                    neuron_list=res[1], activation_list=res[2])
        new_record.save()
        xp, yp, yhp = new_nnb.data['x_plot'], new_nnb.data['y_plot'], new_nnb.data['yhat_plot']
        ixp, iyhp = new_nnb.data['initial_xplot'], new_nnb.data['initial_yhat']
        graphs_record = Graphs(nnb_id=new_record, xplot=list(xp.flatten()), yplot=list(yp.flatten()),
                               yhatplot=list(yhp.flatten()), initial_yhatplot=list(iyhp.flatten()),
                               initial_xplot=list(ixp.flatten()))
        graphs_record.save()
        print(new_nnb.data['best_neurons'], new_nnb.data['best_activations'])
        return render(request=request,
                      template_name="result.html",
                      context={'current_user': request.user, 'data': new_nnb.data, 'graph_id': graphs_record.id})
    return render(request=request,
                  template_name="configure.html",
                  context={'current_user': request.user})


def grafico(request, graph_id):
    query_results = Graphs.objects.all()
    query_results = query_results.filter(id=graph_id).first()
    fig, axs = pyplot.subplots(2, 1)
    size = 0.6
    axs[1].scatter(query_results.xplot, query_results.yplot, label='Actual', s=size)
    axs[1].scatter(query_results.xplot, query_results.yhatplot, label='Predicted', s=size)
    axs[1].set_title('Final model')
    axs[0].scatter(query_results.xplot, query_results.yplot, label='Actual', s=size)
    axs[0].scatter(query_results.initial_xplot, query_results.initial_yhatplot, label='Predicted', s=size)
    axs[0].set_title('Initial model')

    buffer = io.BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    graphIMG = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    graphIMG.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="Image/png")


def choose_model_creation(acts, neurons):
    random_flag = True
    if len(acts.split(',')) != 0 and len(acts.split(',')) == len(neurons.split(',')):
        try:
            neurons, acts = [int(x) for x in neurons.split(',')], [str(x) for x in acts.split(',')]
            random_flag = False
        except ValueError:
            neurons, acts = get_random_model_scheme()

    else:
        neurons, acts = get_random_model_scheme()
    return neurons, acts, random_flag


def setup_initial_parameters(new_nnb, scale, initial_temp, rs_step):
    if scale.replace('.', '', 1).isdigit():
        new_nnb.data['scale'] = scale
    if initial_temp.replace('.', '', 1).isdigit():
        new_nnb.data['initial_temperature'] = initial_temp
    if rs_step.replace('.', '', 1).isdigit():
        new_nnb.data['rs_step'] = rs_step


def get_parameters_configuration(request):
    expr = request.POST.get('expression')
    ls = request.POST.get('linspace')
    t = request.POST.get('time')
    initial_temp = request.POST.get('initial_temperature')
    scale = request.POST.get('scale')
    rs_step = request.POST.get('reset_step')
    acts = request.POST.get('activation_list')
    neurons = request.POST.get('neuronsq_list')
    return expr, ls, t, initial_temp, scale, rs_step, neurons, acts

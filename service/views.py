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
    query_results = NeuralNetworks.objects.all()
    if request.POST.get('Find'):
        print(request.POST.get('Func'))
        query_results = NeuralNetworks.objects.filter(problem=request.POST.get('Func'))
    return render(request=request,
                  template_name="search_nnb.html",
                  context={'current_user': request.user, 'query_results': query_results})


def create_nnb(request):
    if request.POST.get('build'):
        expr = request.POST.get('expression')
        ls = request.POST.get('linspace')
        t = request.POST.get('time')
        neurons, acts = get_random_model_scheme()
        new_nnb = NeuralNetworkDesigner(neurons, acts, ls, translate_pythonic(expr))
        res = new_nnb.simulated_annealing(t)
        print(res)
        new_record = NeuralNetworks(creator=request.user, problem=normalize_pythonic(expr), mse=res[0],
                                    neuron_list=res[1],
                                    activation_list=res[2])
        new_record.save()
        xp, yp, yhp = new_nnb.data['x_plot'], new_nnb.data['y_plot'], new_nnb.data['yhat_plot']
        # xpn = base64.b64encode(pickle.dumps(xp))
        # ypn = base64.b64encode(pickle.dumps(yp))
        # yhpn = base64.b64encode(pickle.dumps(yhp))
        print(xp, yp, yhp)
        print(new_record.id)
        graphs_record = Graphs(nnb_id=new_record, xplot=xp.tolist(), yplot=yp.tolist(), yhatplot=yhp.tolist())
        graphs_record.save()
        return render(request=request,
                      template_name="result.html",
                      context={'current_user': request.user, 'data': new_nnb.data})
    return render(request=request,
                  template_name="configure.html",
                  context={'current_user': request.user})


def grafico(request):
    query_results = Graphs.objects.all()
    # pyplot.scatter(x_plot, y_plot, label='Actual')
    # pyplot.scatter(x_plot, yhat_plot, label='Predicted')
    # pyplot.title('MSE: %.3f' % mse)
    buffer = io.BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    graphIMG = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    graphIMG.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="Image/png")

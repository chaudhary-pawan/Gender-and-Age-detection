from django.shortcuts import render
from .ml_utils import predict_gender_age

def index(request):
    prediction = None

    if request.method == "POST":
        height = float(request.POST.get("height"))
        weight = float(request.POST.get("weight"))
        voice = float(request.POST.get("voice_pitch"))
        bmi = float(request.POST.get("bmi"))

        result = predict_gender_age(height, weight, voice, bmi)

        prediction = {
            "gender": result["gender"],
            "gender_confidence": result["gender_confidence"],
            "age_group": result["age_group"],
            "age_confidence": result["age_confidence"]
        }

    return render(request, "index.html", {"prediction": prediction})


def home(request):
    return render(request, "home.html")

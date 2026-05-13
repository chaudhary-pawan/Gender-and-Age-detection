from django.shortcuts import render
from .ml_utils import predict_gender_age

def index(request):
    prediction = None
    error = None
    form_data = {"height": "", "weight": "", "voice_pitch": "", "bmi": ""}

    if request.method == "POST":
        form_data = {
            "height": request.POST.get("height", "").strip(),
            "weight": request.POST.get("weight", "").strip(),
            "voice_pitch": request.POST.get("voice_pitch", "").strip(),
            "bmi": request.POST.get("bmi", "").strip(),
        }

        try:
            height = float(form_data["height"])
            weight = float(form_data["weight"])
            voice = float(form_data["voice_pitch"])
            bmi = float(form_data["bmi"]) if form_data["bmi"] else None
        except (TypeError, ValueError):
            error = "Please enter numeric values for Height, Weight, Voice Pitch, and BMI."
        else:
            if height <= 0:
                error = "Height must be greater than zero."
            elif weight <= 0:
                error = "Weight must be greater than zero."
            elif voice <= 0:
                error = "Voice Pitch must be greater than zero."
            else:
                if bmi is None or bmi <= 0:
                    bmi = weight / ((height / 100) ** 2)

                form_data["bmi"] = f"{bmi:.2f}"
                result = predict_gender_age(height, weight, voice, bmi)
                prediction = {
                    "gender": result["gender"],
                    "gender_confidence": result["gender_confidence"],
                    "age_group": result["age_group"],
                    "age_confidence": result["age_confidence"],
                }

    return render(request, "index.html", {"prediction": prediction, "error": error, "form_data": form_data})


def home(request):
    return render(request, "home.html")

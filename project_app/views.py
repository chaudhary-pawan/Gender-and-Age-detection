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
            bmi = float(form_data["bmi"]) if form_data["bmi"] else 0

            if height <= 0 or weight <= 0:
                raise ValueError("Height and weight must be greater than zero.")
            if voice <= 0:
                raise ValueError("Voice pitch must be greater than zero.")

            if bmi <= 0:
                bmi = weight / ((height / 100) ** 2)

            form_data["bmi"] = f"{bmi:.2f}"
            result = predict_gender_age(height, weight, voice, bmi)
            prediction = {
                "gender": result["gender"],
                "gender_confidence": result["gender_confidence"],
                "age_group": result["age_group"],
                "age_confidence": result["age_confidence"],
            }
        except (TypeError, ValueError):
            error = "Please enter valid positive values for Height, Weight, and Voice Pitch."

    return render(request, "index.html", {"prediction": prediction, "error": error, "form_data": form_data})


def home(request):
    return render(request, "home.html")

from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="home"),
               path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
               path("UpdateProfileAction", views.UpdateProfileAction, name="UpdateProfileAction"),
               path("UpdateProfile.html", views.UpdateProfile, name="UpdateProfile"),
               path("DetectionAction", views.DetectionAction, name="DetectionAction"),
               path("Detection.html", views.Detection, name="Detection"),
               path("Register.html", views.Register, name="Register"),
               path("RegisterAction", views.RegisterAction, name="RegisterAction")	       
]

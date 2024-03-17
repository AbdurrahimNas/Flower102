from django import forms
from Flower102Interface.models import Flowers

class FlowerForm(forms.ModelForm):

    class Meta:
        model = Flowers
        fields = "__all__"
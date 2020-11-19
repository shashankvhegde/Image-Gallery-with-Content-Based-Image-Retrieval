from django import forms

class FilterForm(forms.Form):
	
	filterText = forms.CharField(max_length = 200, required=False, 
		widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search'}))
